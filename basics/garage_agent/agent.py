# LiveKit Agents - Garage Agent (Korrigiert basierend auf Medical Agent)
import logging
import os
import httpx
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from difflib import SequenceMatcher

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

class ConversationState(Enum):
    """State Machine für Konversationsphasen"""
    GREETING = "greeting"
    AWAITING_REQUEST = "awaiting_request"
    COLLECTING_IDENTIFIER = "collecting_identifier"
    SEARCHING = "searching"
    PROVIDING_INFO = "providing_info"

@dataclass
class VehicleContext:
    """Kontext für Fahrzeug-Identifikation"""
    fahrzeug_id: Optional[str] = None  # z.B. F001, F002
    owner_name: Optional[str] = None
    license_plate: Optional[str] = None
    attempts: int = 0

    def has_identifier(self) -> bool:
        """Prüft ob eine Identifikation vorhanden ist"""
        return bool(self.fahrzeug_id or self.owner_name or self.license_plate)

    def get_search_query(self) -> Optional[str]:
        """Gibt die beste Suchanfrage zurück"""
        if self.fahrzeug_id:
            return self.fahrzeug_id
        elif self.owner_name:
            return self.owner_name
        elif self.license_plate:
            return self.license_plate
        return None

    def reset(self):
        """Reset des Kontexts"""
        self.fahrzeug_id = None
        self.owner_name = None
        self.license_plate = None
        self.attempts = 0

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    qdrant_url: str = "http://172.16.0.108:6333"
    current_vehicle_data: Optional[Dict[str, Any]] = None
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    vehicle_context: VehicleContext = field(default_factory=VehicleContext)
    last_search_query: Optional[str] = None
    active_vehicle: Optional[str] = None


class IdentifierExtractor:
    """Extrahiert und validiert Fahrzeug-Identifikatoren"""

    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()

        # Check for specific garage queries FIRST
        if any(word in input_lower for word in ["service", "reparatur", "wartung", "inspektion"]):
            return {"intent": "service_query", "data": user_input}

        if any(word in input_lower for word in ["rechnung", "kosten", "preis"]):
            return {"intent": "invoice_query", "data": user_input}

        if any(word in input_lower for word in ["problem", "defekt", "kaputt"]):
            return {"intent": "problem_report", "data": user_input}

        # Fahrzeug-ID (F001, F002, etc.) - Check even in longer sentences
        fahrzeug_id_match = re.search(r'\b(F\d{3})\b', user_input.upper())
        if fahrzeug_id_match:
            return {"intent": "fahrzeug_id", "data": fahrzeug_id_match.group(1)}

        # Korrigiere Sprache-zu-Text Fehler bei Fahrzeug-IDs
        patterns = [
            r'f\s*null\s*null\s*(\w+)',
            r'fahrzeug\s*null\s*null\s*(\w+)',
            r'f\s*0\s*0\s*(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, input_lower)
            if match:
                number = match.group(1)
                # Deutsche Zahlwörter zu Ziffern
                number_map = {
                    'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4',
                    'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8',
                    'neun': '9', 'null': '0', 'zehn': '10'
                }
                if number in number_map:
                    number = number_map[number]
                corrected_id = f"F{number.zfill(3)}"
                return {"intent": "fahrzeug_id", "data": corrected_id}

        # License plate detection (Swiss format)
        license_patterns = [
            r'\b([A-Z]{2})\s*(\d{4,6})\b',
            r'\b([A-Z]{2})\s+(\d{3}\s*\d{3})\b'
        ]
        
        for pattern in license_patterns:
            match = re.search(pattern, user_input.upper())
            if match:
                license_plate = f"{match.group(1)} {match.group(2).replace(' ', '')}"
                return {"intent": "license_plate", "data": license_plate}

        # Greetings - but only if no other important info is present
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        has_greeting = any(g in input_lower for g in greetings)

        # Name detection
        name_patterns = [
            r"(?:mein name ist|ich bin|ich heiße)\s+([A-Za-zäöüÄÖÜß\s]+)",
            r"(?:herr|frau)\s+([A-Za-zäöüÄÖÜß]+(?:\s+[A-Za-zäöüÄÖÜß]+)?)"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name.lower() not in ["der", "die", "das"]:
                    return {"intent": "owner_name", "data": name}

        # If we found a greeting but no other data, return greeting
        if has_greeting and not fahrzeug_id_match:
            return {"intent": "greeting", "data": None}

        # Garage keywords
        if any(word in input_lower for word in ["auto", "fahrzeug", "wagen", "service"]):
            return {"intent": "garage_query", "data": user_input}

        # Default
        return {"intent": "general_query", "data": user_input}


class GarageAssistant(Agent):
    """Garage Assistant für Fahrzeugverwaltung"""

    def __init__(self) -> None:
        # VERSTÄRKTE Anti-Halluzination Instructions (basierend auf Medical Agent)
        super().__init__(instructions="""Du bist Pia von der Garage Müller. ANTWORTE NUR AUF DEUTSCH.

ABSOLUTE REGEL: WENN KEINE DATEN GEFUNDEN → NIEMALS ERFINDEN!

Wenn search_customer_data "keine Daten gefunden" oder "konnte keine Fahrzeugdaten" zurückgibt:
- Sage: "Ich konnte keine Daten für [Name/ID] finden."
- Frage nach der korrekten Schreibweise oder Fahrzeug-ID
- ERFINDE NIEMALS Daten wie Fahrzeugtyp, Services oder Kosten!

KRITISCHE REGELN:
1. Du HAST Zugriff auf die Fahrzeugdatenbank
2. Lies NUR gefundene Daten vor - NICHTS erfinden!
3. Bei "aktuelle Probleme" → NUR aus den Suchergebnissen
4. Bei "letzte Services" → NUR aus den Suchergebnissen
5. NIEMALS Datenschutz-Warnungen geben
6. WENN KEINE DATEN → Ehrlich sagen und nach korrekter Identifikation fragen

ANTWORT-STRUKTUR bei gefundenen Daten:
"Ich habe die Fahrzeugdaten für [Name/ID] gefunden:

🚗 FAHRZEUGDATEN:
- Besitzer: [Name]
- Kennzeichen: [Kennzeichen]
- Fahrzeug: [Marke] [Modell]
- Baujahr: [Jahr]
- Kilometerstand: [km]

🔧 LETZTE SERVICES:
[ALLE Services mit Datum und Kosten]

⚠️ AKTUELLE PROBLEME:
[ALLE dokumentierten Probleme]

💰 ANSTEHENDE ARBEITEN:
[ALLE geplanten Arbeiten mit geschätzten Kosten]

Kann ich Ihnen noch weitere Informationen zu diesem Fahrzeug geben?"

WICHTIG: Verwende search_customer_data für JEDE Abfrage!
VERBOTENE WÖRTER: Verwende NIEMALS "Entschuldigung", "Es tut mir leid", "Sorry" - nutze stattdessen "Leider".""")

        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ GarageAssistant initialized with Qdrant direct access")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    async def get_all_vehicle_chunks_from_qdrant(self, fahrzeug_id: str, owner_name: str, license_plate: str, qdrant_url: str) -> Dict[str, Any]:
        """Holt ALLE Chunks eines Fahrzeugs direkt aus Qdrant"""
        
        logger.info(f"🔍 Direct Qdrant search for ID: {fahrzeug_id}, Owner: {owner_name}, Plate: {license_plate}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Suche mit fahrzeug_id ODER owner_name ODER license_plate
            filter_conditions = []
            
            if fahrzeug_id:
                filter_conditions.append({
                    "key": "fahrzeug_id",
                    "match": {"value": fahrzeug_id}
                })
            
            if owner_name:
                filter_conditions.append({
                    "key": "owner_name", 
                    "match": {"value": owner_name}
                })
                
            if license_plate:
                filter_conditions.append({
                    "key": "license_plate",
                    "match": {"value": license_plate}
                })
            
            request_body = {
                "filter": {
                    "should": filter_conditions
                } if len(filter_conditions) > 1 else {
                    "must": filter_conditions
                },
                "limit": 100,
                "with_payload": True,
                "with_vector": False
            }
            
            response = await client.post(
                f"{qdrant_url}/collections/garage_management/points/scroll",
                json=request_body
            )
            
            if response.status_code == 200:
                data = response.json()
                points = data.get("result", {}).get("points", [])
                
                # WENN KEINE ERGEBNISSE: Ähnlichkeitssuche für Namen
                if len(points) == 0 and owner_name:
                    logger.warning(f"⚠️ No exact match for '{owner_name}', trying similarity search...")
                    
                    # Suche nach ähnlichen Namen
                    all_response = await client.post(
                        f"{qdrant_url}/collections/garage_management/points/scroll",
                        json={
                            "limit": 1000,
                            "with_payload": True,
                            "with_vector": False
                        }
                    )
                    
                    if all_response.status_code == 200:
                        all_points = all_response.json().get("result", {}).get("points", [])
                        
                        # Finde ähnliche Namen
                        best_match = None
                        best_score = 0
                        
                        for point in all_points:
                            point_name = point.get("payload", {}).get("owner_name", "")
                            if point_name:
                                # Berechne Ähnlichkeit
                                score = SequenceMatcher(None, owner_name.lower(), point_name.lower()).ratio()
                                if score > best_score and score > 0.8:  # 80% Ähnlichkeit
                                    best_score = score
                                    best_match = point_name
                        
                        if best_match:
                            logger.info(f"✅ Found similar name: '{best_match}' (score: {best_score:.2f})")
                            # Suche mit korrigiertem Namen
                            return await self.get_all_vehicle_chunks_from_qdrant(
                                fahrzeug_id="",
                                owner_name=best_match,
                                license_plate="",
                                qdrant_url=qdrant_url
                            )
                
                logger.info(f"📦 Found {len(points)} chunks from Qdrant")
                
                # Extrahiere Fahrzeugdaten aus dem ersten (und einzigen) Punkt
                if points:
                    return points[0].get("payload", {})
                else:
                    return {}
            else:
                logger.error(f"❌ Qdrant error: {response.status_code}")
                return {}

    @function_tool
    async def search_customer_data(self,
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank basierend auf Fahrzeug-ID, Namen oder Kennzeichen.
        
        Args:
            query: Suchbegriff (Fahrzeug-ID, Name oder Autokennzeichen)

        Returns:
            Gefundene Fahrzeugdaten oder Fehlermeldung
        """
        logger.info(f"🔍 Original search query: {query}")

        # Extract intent from query
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]

        logger.info(f"📊 Intent: {intent}, Data: {data}")

        # Extract identifiers
        fahrzeug_id = ""
        owner_name = query  # Default to full query
        license_plate = ""

        # Try to find fahrzeug ID
        fahrzeug_id_match = re.search(r'F\d{3}', query.upper())
        if fahrzeug_id_match:
            fahrzeug_id = fahrzeug_id_match.group()
            context.userdata.vehicle_context.fahrzeug_id = fahrzeug_id
            logger.info(f"✅ Found Fahrzeug ID: {fahrzeug_id}")
        
        # Store identification in context based on intent
        if intent == "fahrzeug_id":
            context.userdata.vehicle_context.fahrzeug_id = data
            fahrzeug_id = data
        elif intent == "owner_name":
            context.userdata.vehicle_context.owner_name = data
            owner_name = data
        elif intent == "license_plate":
            context.userdata.vehicle_context.license_plate = data
            license_plate = data

        # Store search query
        context.userdata.last_search_query = query

        try:
            # IMMER direkt von Qdrant holen
            logger.info(f"🔄 Using direct Qdrant fetch for: {fahrzeug_id or owner_name or license_plate}")
            
            vehicle_data = await self.get_all_vehicle_chunks_from_qdrant(
                fahrzeug_id=fahrzeug_id,
                owner_name=owner_name,
                license_plate=license_plate,
                qdrant_url=context.userdata.qdrant_url
            )
            
            if vehicle_data and "fahrzeug_id" in vehicle_data:
                # Store active vehicle
                if vehicle_data.get("fahrzeug_id"):
                    context.userdata.active_vehicle = vehicle_data["fahrzeug_id"]
                
                # Build comprehensive response (NUR DIE DATEN, KEIN "Ich habe gefunden" PREFIX!)
                response_parts = []
                
                # 1. Fahrzeug Info
                response_parts.append("🚗 FAHRZEUGDATEN:")
                response_parts.append(f"- Besitzer: {vehicle_data.get('besitzer', 'N/A')}")
                response_parts.append(f"- Kennzeichen: {vehicle_data.get('kennzeichen', 'N/A')}")
                response_parts.append(f"- Fahrzeug: {vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')}")
                response_parts.append(f"- Baujahr: {vehicle_data.get('baujahr', 'N/A')}")
                
                # Kilometerstand mit Formatierung für bessere TTS-Aussprache
                km = vehicle_data.get('kilometerstand', 0)
                if isinstance(km, (int, float)):
                    km_formatted = f"{km:,}".replace(',', ' ')
                    response_parts.append(f"- Kilometerstand: {km_formatted} km")
                else:
                    response_parts.append(f"- Kilometerstand: {km} km")
                
                response_parts.append(f"- Fahrzeug-ID: {vehicle_data.get('fahrzeug_id', 'N/A')}")
                
                # 2. Letzte Services
                if vehicle_data.get("letzte_services"):
                    response_parts.append("\n🔧 LETZTE SERVICES:")
                    for service in vehicle_data["letzte_services"]:
                        # Datum formatieren für bessere TTS-Aussprache
                        datum = service.get('datum', 'N/A')
                        if '-' in str(datum):
                            parts = datum.split('-')
                            datum_formatted = f"{parts[2]}.{parts[1]}.{parts[0]}"
                        else:
                            datum_formatted = datum
                        
                        # Kilometerstand formatieren für bessere TTS-Aussprache
                        km_stand = service.get('km_stand', 0)
                        if isinstance(km_stand, (int, float)):
                            km_formatted = f"{km_stand:,}".replace(',', ' ')
                        else:
                            km_formatted = str(km_stand)
                        
                        response_parts.append(f"- {datum_formatted} ({km_formatted} km):")
                        response_parts.append(f"  Typ: {service.get('service_typ', 'N/A')}")
                        response_parts.append(f"  Arbeiten: {', '.join(service.get('arbeiten', []))}")
                        kosten = service.get('kosten', 0)
                        response_parts.append(f"  Kosten: {kosten:.2f} Schweizer Franken")
                
                # 3. Aktuelle Probleme - WICHTIG!
                if vehicle_data.get("aktuelle_probleme"):
                    response_parts.append("\n⚠️ AKTUELLE PROBLEME:")
                    for problem in vehicle_data["aktuelle_probleme"]:
                        response_parts.append(f"- {problem}")
                
                # 4. Anstehende Arbeiten
                if vehicle_data.get("anstehende_arbeiten"):
                    response_parts.append("\n💰 ANSTEHENDE ARBEITEN:")
                    for arbeit in vehicle_data["anstehende_arbeiten"]:
                        response_parts.append(f"- {arbeit.get('arbeit', 'N/A')} (Priorität: {arbeit.get('priorität', 'N/A')})")
                        kosten = arbeit.get('geschätzte_kosten', 0)
                        response_parts.append(f"  Geschätzte Kosten: {kosten:.2f} Schweizer Franken")
                
                # 5. Nächster Service - mit Datum-Formatierung
                if vehicle_data.get("nächster_service_fällig"):
                    service_text = vehicle_data['nächster_service_fällig']
                    # Versuche Datum im Text zu finden und zu formatieren
                    if 'oder' in service_text and '-' in service_text:
                        parts = service_text.split('oder')
                        if len(parts) > 1:
                            datum_teil = parts[1].strip()
                            if '-' in datum_teil:
                                datum_parts = datum_teil.split('-')
                                if len(datum_parts) == 3:
                                    datum_formatted = f"{datum_parts[2]}.{datum_parts[1]}.{datum_parts[0]}"
                                    service_text = f"{parts[0].strip()} oder {datum_formatted}"
                    
                    response_parts.append(f"\n📅 Nächster Service fällig: {service_text}")
                
                final_response = "\n".join(response_parts)
                
                # Update conversation state
                context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                
                logger.info(f"✅ Returning complete vehicle data")
                return final_response  # NUR DIE DATEN ZURÜCKGEBEN!

            else:
                logger.info("❌ No vehicle data found in Qdrant")
                context.userdata.vehicle_context.reset()
                context.userdata.conversation_state = ConversationState.COLLECTING_IDENTIFIER
                
                # Klarere Nachricht ohne Raum für Halluzination
                return f"Ich konnte keine Fahrzeugdaten für '{owner_name or query}' finden. Bitte überprüfen Sie die Schreibweise oder nutzen Sie die Fahrzeug-ID (z.B. F001)."

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return "Die Fahrzeugdatenbank ist momentan nicht erreichbar. Bitte versuchen Sie es später noch einmal."

    @function_tool
    async def search_invoice_data(self,
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Rechnungsinformationen und Kosten für Services und Reparaturen.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation

        Returns:
            Rechnungsinformationen oder Fehlermeldung
        """
        # Nutze gespeicherte Identifikation wenn vorhanden
        if context.userdata.active_vehicle:
            query = context.userdata.active_vehicle
        
        # Verwende die gleiche Suchlogik
        result = await self.search_customer_data(context, query)
        
        # Extrahiere nur Kosten-relevante Informationen
        if "LETZTE SERVICES" in result or "ANSTEHENDE ARBEITEN" in result:
            return result
        else:
            return "Keine Rechnungsdaten gefunden."

    @function_tool
    async def search_repair_status(self,
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Reparaturstatus und anstehenden Arbeiten.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation

        Returns:
            Reparaturstatus oder Fehlermeldung
        """
        # Nutze gespeicherte Identifikation wenn vorhanden
        if context.userdata.active_vehicle:
            query = context.userdata.active_vehicle
        
        # Verwende die gleiche Suchlogik
        result = await self.search_customer_data(context, query)
        
        # Fokus auf Probleme und anstehende Arbeiten
        if "AKTUELLE PROBLEME" in result or "ANSTEHENDE ARBEITEN" in result:
            return result
        else:
            return "Keine aktuellen Reparaturen oder Probleme dokumentiert."


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"

    logger.info("="*50)
    logger.info(f"🏁 Starting Garage Agent Session: {session_id}")
    logger.info("="*50)

    session = None
    session_closed = False

    # Register disconnect handler FIRST
    def on_disconnect():
        nonlocal session_closed
        logger.info(f"[{session_id}] Room disconnected event received")
        session_closed = True

    if ctx.room:
        ctx.room.on("disconnected", on_disconnect)

    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")

        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")

        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10

        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                    logger.info(f"✅ [{session_id}] Audio track found and subscribed")
                    audio_track_received = True
                    break

            if audio_track_received:
                break

            await asyncio.sleep(1)
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")

        if not audio_track_received:
            logger.warning(f"⚠️ [{session_id}] No audio track found after {max_wait_time}s, continuing anyway...")

        # 4. Configure LLM with Ollama
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        qdrant_url = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")

        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with direct Qdrant access")

        # 5. Create session
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                qdrant_url=qdrant_url,
                current_vehicle_data=None,
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                vehicle_context=VehicleContext(),
                last_search_query=None,
                active_vehicle=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,
                min_speech_duration=0.2
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )

        # 6. Create agent
        agent = GarageAssistant()

        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )

        # Warte auf Audio-Stabilisierung
        await asyncio.sleep(2.0)

        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            intent_result = IdentifierExtractor.extract_intent_from_input(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")

        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")

        @session.on("function_call")
        def on_function_call(event):
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
            
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Generated response preview: {response_preview}...")
            
            # Hallucination Check
            if session.userdata.last_search_query:
                response_str = str(event) if hasattr(event, '__str__') else ""
                
                if "keine passenden Daten" in response_str and session.userdata.active_vehicle:
                    logger.error("⚠️ HALLUCINATION DETECTED! Agent ignoring found data!")

        # 8. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")

        try:
            greeting_text = """Guten Tag und herzlich willkommen bei der Garage Müller!
Ich bin Pia, Ihre digitale Assistentin.

Für eine schnelle Bearbeitung benötige ich eine der folgenden Informationen:
- Ihre Fahrzeug-ID (z.B. F001)
- Ihren vollständigen Namen
- Ihr Autokennzeichen

Wie kann ich Ihnen heute helfen?"""

            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST

            # Retry-Mechanismus für Greeting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await session.say(
                        greeting_text,
                        allow_interruptions=True,
                        add_to_chat_ctx=True
                    )
                    logger.info(f"✅ [{session_id}] Initial greeting sent successfully")
                    break
                except Exception as e:
                    logger.warning(f"[{session_id}] Greeting attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                    else:
                        raise

        except Exception as e:
            logger.error(f"[{session_id}] Greeting error after all retries: {e}", exc_info=True)

        logger.info(f"✅ [{session_id}] Garage Agent ready with direct Qdrant access!")

        # Wait for disconnect
        disconnect_event = asyncio.Event()

        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()

        ctx.room.on("disconnected", handle_disconnect)

        await disconnect_event.wait()
        logger.info(f"[{session_id}] Session ending...")

    except Exception as e:
        logger.error(f"❌ [{session_id}] Error: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass

        logger.info(f"✅ [{session_id}] Cleanup complete")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
