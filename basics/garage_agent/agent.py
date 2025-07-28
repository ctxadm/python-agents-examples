# LiveKit Agents - Garage Management Agent
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
class CustomerContext:
    """Kontext für Kunden-Identifikation"""
    fahrzeug_id: Optional[str] = None  # z.B. F001, F002
    canton_letters: Optional[str] = None
    numbers: Optional[str] = None
    complete_plate: Optional[str] = None
    customer_name: Optional[str] = None
    attempts: int = 0
    
    def has_identifier(self) -> bool:
        """Prüft ob eine Identifikation vorhanden ist"""
        return bool(self.fahrzeug_id or self.complete_plate or self.customer_name)
    
    def get_search_query(self) -> Optional[str]:
        """Gibt die beste Suchanfrage zurück"""
        if self.fahrzeug_id:
            return self.fahrzeug_id
        elif self.complete_plate:
            return self.complete_plate
        elif self.customer_name:
            return self.customer_name
        return None
    
    def combine_plate(self) -> Optional[str]:
        """Kombiniert Canton und Zahlen zu vollständigem Kennzeichen"""
        if self.canton_letters and self.numbers:
            canton = self.canton_letters.replace(".", "").replace(" ", "").upper()
            numbers = self.numbers.replace(" ", "")
            self.complete_plate = f"{canton} {numbers}"
            return self.complete_plate
        return None
    
    def reset(self):
        """Reset des Kontexts"""
        self.fahrzeug_id = None
        self.canton_letters = None
        self.numbers = None
        self.complete_plate = None
        self.customer_name = None
        self.attempts = 0

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    current_vehicle_data: Optional[Dict[str, Any]] = None
    user_language: str = "de"
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    customer_context: CustomerContext = field(default_factory=CustomerContext)
    last_search_query: Optional[str] = None
    hallucination_count: int = 0


class IdentifierExtractor:
    """Extrahiert und validiert Kunden-Identifikatoren"""
    
    SWISS_CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", 
                     "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", 
                     "TI", "UR", "VD", "VS", "ZG", "ZH"]
    
    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()
        
        # Check for specific service queries FIRST
        if any(word in input_lower for word in ["letzte service", "letzten service", "service historie", "servicehistorie"]):
            return {"intent": "service_history", "data": user_input}
        
        if any(word in input_lower for word in ["anstehende arbeiten", "anstehende reparaturen", "was muss gemacht werden"]):
            return {"intent": "pending_work", "data": user_input}
        
        if any(word in input_lower for word in ["kosten", "preis", "wie viel", "was kostet"]):
            if "anstehend" in input_lower or "gesamt" in input_lower:
                return {"intent": "cost_estimate", "data": user_input}
            else:
                return {"intent": "cost_query", "data": user_input}
        
        # Fahrzeug-ID (F001, F002, etc.) - Check even in longer sentences
        fahrzeug_id_match = re.search(r'\b(F\d{3})\b', user_input.upper())
        if fahrzeug_id_match:
            return {"intent": "fahrzeug_id", "data": fahrzeug_id_match.group(1)}
        
        # Greetings - but only if no other important info is present
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        has_greeting = any(g in input_lower for g in greetings)
        
        # Name detection
        name_patterns = [
            r"(?:mein name ist|ich heiße|ich bin)\s+([A-Za-zäöüÄÖÜß\s]+)",
            r"^([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)+)$"  # Vorname Nachname
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name.lower() not in ["die", "der", "das", "mein", "dein", "ihr"]:
                    return {"intent": "customer_name", "data": name}
        
        # If we found a greeting but no other data, return greeting
        if has_greeting and not fahrzeug_id_match:
            return {"intent": "greeting", "data": None}
        
        # Complete license plate
        plate_match = re.search(r'([A-Z]{2})\s*(\d{3,6})', user_input.upper())
        if plate_match:
            canton = plate_match.group(1)
            numbers = plate_match.group(2)
            if cls.validate_canton(canton):
                return {"intent": "license_plate_complete", "data": f"{canton} {numbers}"}
        
        # Service/repair/cost keywords
        if any(word in input_lower for word in ["reparatur", "service", "status", "kosten", "rechnung", "inspektion"]):
            return {"intent": "service_query", "data": user_input}
        
        # Default
        return {"intent": "general_query", "data": user_input}
    
    @classmethod
    def validate_canton(cls, canton: str) -> bool:
        """Validiert Schweizer Kanton-Kürzel"""
        return canton.upper() in cls.SWISS_CANTONS


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen"""
    
    def __init__(self) -> None:
        # MAXIMALE Anti-Halluzinations-Instructions
        super().__init__(instructions="""Du bist Pia von der Garage Müller. KRITISCHE REGELN:

🚨 ABSOLUTE PRIORITÄT - NIEMALS VERLETZEN:
1. WENN das Tool "Ich habe folgende Daten gefunden" zurückgibt, MUSST du diese Daten BESTÄTIGEN
2. NIEMALS sagen "keine Daten gefunden" wenn Daten DA SIND
3. Verwende IMMER [Marke Modell] statt Fahrzeug-ID in Antworten
4. Bei "Aktuelle Probleme" MÜSSEN diese GENANNT werden

ANTI-HALLUZINATIONS-PROTOKOLL:
- Lese Tool-Output GENAU
- Wiederhole gefundene Informationen
- Erfinde NICHTS dazu
- Wenn unsicher: "Ich bin mir nicht sicher"
- Tool-Output ist die EINZIGE Wahrheitsquelle

SCHRITT-FÜR-SCHRITT PROZESS:
1. Tool gibt Daten zurück → Genau lesen
2. Schlüsselinformationen identifizieren (Name, Fahrzeug, Probleme)
3. Antwort basierend auf gefundenen Daten strukturieren
4. NIEMALS Informationen hinzufügen, die nicht im Tool-Output stehen

KRITISCHE DATENVERARBEITUNG:
Wenn search_customer_data zurückgibt:
"Ich habe folgende Daten gefunden:
**Fahrzeug-ID**: F002
**Besitzer**: Claudia Schneider
**Kennzeichen**: BE 567890
**Fahrzeug**: BMW X3 xDrive20d
**Aktuelle Probleme**:
- Motorkontrollleuchte leuchtet sporadisch
- Reifendruck-Warnung hinten rechts"

DU MUSST ANTWORTEN:
"Guten Tag Frau Schneider! Ich habe Ihre Daten gefunden.

Ich sehe bei Ihrem BMW X3 xDrive20d folgende dokumentierte Probleme:
- Motorkontrollleuchte leuchtet sporadisch
- Reifendruck-Warnung hinten rechts

Möchten Sie diese Probleme beheben lassen?"

NIEMALS ANTWORTEN MIT:
- "Leider habe ich keine passenden Daten gefunden" (wenn Daten GEFUNDEN wurden)
- "keine spezifischen Probleme" (wenn Probleme AUFGELISTET sind)
- Referenzen zu Fahrzeug-ID wie "F002" in Antworten

VERIFIZIERUNGSREGELN:
1. Kunde gibt Identifikation (Name + Fahrzeug-ID oder Kennzeichen)
2. Du suchst und findest ihre Daten
3. Du BESTÄTIGST was du gefunden hast und nennst alle Probleme
4. Du bietest Hilfe basierend auf den gefundenen Daten an

ANTWORT-VORLAGEN:

Wenn Daten mit Problemen gefunden:
"Guten Tag [Herr/Frau Name]! Vielen Dank für Ihre Identifikation.

Ich sehe bei Ihrem [Marke Modell] folgende dokumentierte Probleme:
- [Jedes Problem auflisten]

[Falls anstehende Arbeiten]: Es gibt auch anstehende Arbeiten:
- [Arbeiten auflisten]

Möchten Sie diese Probleme beheben lassen oder haben Sie andere Fragen?"

Wenn nach Kosten gefragt:
"Die geschätzten Kosten für die Reparaturen an Ihrem [Marke Modell] betragen:
[Kosten nach Priorität auflisten]
Gesamtkosten: CHF [Summe]"

Wenn keine Daten gefunden:
"Ich habe keine passenden Daten gefunden. Können Sie mir bitte Ihre Fahrzeug-ID (z.B. F001), Ihren vollständigen Namen oder Ihr Autokennzeichen nennen?"

VERBOTEN:
- "Entschuldigung" → Verwende "Leider"
- "Es tut mir leid" → Verwende "Bedauerlicherweise"
- Niemals Fahrzeug-ID in Antworten referenzieren, immer Marke + Modell verwenden

Denke daran: DER TOOL-OUTPUT IST DIE WAHRHEIT. Widerspreche niemals dem, was das Tool zurückgibt!

ANTWORTE NUR AUF DEUTSCH.""")
        
        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ GarageAssistant initialized with MAXIMUM anti-hallucination rules")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    @function_tool
    async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank basierend auf Fahrzeug-ID, Namen oder Kennzeichen.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Kundendaten gesucht werden soll.
        
        Args:
            query: Suchbegriff (Fahrzeug-ID, Name oder Autokennzeichen)
            
        Returns:
            Gefundene Kundendaten oder Fehlermeldung
        """
        logger.info(f"🔍 Original search query: {query}")
        
        # Extract intent from query
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]
        
        logger.info(f"📊 Intent: {intent}, Data: {data}")
        
        # Store identification in context
        if intent == "fahrzeug_id":
            context.userdata.customer_context.fahrzeug_id = data
            query = data
        elif intent == "customer_name":
            context.userdata.customer_context.customer_name = data
            query = data
        elif intent == "license_plate_complete":
            context.userdata.customer_context.complete_plate = data
            query = data
        
        # Store search query
        context.userdata.last_search_query = query
        
        try:
            # DIREKTE QDRANT ABFRAGE
            async with httpx.AsyncClient() as client:
                qdrant_response = await client.post(
                    "http://172.16.0.108:6333/collections/garage_management/points/scroll",
                    json={
                        "limit": 100,
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=5.0
                )
                
                if qdrant_response.status_code == 200:
                    qdrant_data = qdrant_response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    # Suche nach dem Query
                    search_lower = query.lower().strip()
                    vehicle_data = None
                    
                    for point in points:
                        payload = point.get("payload", {})
                        # Check verschiedene Felder
                        if (search_lower in str(payload.get("fahrzeug_id", "")).lower() or
                            search_lower in str(payload.get("besitzer", "")).lower() or
                            search_lower in str(payload.get("kennzeichen", "")).lower().replace(" ", "")):
                            vehicle_data = payload
                            break
                    
                    if vehicle_data:
                        # Store vehicle data in context
                        context.userdata.current_vehicle_data = vehicle_data
                        
                        # Build response with clear structure
                        response_text = "Ich habe folgende Daten gefunden:\n"
                        
                        if 'fahrzeug_id' in vehicle_data:
                            response_text += f"\n**Fahrzeug-ID**: {vehicle_data['fahrzeug_id']}\n"
                        if 'besitzer' in vehicle_data:
                            response_text += f"**Besitzer**: {vehicle_data['besitzer']}\n"
                        if 'kennzeichen' in vehicle_data:
                            response_text += f"**Kennzeichen**: {vehicle_data['kennzeichen']}\n"
                        if 'marke' in vehicle_data and 'modell' in vehicle_data:
                            response_text += f"**Fahrzeug**: {vehicle_data['marke']} {vehicle_data['modell']}\n"
                        if 'baujahr' in vehicle_data:
                            response_text += f"**Baujahr**: {vehicle_data['baujahr']}\n"
                        if 'kilometerstand' in vehicle_data:
                            response_text += f"**Kilometerstand**: {vehicle_data['kilometerstand']} km\n"
                        
                        # Show latest service if available
                        if 'letzte_services' in vehicle_data and vehicle_data['letzte_services']:
                            latest_service = vehicle_data['letzte_services'][0]
                            response_text += f"\n**Letzter Service** ({latest_service.get('datum', 'unbekannt')}):\n"
                            response_text += f"- Typ: {latest_service.get('service_typ', 'unbekannt')}\n"
                            response_text += f"- Kosten: CHF {latest_service.get('kosten', 0):.2f}\n"
                        
                        # ALWAYS show current problems if any
                        if 'aktuelle_probleme' in vehicle_data and vehicle_data['aktuelle_probleme']:
                            response_text += f"\n**Aktuelle Probleme**:\n"
                            for problem in vehicle_data['aktuelle_probleme']:
                                response_text += f"- {problem}\n"
                        
                        # CRITICAL: Log the exact response for debugging
                        logger.info(f"✅ Tool response length: {len(response_text)} chars")
                        logger.info(f"📄 Tool response preview: {response_text[:200]}...")
                        logger.info(f"🔥 FOUND: {vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')} for {vehicle_data.get('besitzer', '')}")
                        
                        # Set a flag to track that data was found
                        context.userdata.hallucination_count = 0  # Reset counter on successful find
                        
                        return response_text
                    else:
                        logger.info("❌ No results found in Qdrant")
                        context.userdata.customer_context.reset()
                        return "Ich habe keine passenden Daten gefunden. Können Sie mir bitte Ihre Fahrzeug-ID (z.B. F001), Ihren vollständigen Namen oder Ihr Autokennzeichen nennen?"
                else:
                    logger.error(f"Qdrant error: {qdrant_response.status_code}")
                    return "Es gab ein Problem mit der Datenbank-Verbindung. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Es gab einen Fehler bei der Suche. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Rechnungsinformationen und Kosten für Services und Reparaturen.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Kosten oder Rechnungen gefragt wird.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation
            
        Returns:
            Gefundene Rechnungsdaten oder Fehlermeldung
        """
        logger.info(f"💰 Searching invoice data for: {query}")
        
        # Extract intent to understand what user is asking for
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        
        # Use stored identification if available
        if context.userdata.customer_context.has_identifier():
            search_query = context.userdata.customer_context.get_search_query()
            logger.info(f"📋 Using stored identifier: {search_query}")
        else:
            search_query = query
            
        try:
            # Use stored vehicle data if available
            if context.userdata.current_vehicle_data:
                vehicle_data = context.userdata.current_vehicle_data
                
                # Get vehicle name for responses
                vehicle_name = f"{vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')}"
                
                if intent == "cost_estimate" or "anstehend" in query.lower():
                    # User wants cost estimate for pending work
                    if 'anstehende_arbeiten' in vehicle_data and vehicle_data['anstehende_arbeiten']:
                        response_text = f"Die geschätzten Kosten für die Reparaturen an Ihrem {vehicle_name} betragen:\n"
                        total_cost = 0
                        
                        # Group by priority
                        high_priority = []
                        medium_priority = []
                        low_priority = []
                        
                        for arbeit in vehicle_data['anstehende_arbeiten']:
                            priority = arbeit.get('priorität', 'normal')
                            if priority == 'hoch':
                                high_priority.append(arbeit)
                            elif priority == 'mittel':
                                medium_priority.append(arbeit)
                            else:
                                low_priority.append(arbeit)
                        
                        if high_priority:
                            response_text += "\n**Hohe Priorität:**\n"
                            for arbeit in high_priority:
                                cost = arbeit.get('geschätzte_kosten', 0)
                                response_text += f"- {arbeit.get('arbeit', 'Arbeit')}: CHF {cost:.2f}\n"
                                total_cost += cost
                        
                        if medium_priority:
                            response_text += "\n**Mittlere Priorität:**\n"
                            for arbeit in medium_priority:
                                cost = arbeit.get('geschätzte_kosten', 0)
                                response_text += f"- {arbeit.get('arbeit', 'Arbeit')}: CHF {cost:.2f}\n"
                                total_cost += cost
                        
                        if low_priority:
                            response_text += "\n**Niedrige Priorität:**\n"
                            for arbeit in low_priority:
                                cost = arbeit.get('geschätzte_kosten', 0)
                                response_text += f"- {arbeit.get('arbeit', 'Arbeit')}: CHF {cost:.2f}\n"
                                total_cost += cost
                        
                        response_text += f"\n**Gesamtkosten**: CHF {total_cost:.2f}"
                        return response_text
                    else:
                        return "Es sind keine anstehenden Arbeiten für Ihr Fahrzeug dokumentiert."
                else:
                    return "Ich kann Ihnen gerne die Kosteninformationen zeigen. Was möchten Sie genau wissen?"
            else:
                return "Bitte identifizieren Sie sich zuerst mit Ihrer Fahrzeug-ID, Namen oder Kennzeichen."
                    
        except Exception as e:
            logger.error(f"Invoice search error: {e}")
            return "Es gab einen Fehler bei der Kostenabfrage. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Reparaturstatus und anstehenden Arbeiten.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Reparaturen gefragt wird.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation
            
        Returns:
            Gefundene Reparaturdaten oder Fehlermeldung
        """
        logger.info(f"🔧 Searching repair status for: {query}")
        
        # Use stored vehicle data if available
        if context.userdata.current_vehicle_data:
            vehicle_data = context.userdata.current_vehicle_data
            vehicle_name = f"{vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')}"
            
            response_text = f"Hier ist der Status für Ihr {vehicle_name}:\n"
            
            # Always show current problems first
            if 'aktuelle_probleme' in vehicle_data and vehicle_data['aktuelle_probleme']:
                response_text += "\n**Aktuelle Probleme:**\n"
                for problem in vehicle_data['aktuelle_probleme']:
                    response_text += f"- {problem}\n"
            else:
                response_text += "\n**Keine aktuellen Probleme dokumentiert.**\n"
            
            # Show pending work
            if 'anstehende_arbeiten' in vehicle_data and vehicle_data['anstehende_arbeiten']:
                response_text += "\n**Anstehende Arbeiten:**\n"
                
                # Group by priority
                high_priority = []
                medium_priority = []
                low_priority = []
                
                for arbeit in vehicle_data['anstehende_arbeiten']:
                    priority = arbeit.get('priorität', 'normal')
                    if priority == 'hoch':
                        high_priority.append(arbeit)
                    elif priority == 'mittel':
                        medium_priority.append(arbeit)
                    else:
                        low_priority.append(arbeit)
                
                if high_priority:
                    response_text += "\n*Hohe Priorität:*\n"
                    for arbeit in high_priority:
                        response_text += f"- {arbeit.get('arbeit', 'Arbeit')}\n"
                
                if medium_priority:
                    response_text += "\n*Mittlere Priorität:*\n"
                    for arbeit in medium_priority:
                        response_text += f"- {arbeit.get('arbeit', 'Arbeit')}\n"
                
                if low_priority:
                    response_text += "\n*Niedrige Priorität:*\n"
                    for arbeit in low_priority:
                        response_text += f"- {arbeit.get('arbeit', 'Arbeit')}\n"
            
            # Show next service
            if 'nächster_service_fällig' in vehicle_data:
                response_text += f"\n**Nächster Service fällig**: {vehicle_data['nächster_service_fällig']}"
            
            # Show warranty status
            if 'garantie_bis' in vehicle_data:
                response_text += f"\n**Garantie bis**: {vehicle_data['garantie_bis']}"
            
            return response_text
        else:
            return "Bitte identifizieren Sie sich zuerst mit Ihrer Fahrzeug-ID, Namen oder Kennzeichen."


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
    logger.info(f"🚗 Starting Garage Agent Session: {session_id}")
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
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
                
            await asyncio.sleep(1)
        
        # 4. Configure LLM with Ollama - VOLLSTÄNDIG OPTIMIERT
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,
            top_k=5,
            top_p=0.05,
            repeat_penalty=2.0,
            num_ctx=4096,
            num_predict=200,
            seed=42,
            # Mirostat direkt als Parameter, nicht in options
            mirostat=2,
            mirostat_tau=2.0,
            mirostat_eta=0.1,
            stop=["User:", "Human:", "###", "Kunde:"],
            }
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with MAXIMUM anti-hallucination settings")
        
        # 5. Create session with enhanced configuration
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                current_vehicle_data=None,
                user_language="de",
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                customer_context=CustomerContext(),
                last_search_query=None,
                hallucination_count=0
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.4,
                min_speech_duration=0.15,
                activation_threshold=0.5  # Standard threshold
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova",
                speed=1.0
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0,
            min_interruption_duration=0.5,  # Prevent accidental interruptions
            disable_early_inference=True,    # Wait for complete user input
            # Zusätzliche Session-Parameter für bessere Kontrolle
            conn_options={
                "max_retry": 3,
                "retry_interval": 2.0,
                "timeout": 10.0
            }
        )
        
        # 6. Create agent
        agent = GarageAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Enhanced event handlers with debugging
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            
            # Intent detection
            intent_result = IdentifierExtractor.extract_intent_from_input(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("function_call")
        def on_function_call(event):
            """Log function calls für Debugging"""
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
        
        @session.on("agent_response")
        def on_agent_response(event):
            """Monitor agent responses for hallucinations"""
            response = str(event)
            logger.info(f"[{session_id}] 💬 Agent response preview: {response[:100]}...")
            
            if session.userdata.current_vehicle_data:
                vehicle_info = f"{session.userdata.current_vehicle_data.get('marke', '')} {session.userdata.current_vehicle_data.get('modell', '')}"
                
                # Check for hallucination patterns
                if "keine passenden Daten gefunden" in response or "keine Daten gefunden" in response:
                    logger.error(f"[{session_id}] 🚨 HALLUCINATION DETECTED! Agent claims no data when data exists for {vehicle_info}!")
                    session.userdata.hallucination_count += 1
                    logger.error(f"[{session_id}] 📊 Hallucination count: {session.userdata.hallucination_count}")
        
        @session.on("llm_response_received")
        def on_llm_response(event):
            """Log raw LLM responses for debugging"""
            logger.debug(f"[{session_id}] 🧠 LLM raw response: {event}")
            
            # Check for hallucination patterns
            if hasattr(event, 'content'):
                content = str(event.content)
                if session.userdata.current_vehicle_data:
                    vehicle_name = f"{session.userdata.current_vehicle_data.get('marke', '')} {session.userdata.current_vehicle_data.get('modell', '')}"
                    if "keine daten" in content.lower() or "nicht gefunden" in content.lower():
                        logger.error(f"[{session_id}] ⚠️ POTENTIAL HALLUCINATION: LLM ignoring data for {vehicle_name}")
                        # Force correction in next response
                        session.userdata.hallucination_count += 1
        
        # 8. Initial greeting
        await asyncio.sleep(1.5)
        
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
            
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )
            
            logger.info(f"✅ [{session_id}] Initial greeting sent")
            
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}")
        
        logger.info(f"✅ [{session_id}] Garage Agent ready with MAXIMUM anti-hallucination!")
        
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
