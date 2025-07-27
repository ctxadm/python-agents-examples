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
    COLLECTING_LICENSE_PLATE = "collecting_license_plate"
    SEARCHING = "searching"
    PROVIDING_INFO = "providing_info"

@dataclass
class LicensePlateContext:
    """Kontext für Kennzeichen-Erfassung"""
    canton_letters: Optional[str] = None
    numbers: Optional[str] = None
    complete_plate: Optional[str] = None
    attempts: int = 0
    
    def is_complete(self) -> bool:
        """Prüft ob Kennzeichen vollständig ist"""
        return self.complete_plate is not None
    
    def combine(self) -> Optional[str]:
        """Kombiniert Canton und Zahlen zu vollständigem Kennzeichen"""
        if self.canton_letters and self.numbers:
            canton = self.canton_letters.replace(".", "").replace(" ", "").upper()
            numbers = self.numbers.replace(" ", "")
            self.complete_plate = f"{canton} {numbers}"
            return self.complete_plate
        return None
    
    def reset(self):
        """Reset des Kontexts"""
        self.canton_letters = None
        self.numbers = None
        self.complete_plate = None
        self.attempts = 0

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    current_customer_id: Optional[str] = None
    active_repair_id: Optional[str] = None
    user_language: str = "de"
    greeting_sent: bool = False
    customer_name: Optional[str] = None
    conversation_state: ConversationState = ConversationState.GREETING
    license_plate_context: LicensePlateContext = field(default_factory=LicensePlateContext)
    last_search_query: Optional[str] = None
    hallucination_count: int = 0


class HalluccinationDetector:
    """Erkennt und verhindert Halluzinationen"""
    
    EXAMPLE_PLATES = ["LU 234567", "ZH 123456", "BE 987654"]
    SWISS_CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", 
                     "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", 
                     "TI", "UR", "VD", "VS", "ZG", "ZH"]
    
    @classmethod
    def is_hallucinated_plate(cls, plate: str) -> bool:
        """Prüft ob ein Kennzeichen halluziniert ist"""
        if not plate:
            return False
        
        # Check gegen bekannte Beispiele
        normalized_plate = plate.replace(" ", "").upper()
        for example in cls.EXAMPLE_PLATES:
            if normalized_plate == example.replace(" ", ""):
                logger.warning(f"⚠️ Halluziniertes Beispiel-Kennzeichen erkannt: {plate}")
                return True
        
        # Prüfe auf unrealistische Muster
        if re.match(r'^[A-Z]{2}\s*123456$', plate):
            logger.warning(f"⚠️ Verdächtiges Kennzeichen-Muster: {plate}")
            return True
            
        return False
    
    @classmethod
    def validate_canton(cls, canton: str) -> bool:
        """Validiert Schweizer Kanton-Kürzel"""
        return canton.upper() in cls.SWISS_CANTONS
    
    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()
        
        # Greetings
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        if any(g in input_lower for g in greetings):
            return {"intent": "greeting", "data": None}
        
        # Buchstabierte Buchstaben (z.B. "Z wie Zeppelin, H wie Heinrich")
        buchstabiert_pattern = r'([A-Z])\s*wie\s*\w+(?:\s*[,.]?\s*([A-Z])\s*wie\s*\w+)?'
        buchstabiert_match = re.findall(buchstabiert_pattern, user_input, re.IGNORECASE)
        if buchstabiert_match:
            # Extrahiere alle Buchstaben
            letters = []
            for match in buchstabiert_match:
                for letter in match:
                    if letter:
                        letters.append(letter.upper())
            
            if len(letters) == 2:
                canton = ''.join(letters)
                if cls.validate_canton(canton):
                    return {"intent": "canton_provided", "data": canton}
            elif len(letters) == 1:
                return {"intent": "partial_canton", "data": letters[0]}
        
        # Canton letters (z.B. "A.G.", "AG", "A G")
        canton_patterns = [
            r'^([A-Z])\s*\.?\s*([A-Z])\.?$',  # A.G. oder A G
            r'^([A-Z]{2})$',  # AG
        ]
        
        for pattern in canton_patterns:
            match = re.match(pattern, user_input.upper().strip())
            if match:
                if len(match.groups()) == 2:
                    canton = match.group(1) + match.group(2)
                else:
                    canton = match.group(1)
                
                if cls.validate_canton(canton):
                    return {"intent": "canton_provided", "data": canton}
        
        # Numbers only (z.B. "5 6 7 8 9" oder "56789")
        # Auch mit Kommas (z.B. "1, 2, 3...")
        numbers_pattern = r'^[\d\s,\.]+$'
        if re.match(numbers_pattern, user_input.strip()):
            # Entferne alles außer Zahlen
            numbers = re.sub(r'[^\d]', '', user_input)
            if 3 <= len(numbers) <= 6:
                return {"intent": "numbers_provided", "data": numbers}
        
        # Complete license plate
        plate_match = re.search(r'([A-Z]{2})\s*(\d{3,6})', user_input.upper())
        if plate_match:
            canton = plate_match.group(1)
            numbers = plate_match.group(2)
            if cls.validate_canton(canton):
                return {"intent": "complete_plate", "data": f"{canton} {numbers}"}
        
        # Check for repair/service/cost keywords
        if any(word in input_lower for word in ["reparatur", "service", "status", "kosten", "rechnung"]):
            return {"intent": "service_query", "data": user_input}
        
        # Default: general query
        return {"intent": "general_query", "data": user_input}


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen"""
    
    def __init__(self) -> None:
        # VERBESSERTE Instructions mit Few-Shot Examples und Anti-Halluzination
        super().__init__(instructions="""You are Pia, the digital assistant of Garage Müller. RESPOND ONLY IN GERMAN.

CRITICAL ANTI-HALLUCINATION RULES:
1. NEVER use example license plates like "LU 234567" or "ZH 123456"
2. ONLY use the ACTUAL input provided by the user
3. When combining canton and numbers, use EXACTLY what the user said
4. If unsure, ASK for clarification instead of guessing

CONVERSATION EXAMPLES (Learn from these):

Example 1 - Correct license plate handling:
User: "A.G."
You: "Ich habe AG notiert. Können Sie mir bitte die Zahlen Ihres Kennzeichens nennen?"
User: "5 6 7 8 9"
You: [SEARCH with "AG 56789"] (NOT "LU 234567"!)

Example 2 - Name search:
User: "Mein Name ist Thomas Meier"
You: [SEARCH with "Thomas Meier"]

Example 3 - Direct plate:
User: "Mein Kennzeichen ist ZH 789123"
You: [SEARCH with "ZH 789123"]

FORBIDDEN WORDS (use alternatives):
- "Entschuldigung" → "Leider"
- "Es tut mir leid" → "Bedauerlicherweise"
- "Sorry" → "Leider"

STATE AWARENESS:
- Track conversation state
- Remember partial inputs (canton, then numbers)
- Combine inputs correctly

RESPONSE RULES:
1. Be friendly and professional
2. For greetings: Welcome and ask how to help
3. For data requests: Search immediately
4. If no data found: Ask for license plate
5. NEVER invent data or dates

SEARCH TRIGGERS:
- Complete license plates
- Customer names
- Service/repair queries
- Cost/invoice queries

Remember: ALWAYS use actual user input, NEVER use examples!""")
        
        self.hallucination_detector = HalluccinationDetector()
        logger.info("✅ GarageAssistant initialized with anti-hallucination measures")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    def _validate_and_clean_query(self, query: str, context: RunContext[GarageUserData]) -> Optional[str]:
        """Validiert und bereinigt die Suchanfrage"""
        # Prüfe auf halluzinierte Kennzeichen
        if self.hallucination_detector.is_hallucinated_plate(query):
            logger.error(f"❌ Halluziniertes Kennzeichen erkannt: {query}")
            
            # Versuche aus Kontext zu rekonstruieren
            if context.userdata.license_plate_context.is_complete():
                correct_plate = context.userdata.license_plate_context.complete_plate
                logger.info(f"✅ Verwende korrektes Kennzeichen aus Kontext: {correct_plate}")
                return correct_plate
            
            return None
        
        return query

    @function_tool
    async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank.
        
        Args:
            query: Suchbegriff (Name, Telefonnummer oder Autokennzeichen)
        """
        logger.info(f"🔍 Original search query: {query}")
        
        # WICHTIG: Prüfe ob Query buchstabierte Buchstaben enthält
        # z.B. "B Bertha E Emil" sollte zu "BE" werden
        buchstabiert_match = re.search(r'([A-Z])\s*\w+\s+([A-Z])\s*\w+', query)
        if buchstabiert_match:
            canton = buchstabiert_match.group(1) + buchstabiert_match.group(2)
            # Extrahiere Zahlen falls vorhanden
            numbers_match = re.search(r'\d+', query)
            if numbers_match and self.hallucination_detector.validate_canton(canton):
                query = f"{canton} {numbers_match.group()}"
                logger.info(f"✅ Korrigierte buchstabierte Eingabe zu: {query}")
        
        # Validiere Query gegen Halluzinationen
        validated_query = self._validate_and_clean_query(query, context)
        if validated_query is None:
            return "Ich konnte das Kennzeichen nicht korrekt erfassen. Bitte nennen Sie es mir noch einmal."
        
        query = validated_query
        context.userdata.last_search_query = query
        
        # Intent-Erkennung
        intent_result = self.hallucination_detector.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]
        
        logger.info(f"📊 Intent: {intent}, Data: {data}")
        
        # Handle verschiedene Intents
        if intent == "greeting":
            return "Für die Suche benötige ich Ihren Namen oder Ihr Autokennzeichen."
        
        elif intent == "canton_provided":
            # Speichere Canton im Kontext
            context.userdata.license_plate_context.canton_letters = data
            context.userdata.conversation_state = ConversationState.COLLECTING_LICENSE_PLATE
            return f"Danke, ich habe {data} notiert. Wie lauten die Zahlen Ihres Kennzeichens?"
        
        elif intent == "partial_canton":
            # Nur ein Buchstabe wurde gegeben
            if context.userdata.license_plate_context.canton_letters:
                # Wir haben bereits einen Buchstaben, kombiniere sie
                full_canton = context.userdata.license_plate_context.canton_letters + data
                if self.hallucination_detector.validate_canton(full_canton):
                    context.userdata.license_plate_context.canton_letters = full_canton
                    return f"Danke, ich habe {full_canton} notiert. Wie lauten die Zahlen Ihres Kennzeichens?"
            else:
                # Erster Buchstabe
                context.userdata.license_plate_context.canton_letters = data
                return "Ich habe den ersten Buchstaben notiert. Bitte nennen Sie mir noch den zweiten Buchstaben des Kantons."
        
        elif intent == "numbers_provided":
            # Prüfe ob wir bereits einen Canton haben
            if context.userdata.license_plate_context.canton_letters:
                context.userdata.license_plate_context.numbers = data
                complete_plate = context.userdata.license_plate_context.combine()
                
                if complete_plate:
                    logger.info(f"✅ Kennzeichen komplett: {complete_plate}")
                    # Rekursiver Aufruf mit vollständigem Kennzeichen
                    return await self.search_customer_data(context, complete_plate)
            else:
                return "Mir fehlt noch der Kanton (die zwei Buchstaben). Bitte nennen Sie mir das vollständige Kennzeichen."
        
        elif intent == "complete_plate":
            # Speichere vollständiges Kennzeichen
            context.userdata.license_plate_context.complete_plate = data
            query = data
        
        # Führe die eigentliche Suche durch
        try:
            async with httpx.AsyncClient() as client:
                # Erstelle Filter für präzisere Suche
                filter_conditions = None
                
                # Bei Kennzeichen-Suche
                if re.match(r'^[A-Z]{2}\s*\d{3,6}$', query.upper()):
                    kennzeichen_normalized = query.replace(" ", "").upper()
                    filter_conditions = {
                        "should": [
                            {
                                "key": "license_plate",
                                "match": {"value": query.upper()}
                            },
                            {
                                "key": "kennzeichen",
                                "match": {"value": query.upper()}
                            }
                        ]
                    }
                    logger.info(f"🔍 Suche mit Kennzeichen-Filter: {query}")
                
                # Baue Request
                search_request = {
                    "query": query,
                    "agent_type": "garage",
                    "top_k": 5,
                    "collection": "garage_management"
                }
                
                if filter_conditions:
                    search_request["filter"] = filter_conditions
                
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json=search_request
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} results")
                        
                        # Update conversation state
                        context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                        
                        # Sammle relevante Ergebnisse
                        relevant_results = []
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            payload = result.get("payload", {})
                            
                            # Skip irrelevante Ergebnisse
                            if not content or any(word in content.lower() for word in ["mhz", "bakom", "funk", "frequenz"]):
                                continue
                            
                            # Bei vehicle_complete Daten
                            if payload.get("data_type") == "vehicle_complete":
                                # Verwende searchable_text wenn vorhanden
                                if payload.get("searchable_text"):
                                    content = payload["searchable_text"]
                                
                                content = self._format_garage_data(content)
                                relevant_results.append(content)
                        
                        if relevant_results:
                            # Reset license plate context nach erfolgreicher Suche
                            context.userdata.license_plate_context.reset()
                            return "\n\n".join(relevant_results[:2])
                        else:
                            return "Ich habe keine passenden Daten gefunden. Können Sie mir bitte Ihr Autokennzeichen nennen?"
                    
                    # Keine Ergebnisse
                    if re.match(r'^[A-Z]{2}\s*\d{3,6}$', query.upper()):
                        return f"Ich habe keine Daten zum Kennzeichen {query} gefunden. Bitte prüfen Sie, ob das Kennzeichen korrekt ist."
                    else:
                        return "Ich habe keine Kundendaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                    
                else:
                    logger.error(f"Search failed: {response.status_code}")
                    return "Die Datenbank ist momentan nicht erreichbar. Bitte versuchen Sie es in einem Moment erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Reparaturstatus und Aufträgen.
        
        Args:
            query: Kundenname, Autokennzeichen oder Auftragsnummer
        """
        logger.info(f"🔧 Searching repair status for: {query}")
        
        # Validierung
        validated_query = self._validate_and_clean_query(query, context)
        if validated_query is None:
            return "Ich konnte die Suchanfrage nicht verarbeiten. Bitte nennen Sie mir Ihr Kennzeichen."
        
        query = validated_query
        
        if len(query) < 3:
            return "Für die Reparatursuche benötige ich einen Namen oder ein Autokennzeichen."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": f"Reparatur Service Status {query}",
                        "agent_type": "garage",
                        "top_k": 5,
                        "collection": "garage_management"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        relevant_results = []
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            if content and "service" in content.lower():
                                content = self._format_garage_data(content)
                                relevant_results.append(content)
                        
                        if relevant_results:
                            context.userdata.license_plate_context.reset()
                            return "\n\n".join(relevant_results[:2])
                        else:
                            return "Ich habe keine Reparaturdaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                    
                    return "Keine Reparaturdaten vorhanden. Können Sie mir Ihr Autokennzeichen nennen?"
                    
                else:
                    return "Die Reparaturdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Repair search error: {e}")
            return "Technischer Fehler bei der Reparatursuche."

    @function_tool
    async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Rechnungsinformationen und Kosten.
        
        Args:
            query: Kundenname, Rechnungsnummer oder Kennzeichen
        """
        logger.info(f"💰 Searching invoice/cost data for: {query}")
        
        # Wenn wir ein vollständiges Kennzeichen im Kontext haben, verwende es
        if context.userdata.license_plate_context.is_complete():
            stored_plate = context.userdata.license_plate_context.complete_plate
            logger.info(f"✅ Verwende gespeichertes Kennzeichen: {stored_plate}")
            query = f"{stored_plate} {query}"
        
        # Validierung
        validated_query = self._validate_and_clean_query(query, context)
        if validated_query is None:
            return "Ich konnte die Suchanfrage nicht verarbeiten. Bitte nennen Sie mir Ihr Kennzeichen."
        
        query = validated_query
        
        if len(query) < 3:
            return "Für die Rechnungssuche benötige ich einen Namen oder ein Autokennzeichen."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": f"Rechnung Kosten Service {query}",
                        "agent_type": "garage",
                        "top_k": 5,
                        "collection": "garage_management"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        cost_results = []
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            payload = result.get("payload", {})
                            
                            if payload.get("searchable_text") and payload.get("data_type") == "vehicle_complete":
                                content = payload["searchable_text"]
                            
                            if content and any(word in content.lower() for word in ["kosten", "franken", "chf", "rechnung", "service"]):
                                content = self._format_garage_data(content)
                                cost_results.append(content)
                        
                        if cost_results:
                            # Kontext NICHT resetten, da wir ihn noch brauchen könnten
                            return "\n\n".join(cost_results[:2])
                        else:
                            return "Ich habe keine Kosteninformationen gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                    
                    return "Keine Rechnungsdaten vorhanden. Können Sie mir Ihr Autokennzeichen nennen?"
                    
                else:
                    return "Die Rechnungsdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Invoice search error: {e}")
            return "Technischer Fehler bei der Rechnungssuche."

    def _format_garage_data(self, content: str) -> str:
        """Formatiert Garagendaten für bessere Lesbarkeit"""
        # Ersetze Unterstriche
        content = content.replace('_', ' ')
        
        # Formatiere Währungen
        content = re.sub(r'CHF\s*(\d+)\.(\d{2})', r'\1 Franken \2', content)
        content = re.sub(r'(\d+)\.(\d{2})\s*CHF', r'\1 Franken \2', content)
        content = re.sub(r'(\d+)\.00', r'\1 Franken', content)
        
        # Formatiere Datum (YYYY-MM-DD zu DD.MM.YYYY)
        content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3.\2.\1', content)
        
        # Entferne technische Begriffe
        content = content.replace('data_type:', '')
        content = content.replace('primary_key:', '')
        content = content.replace('payload:', '')
        
        return content.strip()


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
        
        # 4. Configure LLM mit optimierten Settings für Llama 3.2
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # OPTIMIERTE Llama 3.2 Konfiguration
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.0  # Minimale Temperatur für maximale Konsistenz
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with anti-hallucination settings")
        
        # 5. Create session mit optimierten VAD-Settings
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                current_customer_id=None,
                active_repair_id=None,
                user_language="de",
                greeting_sent=False,
                customer_name=None,
                conversation_state=ConversationState.GREETING,
                license_plate_context=LicensePlateContext(),
                last_search_query=None,
                hallucination_count=0
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.4,
                min_speech_duration=0.15
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
        
        # Event handlers NACH session.start()
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            
            # Intent-Erkennung für besseres State Management
            intent_result = HalluccinationDetector.extract_intent_from_input(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("function_call")
        def on_function_call(event):
            """Log function calls für Debugging"""
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
        
        # 8. Initial greeting
        await asyncio.sleep(1.5)
        
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = "Guten Tag und herzlich willkommen bei der Garage Müller! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?"
            
            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST
            
            # Verwende say() für die Begrüßung
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )
            
            logger.info(f"✅ [{session_id}] Initial greeting sent")
            
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}")
            # Fallback
            try:
                await session.generate_reply(
                    instructions="Begrüße den Kunden als Pia von Garage Müller. Frage wie du helfen kannst. Auf Deutsch!"
                )
            except:
                pass
        
        logger.info(f"✅ [{session_id}] Garage Agent ready with anti-hallucination measures!")
        
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
