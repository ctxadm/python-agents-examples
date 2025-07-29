# LiveKit Agents - Medical Agent mit korrigierter Kontext-Persistierung
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
logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-medical-3")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")
MEDICAL_COLLECTION = "medical_nutrition"

class ConversationState(Enum):
    """State Machine für Konversationsphasen"""
    GREETING = "greeting"
    AWAITING_REQUEST = "awaiting_request"
    COLLECTING_IDENTIFIER = "collecting_identifier"
    SEARCHING = "searching"
    PROVIDING_INFO = "providing_info"

@dataclass
class PatientContext:
    """Kontext für Patienten-Identifikation"""
    patient_id: Optional[str] = None  # z.B. P001, P002
    patient_name: Optional[str] = None
    attempts: int = 0

    def has_identifier(self) -> bool:
        """Prüft ob eine Identifikation vorhanden ist"""
        return bool(self.patient_id or self.patient_name)

    def get_search_query(self) -> Optional[str]:
        """Gibt die beste Suchanfrage zurück"""
        if self.patient_id:
            return self.patient_id
        elif self.patient_name:
            return self.patient_name
        return None

    def reset(self):
        """Reset des Kontexts"""
        self.patient_id = None
        self.patient_name = None
        self.attempts = 0

@dataclass
class MedicalUserData:
    """User data context für den Medical Agent"""
    authenticated_doctor: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    qdrant_url: str = QDRANT_URL
    current_patient_data: Optional[Dict[str, Any]] = None
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    patient_context: PatientContext = field(default_factory=PatientContext)
    last_search_query: Optional[str] = None
    active_patient: Optional[str] = None
    # Strukturierte Patientendaten
    patient_info: Optional[str] = None
    patient_medication: Optional[str] = None
    patient_treatments: List[str] = field(default_factory=list)
    patient_allergies: Optional[str] = None
    patient_chronic_conditions: Optional[str] = None


class IdentifierExtractor:
    """Extrahiert und validiert Patienten-Identifikatoren"""

    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()

        # Check for specific medical queries FIRST
        if any(word in input_lower for word in ["diagnose", "symptome", "behandlung", "medikation"]):
            return {"intent": "medical_query", "data": user_input}

        if any(word in input_lower for word in ["labor", "blutwerte", "ergebnisse"]):
            return {"intent": "lab_results", "data": user_input}

        if any(word in input_lower for word in ["allergie", "unverträglichkeit"]):
            return {"intent": "allergy_info", "data": user_input}

        # Patienten-ID (P001, P002, etc.) - Check even in longer sentences
        patient_id_match = re.search(r'\b(P\d{3})\b', user_input.upper())
        if patient_id_match:
            return {"intent": "patient_id", "data": patient_id_match.group(1)}

        # Korrigiere Sprache-zu-Text Fehler bei Patienten-IDs
        patterns = [
            r'p\s*null\s*null\s*(\w+)',
            r'patient\s*null\s*null\s*(\w+)',
            r'p\s*0\s*0\s*(\w+)'
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
                corrected_id = f"P{number.zfill(3)}"
                return {"intent": "patient_id", "data": corrected_id}

        # Greetings - but only if no other important info is present
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        has_greeting = any(g in input_lower for g in greetings)

        # Name detection
        name_patterns = [
            r"(?:patient|patientin)\s+(?:heißt|ist)\s+([A-Za-zäöüÄÖÜß\s]+)",
            r"(?:herr|frau)\s+([A-Za-zäöüÄÖÜß]+(?:\s+[A-Za-zäöüÄÖÜß]+)?)"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name.lower() not in ["patient", "patientin", "der", "die", "das"]:
                    return {"intent": "patient_name", "data": name}

        # If we found a greeting but no other data, return greeting
        if has_greeting and not patient_id_match:
            return {"intent": "greeting", "data": None}

        # Medical keywords
        if any(word in input_lower for word in ["patient", "behandlung", "diagnose", "medikament"]):
            return {"intent": "medical_query", "data": user_input}

        # Default
        return {"intent": "general_query", "data": user_input}


class MedicalAssistant(Agent):
    """Medical Assistant für Patientenverwaltung"""

    def __init__(self) -> None:
        # Instructions erweitert für kontextbewusste Funktionen
        super().__init__(instructions="""You are Lisa, the digital assistant of Klinik St. Anna. RESPOND ONLY IN GERMAN.

CRITICAL ANTI-HALLUCINATION RULES:
1. NEVER invent data - if search returns "keine passenden Daten", SAY THAT
2. NEVER claim to have found data when the search failed
3. NEVER make up diagnoses, treatments, or any medical information
4. When you get "keine passenden Daten", ask for identification again
5. ALWAYS acknowledge found symptoms when they are listed in the data

CONTEXT-AWARE FUNCTION USAGE:
1. Use 'search_patient_data' ONLY for:
   - Initial patient lookup (by ID or name)
   - Switching to a different patient
   - When explicitly asked to search again

2. Use 'get_current_patient_details' for:
   - Questions about the already loaded patient
   - Specific details like medication, treatments, allergies
   - Any follow-up questions about the current patient

WORKFLOW EXAMPLE:
User: "P005"
You: [Use search_patient_data("P005")]
User: "Was ist die aktuelle Medikation?"
You: [Use get_current_patient_details("medication")]
User: "Welche chronischen Erkrankungen?"
You: [Use get_current_patient_details("chronic_conditions")]

AVAILABLE DETAIL TYPES for get_current_patient_details:
- "medication" - Aktuelle Medikation
- "treatments" - Letzte Behandlungen
- "allergies" - Allergien
- "chronic_conditions" - Chronische Erkrankungen
- "all" - Alle verfügbaren Informationen

PATIENT IDENTIFICATION OPTIONS:
1. Patienten-ID (z.B. "P001", "P002", etc.) - PREFERRED METHOD
2. Full name (z.B. "Maria Schmidt")

FORBIDDEN WORDS (use alternatives):
- "Entschuldigung" → "Leider"
- "Es tut mir leid" → "Bedauerlicherweise"
- "Sorry" → "Leider"

RESPONSE RULES:
1. Be professional and precise
2. If search returns no data, SAY SO and ask for identification
3. NEVER invent medical information
4. Always acknowledge symptoms found in the data
5. Use the context-aware functions appropriately

Remember: ALWAYS report exactly what the functions return, NEVER invent data!""")

        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ MedicalAssistant initialized with Patient-ID support")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    def _extract_patient_data_from_content(self, content: str, context: RunContext[MedicalUserData]):
        """Extrahiert strukturierte Daten aus dem Content und speichert sie im Context"""
        # Extract patient ID from content
        patient_id_match = re.search(r'Patient:\s*(P\d{3})', content)
        if not patient_id_match:
            patient_id_match = re.search(r'(P\d{3})', content)
        
        if patient_id_match:
            patient_id = patient_id_match.group(1)
            context.userdata.active_patient = patient_id
            context.userdata.patient_context.patient_id = patient_id
            logger.info(f"✅ Set active patient to: {patient_id}")
        
        # Extract patient name
        name_match = re.search(r'Patient:\s*([^\n]+)', content)
        if name_match and not name_match.group(1).startswith('P'):
            patient_name = name_match.group(1).strip()
            context.userdata.patient_context.patient_name = patient_name
        
        # Extract allergies
        if "Allergien:" in content:
            allergies_match = re.search(r'Allergien:\s*([^\n]+)', content)
            if allergies_match:
                context.userdata.patient_allergies = allergies_match.group(1).strip()
                
        # Extract chronic conditions
        if "Chronische Erkrankungen:" in content:
            chronic_match = re.search(r'Chronische Erkrankungen:\s*([^\n]+)', content)
            if chronic_match:
                context.userdata.patient_chronic_conditions = chronic_match.group(1).strip()

    @function_tool
    async def search_patient_data(self,
                                 context: RunContext[MedicalUserData],
                                 query: str) -> str:
        """
        Sucht nach Patientendaten in der medizinischen Datenbank basierend auf Patienten-ID oder Namen.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Patientendaten gesucht werden soll.

        Args:
            query: Suchbegriff (Patienten-ID oder Name)

        Returns:
            Gefundene Patientendaten oder Fehlermeldung
        """
        logger.info(f"🔍 Original search query: {query}")

        # Extract intent from query
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]

        logger.info(f"📊 Intent: {intent}, Data: {data}")

        # Store identification in context
        if intent == "patient_id":
            context.userdata.patient_context.patient_id = data
            query = data
        elif intent == "patient_name":
            context.userdata.patient_context.patient_name = data
            query = data

        # Process queries for common speech-to-text errors
        processed_query = self._process_patient_id(query)
        if processed_query != query:
            logger.info(f"✅ Corrected query from '{query}' to '{processed_query}'")
            query = processed_query

        # Store search query
        context.userdata.last_search_query = query

        # Clear previous patient data
        context.userdata.patient_info = None
        context.userdata.patient_medication = None
        context.userdata.patient_treatments = []
        context.userdata.patient_allergies = None
        context.userdata.patient_chronic_conditions = None
        context.userdata.active_patient = None  # WICHTIG: Reset active patient

        try:
            # NEUE DIREKTE QDRANT INTEGRATION
            
            # Check if query is a patient ID (P followed by 3 digits)
            is_patient_id = bool(re.match(r'^P\d{3}$', query.upper()))
            
            if is_patient_id:
                # Direkte Qdrant-Abfrage für Patienten-IDs
                logger.info(f"🔍 Direct Qdrant search for patient ID: {query}")
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{context.userdata.qdrant_url}/collections/{MEDICAL_COLLECTION}/points/scroll",
                        json={
                            "filter": {
                                "must": [
                                    {
                                        "key": "patient_id",
                                        "match": {
                                            "value": query.upper()
                                        }
                                    }
                                ]
                            },
                            "limit": 20,
                            "with_payload": True,
                            "with_vector": False
                        },
                        timeout=5.0
                    )

                    if response.status_code == 200:
                        qdrant_data = response.json()
                        points = qdrant_data.get("result", {}).get("points", [])
                        
                        if points:
                            logger.info(f"✅ Found {len(points)} results from Qdrant")
                            
                            # Store patient ID
                            context.userdata.active_patient = query.upper()
                            context.userdata.patient_context.patient_id = query.upper()
                            
                            # Format results by data type and store in context
                            patient_info = None
                            medication = None
                            treatments = []
                            
                            for point in points:
                                payload = point.get("payload", {})
                                data_type = payload.get("data_type", "")
                                content = payload.get("content", "")
                                
                                if data_type == "patient_info":
                                    patient_info = content
                                    context.userdata.patient_info = content
                                    # Extract data from content
                                    self._extract_patient_data_from_content(content, context)
                                elif data_type == "medication":
                                    medication = content
                                    context.userdata.patient_medication = content
                                elif data_type == "treatment":
                                    treatments.append(content)
                            
                            # Store treatments
                            context.userdata.patient_treatments = treatments
                            
                            # Build response
                            response_text = "Ich habe folgende Patientendaten gefunden:\n\n"
                            
                            if patient_info:
                                response_text += f"**Patientendaten:**\n{patient_info}\n\n"
                            
                            if medication:
                                response_text += f"**{medication}\n\n"
                            
                            if treatments:
                                response_text += "**Behandlungen:**\n"
                                for treatment in treatments[:3]:  # Limit to 3 most recent
                                    response_text += f"{treatment}\n\n"
                            
                            # Update conversation state
                            context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                            
                            return response_text.strip()
                        
                        else:
                            logger.info("❌ No results found in Qdrant")
                            context.userdata.patient_context.reset()
                            return "Ich habe keine Patientendaten für diese ID gefunden. Bitte überprüfen Sie die Patienten-ID."
                    
                    else:
                        logger.error(f"Qdrant error: {response.status_code}")
                        # Fallback to RAG service
                        return await self._search_via_rag_service(context, query)
            
            else:
                # Für Namen-Suchen verwenden wir weiterhin den RAG Service (Vektor-Suche)
                logger.info(f"🔍 Using RAG service for name search: {query}")
                return await self._search_via_rag_service(context, query)

        except Exception as e:
            logger.error(f"Search error: {e}")
            # Fallback to RAG service
            try:
                return await self._search_via_rag_service(context, query)
            except:
                return "Die Patientendatenbank ist momentan nicht erreichbar. Bitte versuchen Sie es später noch einmal."

    @function_tool
    async def get_current_patient_details(self,
                                        context: RunContext[MedicalUserData],
                                        detail_type: str) -> str:
        """
        Gibt spezifische Details des aktuell geladenen Patienten zurück.
        Verwenden Sie diese Funktion NUR wenn bereits ein Patient geladen wurde.

        Args:
            detail_type: Art der gewünschten Information 
                        ("medication", "treatments", "allergies", "chronic_conditions", "all")

        Returns:
            Die angeforderten Patientendetails oder eine Fehlermeldung
        """
        logger.info(f"📋 Getting current patient details: {detail_type}")
        logger.info(f"📋 Active patient: {context.userdata.active_patient}")
        logger.info(f"📋 Patient context ID: {context.userdata.patient_context.patient_id}")
        
        # Check if we have a patient loaded
        if not context.userdata.active_patient and not context.userdata.patient_context.patient_id:
            return "Es ist kein Patient geladen. Bitte geben Sie zuerst eine Patienten-ID oder einen Namen an."
        
        patient_id = context.userdata.active_patient or context.userdata.patient_context.patient_id
        response_text = f"Details für Patient {patient_id}:\n\n"
        
        if detail_type == "medication" or detail_type == "all":
            if context.userdata.patient_medication:
                response_text += f"**{context.userdata.patient_medication}\n\n"
            else:
                response_text += "**Medikation:** Keine Medikationsdaten gefunden.\n\n"
        
        if detail_type == "treatments" or detail_type == "all":
            if context.userdata.patient_treatments:
                response_text += "**Letzte Behandlungen:**\n"
                for treatment in context.userdata.patient_treatments:
                    response_text += f"{treatment}\n\n"
            else:
                response_text += "**Behandlungen:** Keine Behandlungsdaten gefunden.\n\n"
        
        if detail_type == "allergies" or detail_type == "all":
            if context.userdata.patient_allergies:
                response_text += f"**Allergien:** {context.userdata.patient_allergies}\n\n"
            else:
                response_text += "**Allergien:** Keine Allergiedaten gefunden.\n\n"
        
        if detail_type == "chronic_conditions" or detail_type == "all":
            if context.userdata.patient_chronic_conditions:
                response_text += f"**Chronische Erkrankungen:** {context.userdata.patient_chronic_conditions}\n\n"
            else:
                response_text += "**Chronische Erkrankungen:** Keine Daten zu chronischen Erkrankungen gefunden.\n\n"
        
        if detail_type not in ["medication", "treatments", "allergies", "chronic_conditions", "all"]:
            return f"Unbekannter Detail-Typ: {detail_type}. Verfügbare Optionen: medication, treatments, allergies, chronic_conditions, all"
        
        return response_text.strip()

    async def _search_via_rag_service(self, context: RunContext[MedicalUserData], query: str) -> str:
        """Fallback Suche über RAG Service"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{context.userdata.rag_url}/search",
                json={
                    "query": query,
                    "agent_type": "medical",
                    "top_k": 5,
                    "collection": "medical_nutrition"
                }
            )

            if response.status_code == 200:
                results = response.json().get("results", [])

                if results:
                    logger.info(f"✅ Found {len(results)} results from RAG")

                    # WICHTIG: Extrahiere Patient ID aus den Ergebnissen
                    for result in results:
                        content = result.get("content", "")
                        self._extract_patient_data_from_content(content, context)

                    # Format results
                    formatted = []
                    patient_info = None
                    medication = None
                    treatments = []
                    
                    for i, result in enumerate(results[:5]):
                        content = result.get("content", "").strip()
                        metadata = result.get("metadata", {})
                        data_type = metadata.get("data_type", "")
                        
                        if content:
                            content = self._format_medical_data(content)
                            
                            if data_type == "patient_info" and not patient_info:
                                patient_info = content
                                context.userdata.patient_info = content
                            elif data_type == "medication" and not medication:
                                medication = content
                                context.userdata.patient_medication = content
                            elif data_type == "treatment":
                                treatments.append(content)

                    # Store treatments
                    context.userdata.patient_treatments = treatments

                    # Build structured response
                    response_text = "Ich habe folgende Patientendaten gefunden:\n\n"
                    
                    if patient_info:
                        response_text += f"**Patientendaten:**\n{patient_info}\n\n"
                    
                    if medication:
                        response_text += f"**{medication}\n\n"
                    
                    if treatments:
                        response_text += "**Behandlungen:**\n"
                        for treatment in treatments[:3]:
                            response_text += f"{treatment}\n\n"
                    
                    # Update conversation state
                    context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                    
                    return response_text.strip()

                else:
                    logger.info("❌ No results found from RAG")
                    context.userdata.patient_context.reset()
                    context.userdata.conversation_state = ConversationState.COLLECTING_IDENTIFIER
                    return "Ich habe keine passenden Patientendaten gefunden. Können Sie mir bitte die Patienten-ID (z.B. P001) oder den vollständigen Namen des Patienten nennen?"
            else:
                logger.error(f"RAG search failed: {response.status_code}")
                return "Es gab ein Problem mit der Datenbank-Verbindung. Bitte versuchen Sie es erneut."

    def _process_patient_id(self, text: str) -> str:
        """Korrigiert Sprache-zu-Text Fehler bei Patienten-IDs"""
        # Pattern für verschiedene Varianten
        patterns = [
            r'p\s*null\s*null\s*(\w+)',
            r'patient\s*null\s*null\s*(\w+)',
            r'p\s*0\s*0\s*(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
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

                # Erstelle korrekte ID
                corrected_id = f"P{number.zfill(3)}"
                text = re.sub(pattern, corrected_id, text, flags=re.IGNORECASE)
                logger.info(f"✅ Corrected patient ID to '{corrected_id}'")
                break

        return text

    def _format_medical_data(self, content: str) -> str:
        """Formatiert medizinische Daten für bessere Lesbarkeit"""
        # Ersetze Unterstriche durch Leerzeichen
        content = content.replace('_', ' ')

        # Formatiere Währungen
        content = re.sub(r'(\d+)\.(\d{2})', r'\1 Franken \2', content)

        return content


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Medical Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"

    logger.info("="*50)
    logger.info(f"🏥 Starting Medical Agent Session: {session_id}")
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

        # 4. Configure LLM with Ollama - KORREKTE API WIE GARAGE AGENT!
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        qdrant_url = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")

        # Llama 3.2 with Ollama configuration - GENAU WIE GARAGE AGENT
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,  # Deterministisch für medizinische Präzision
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with anti-hallucination settings")

        # 5. Create session - EXAKT WIE GARAGE AGENT
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                authenticated_doctor=None,
                rag_url=rag_url,
                qdrant_url=qdrant_url,
                current_patient_data=None,
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                patient_context=PatientContext(),
                last_search_query=None,
                active_patient=None,
                patient_info=None,
                patient_medication=None,
                patient_treatments=[],
                patient_allergies=None,
                patient_chronic_conditions=None
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
        agent = MedicalAssistant()

        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )

        # Event handlers
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

        # 8. Initial greeting
        await asyncio.sleep(1.5)

        logger.info(f"📢 [{session_id}] Sending initial greeting...")

        try:
            greeting_text = """Guten Tag und herzlich willkommen bei der Klinik St. Anna!
Ich bin Lisa, Ihre digitale medizinische Assistentin.

Für eine schnelle Bearbeitung benötige ich eine der folgenden Informationen:
- Die Patienten-ID (z.B. P001)
- Den vollständigen Namen des Patienten

Welche Patientendaten benötigen Sie heute, Herr Doktor?"""

            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST

            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )

            logger.info(f"✅ [{session_id}] Initial greeting sent")

        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}", exc_info=True)

        logger.info(f"✅ [{session_id}] Medical Agent ready with Patient-ID support!")

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
