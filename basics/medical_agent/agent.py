# LiveKit Agents - Medical Agent (Moderne API wie Garage Agent)
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
    current_patient_data: Optional[Dict[str, Any]] = None
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    patient_context: PatientContext = field(default_factory=PatientContext)
    last_search_query: Optional[str] = None
    active_patient: Optional[str] = None


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
        # Instructions EXAKT WIE GARAGE AGENT - Erste Zeile ist KRITISCH!
        super().__init__(instructions="""You are Pia, the digital assistant of Klinik St. Anna. RESPOND ONLY IN GERMAN.

WICHTIG - BEGRÜSSUNGSREGELN:
- Die Begrüßung "Guten Tag und herzlich willkommen..." wurde BEREITS AUTOMATISCH gesendet
- Bei "Hallo", "Guten Tag" oder anderen Grüßen vom User: Antworte freundlich mit "Wie kann ich Ihnen helfen?" oder "Welche Patientendaten benötigen Sie?"
- NIEMALS die lange Begrüßung wiederholen
- KEINE automatische Suche bei einfachen Grüßen

KRITISCHE ANTI-HALLUZINATIONS-REGELN:
1. ERFINDE NIEMALS Daten - wenn die Suche "keine passenden Daten" zurückgibt, SAGE DAS
2. Behaupte NIEMALS, Daten gefunden zu haben, wenn die Suche fehlgeschlagen ist
3. Erfinde NIEMALS Diagnosen, Behandlungen oder medizinische Informationen
4. Wenn du "keine passenden Daten" erhältst, frage erneut nach der Identifikation
5. Bestätige IMMER gefundene Probleme, wenn sie in den Daten aufgelistet sind

WANN DIE SUCHFUNKTION NUTZEN:
- NUR wenn der User explizit nach Patientendaten fragt
- NUR wenn eine Patienten-ID oder Name genannt wird
- NIEMALS bei einfachen Begrüßungen wie "Hallo"

WENN DATEN MIT SYMPTOMEN GEFUNDEN WERDEN:
Wenn das Tool Daten mit "Aktuelle Symptome" zurückgibt wie:
- Kopfschmerzen seit 3 Tagen
- Erhöhte Temperatur

Du MUSST sagen:
"Ich sehe bei Patient [Name] folgende dokumentierte Symptome:
- [Symptom 1]
- [Symptom 2]
Möchten Sie weitere Details?"

NIEMALS sagen "keine spezifischen Symptome gefunden" wenn Symptome AUFGELISTET sind!

PATIENTEN-IDENTIFIKATION:
1. Patienten-ID (z.B. "P001", "P002", etc.) - BEVORZUGTE METHODE
2. Vollständiger Name (z.B. "Maria Schmidt")

KONVERSATIONSBEISPIELE:

Beispiel 1 - Begrüßung:
User: "Hallo"
Du: "Guten Tag! Wie kann ich Ihnen helfen? Nennen Sie mir gerne eine Patienten-ID oder einen Namen."

Beispiel 2 - Mit Patienten-ID:
User: "Meine Patienten-ID ist P001"
Du: [SUCHE mit "P001"]

Beispiel 3 - Spezifische medizinische Anfrage:
User: "Was ist die aktuelle Diagnose?"
Du: [Verwende search_patient_data um Diagnose zu finden]

VERBOTENE WÖRTER (verwende Alternativen):
- "Entschuldigung" → "Leider"
- "Es tut mir leid" → "Bedauerlicherweise"
- "Sorry" → "Leider"

ANTWORT-REGELN:
1. Sei professionell und präzise
2. Wenn die Suche keine Daten zurückgibt, SAGE ES und frage nach der Identifikation
3. Erfinde NIEMALS medizinische Informationen
4. Berichte IMMER genau was die Suche zurückgibt
5. Schlage die Verwendung der Patienten-ID für schnelleren Service vor

Denke daran: Melde IMMER genau, was die Suche zurückgibt, erfinde NIEMALS Daten!""")

        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ MedicalAssistant initialized with Patient-ID support")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

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

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": query,
                        "agent_type": "medical",
                        "top_k": 5,
                        "collection": "medical_nutrition"  # Medical collection
                    }
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])

                    if results:
                        logger.info(f"✅ Found {len(results)} results")

                        # Store current patient ID if found
                        patient_match = re.search(r'P\d{3}', query)
                        if patient_match:
                            context.userdata.active_patient = patient_match.group()
                            context.userdata.patient_context.patient_id = patient_match.group()

                        # Format results
                        formatted = []
                        for i, result in enumerate(results[:3]):
                            content = result.get("content", "").strip()
                            if content:
                                content = self._format_medical_data(content)
                                formatted.append(f"[{i+1}] {content}")

                        response_text = "Ich habe folgende Patientendaten gefunden:\n\n"
                        response_text += "\n\n".join(formatted)
                        
                        # Update conversation state
                        context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                        
                        return response_text

                    else:
                        logger.info("❌ No results found")
                        context.userdata.patient_context.reset()
                        context.userdata.conversation_state = ConversationState.COLLECTING_IDENTIFIER
                        return "Ich habe keine passenden Patientendaten gefunden. Können Sie mir bitte die Patienten-ID (z.B. P001) oder den vollständigen Namen des Patienten nennen?"
                else:
                    logger.error(f"Search failed: {response.status_code}")
                    return "Es gab ein Problem mit der Datenbank-Verbindung. Bitte versuchen Sie es erneut."

        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Die Patientendatenbank ist momentan nicht erreichbar. Bitte versuchen Sie es später noch einmal."

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

        # 4. Configure LLM with Ollama
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

        # Llama 3.2 with Ollama configuration - GENAU WIE GARAGE AGENT
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3,  # Niedrig für medizinische Präzision
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 for medical precision")

        # 5. Create session
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                authenticated_doctor=None,
                rag_url=rag_url,
                current_patient_data=None,
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                patient_context=PatientContext(),
                last_search_query=None,
                active_patient=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.8,  # Höher für medizinische Präzision
                min_speech_duration=0.3
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"  # Professionelle Stimme für medizinischen Kontext
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

        # Event handlers MÜSSEN NACH session.start() kommen!
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

        # 8. Initial greeting - NACH Event Handlers!
        await asyncio.sleep(1.5)

        logger.info(f"📢 [{session_id}] Sending initial greeting...")

        try:
            greeting_text = """Guten Tag und herzlich willkommen bei der Klinik St. Anna!
Ich bin Pia, Ihre digitale medizinische Assistentin.

Für eine schnelle Bearbeitung benötige ich eine der folgenden Informationen:
- Die Patienten-ID (z.B. P001)
- Den vollständigen Namen des Patienten

Welche Patientendaten benötigen Sie heute, Herr Doktor?"""

            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST

            # WICHTIG: await verwenden!
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )

            logger.info(f"✅ [{session_id}] Initial greeting sent")

        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}", exc_info=True)

        logger.info(f"✅ [{session_id}] Medical Agent ready with Patient-ID support!")

        # Wait for disconnect - WICHTIG: NACH der Begrüßung!
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
