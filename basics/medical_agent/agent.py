# LiveKit Agents - Medical Agent (Moderne API wie Garage Agent)
import logging
import os
import httpx
import asyncio
import re
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-medical-3")

@dataclass
class MedicalUserData:
    """User data context für den Medical Agent"""
    authenticated_doctor: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    active_patient: Optional[str] = None


class MedicalAssistant(Agent):
    """Medical Assistant mit korrekter API-Nutzung"""

    def __init__(self) -> None:
        # Instructions ANGEPASST für Medical Context
        super().__init__(instructions="""Du bist ein medizinischer Assistent der Klinik St. Anna. ANTWORTE NUR AUF DEUTSCH.

KRITISCHE ANTI-HALLUZINATIONS-REGELN:
1. ERFINDE NIEMALS Daten - wenn die Suche "keine passenden Daten" zurückgibt, SAGE DAS
2. Behaupte NIEMALS, Daten gefunden zu haben, wenn die Suche fehlgeschlagen ist
3. Erfinde NIEMALS Diagnosen, Behandlungen oder medizinische Informationen
4. Wenn du "keine passenden Daten" erhältst, frage erneut nach der Patienten-ID
5. Bestätige IMMER gefundene Probleme, wenn sie in den Daten aufgelistet sind

WENN DATEN MIT PROBLEMEN GEFUNDEN WERDEN:
Wenn das Tool Daten mit "Aktuelle Symptome" zurückgibt wie:
- Kopfschmerzen seit 3 Tagen
- Erhöhte Temperatur

Du MUSST sagen:
"Ich sehe bei Patient [Name] folgende dokumentierte Symptome:
- [Symptom 1]
- [Symptom 2]
Möchten Sie weitere Details zu diesen Symptomen?"

NIEMALS sagen "keine spezifischen Probleme gefunden" wenn Probleme AUFGELISTET sind!

PATIENTEN-IDENTIFIKATION:
1. Patienten-ID (z.B. "P001", "P002", etc.) - BEVORZUGTE METHODE
2. Vollständiger Name (z.B. "Maria Schmidt")

KONVERSATIONSBEISPIELE:

Beispiel 1 - Mit Patienten-ID:
User: "Meine Patienten-ID ist P001"
Du: [SUCHE mit "P001"]

Beispiel 2 - Spezifische Anfrage:
User: "Was ist die Diagnose?"
Du: [Verwende search_patient_data um Diagnose zu finden]

VERBOTENE WÖRTER (verwende Alternativen):
- "Entschuldigung" → "Leider"
- "Es tut mir leid" → "Bedauerlicherweise"
- "Sorry" → "Leider"

ANTWORT-REGELN:
1. Sei professionell und präzise
2. Wenn die Suche keine Daten zurückgibt, SAGE ES und frage nach der Patienten-ID
3. Erfinde NIEMALS medizinische Informationen
4. Berichte IMMER genau was die Suche zurückgibt
5. Schlage die Verwendung der Patienten-ID für schnelleren Service vor

Denke daran: Melde IMMER genau, was die Suche zurückgibt, erfinde NIEMALS Daten!""")
        logger.info("✅ MedicalAssistant initialized")

    @function_tool
    async def search_patient_data(self,
                                 context: RunContext[MedicalUserData],
                                 query: str) -> str:
        """
        Sucht in der Patientendatenbank nach Informationen.

        Args:
            query: Die Suchanfrage (z.B. Patienten-ID oder Symptome)
        """
        logger.info(f"🔍 Searching for: {query}")

        try:
            # Korrigiere Patienten-IDs
            processed_query = self._process_patient_id(query)
            logger.info(f"🔎 Processed query: {processed_query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "medical",
                        "top_k": 5,
                        "collection": "medical_nutrition"  # MEDICAL COLLECTION
                    }
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])

                    if results:
                        logger.info(f"✅ Found {len(results)} results")

                        # Speichere aktuelle Patienten-ID wenn gefunden
                        patient_match = re.search(r'P\d{3}', processed_query)
                        if patient_match:
                            context.userdata.active_patient = patient_match.group()

                        # Formatiere die Ergebnisse
                        formatted = []
                        for i, result in enumerate(results[:3]):  # Max 3 Ergebnisse
                            content = result.get("content", "").strip()
                            if content:
                                # Formatiere für bessere Lesbarkeit
                                content = self._format_medical_data(content)
                                formatted.append(f"[{i+1}] {content}")

                        response_text = "Hier sind die Patientendaten:\n\n"
                        response_text += "\n\n".join(formatted)
                        return response_text

                    return "Zu dieser Anfrage konnte ich keine Daten in der Patientendatenbank finden."

                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return "Es gab einen Fehler beim Zugriff auf die Datenbank. Bitte versuchen Sie es erneut."

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
    """Entry point EXAKT WIE GARAGE AGENT"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"

    logger.info("="*50)
    logger.info(f"🚀 Starting Medical Agent Session: {session_id}")
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

        # 4. Configure LLM with Ollama - EXAKT WIE GARAGE AGENT
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

        # Llama 3.2 with Ollama configuration - EXAKT WIE GARAGE AGENT
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3  # Niedrig für medizinische Präzision (Garage hat 0.0)
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")

        # 5. Create session - MEDIZINISCHE PARAMETER
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                authenticated_doctor=None,
                rag_url=rag_url,
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
                voice="shimmer"  # Professionelle Stimme
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

        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")

        @session.on("function_call")
        def on_function_call(event):
            logger.info(f"[{session_id}] 🔧 Function call: {event}")

        # 8. Initial greeting - EXAKT WIE GARAGE AGENT
        await asyncio.sleep(1.5)

        logger.info(f"📢 [{session_id}] Sending initial greeting...")

        try:
            # MEDIZINISCHER GREETING TEXT
            greeting_text = """Guten Tag Herr Doktor, willkommen bei der Klinik St. Anna!
Ich bin Ihr digitaler Assistent.

Für eine schnelle Bearbeitung benötige ich:
- Die Patienten-ID (z.B. P001)
- Oder den vollständigen Namen des Patienten

Welche Patientendaten benötigen Sie?"""

            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )

            logger.info(f"✅ [{session_id}] Initial greeting sent")

        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}")

        logger.info(f"✅ [{session_id}] Medical Agent ready!")

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
