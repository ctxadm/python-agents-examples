# LiveKit Agents - Vereinfachter Medical Agent
import logging
import os
import httpx
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
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

# Agent Name
AGENT_NAME = os.getenv("AGENT_NAME", "agent-medical-3")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")
MEDICAL_COLLECTION = "medical_nutrition"

@dataclass
class MedicalUserData:
    """User data context"""
    qdrant_url: str = QDRANT_URL
    greeting_sent: bool = False
    # Aktueller Patient
    patient_name: Optional[str] = None
    patient_birth_year: Optional[int] = None
    patient_loaded: bool = False


class MedicalAssistant(Agent):
    """Vereinfachter Medical Assistant"""

    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Lisa von der Klinik St. Anna. ANTWORTE NUR AUF DEUTSCH.

DEINE AUFGABE:
1. Frage nach dem Namen des Patienten
2. Frage nach dem Geburtsjahr
3. Verwende 'load_patient_data' um ALLE Daten zu laden
4. Lies die kompletten Daten vor (Medikation und letzte Behandlungen)

WICHTIG:
- Erfinde NIEMALS Daten
- Wenn die Funktion Daten zurückgibt, lies sie KOMPLETT vor
- Vergiss keine Details""")

        logger.info("✅ MedicalAssistant initialized")

    @function_tool
    async def load_patient_data(self,
                               context: RunContext[MedicalUserData],
                               name: str,
                               birth_year: int) -> str:
        """
        Lädt ALLE Patientendaten basierend auf Name und Geburtsjahr.
        
        Args:
            name: Vollständiger Name des Patienten
            birth_year: Geburtsjahr (z.B. 2010)
            
        Returns:
            Alle gefundenen Patientendaten formatiert
        """
        logger.info(f"🔍 Loading patient data for: {name}, born {birth_year}")
        
        # Speichere in Context
        context.userdata.patient_name = name
        context.userdata.patient_birth_year = birth_year
        
        try:
            async with httpx.AsyncClient() as client:
                # Hole ALLE Dokumente
                response = await client.post(
                    f"{context.userdata.qdrant_url}/collections/{MEDICAL_COLLECTION}/points/scroll",
                    json={
                        "limit": 100,
                        "with_payload": true,
                        "with_vector": false
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    qdrant_data = response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    logger.info(f"📊 Checking {len(points)} documents")
                    
                    # Suche nach dem Patienten
                    patient_docs = []
                    patient_id = None
                    
                    for point in points:
                        payload = point.get("payload", {})
                        content = payload.get("content", "")
                        stored_name = payload.get("patient_name", "")
                        
                        # Prüfe Name (case-insensitive)
                        if name.lower() in stored_name.lower():
                            # Prüfe Geburtsjahr im Content
                            if str(birth_year) in content:
                                patient_docs.append(point)
                                if not patient_id:
                                    patient_id = payload.get("patient_id", "")
                                    logger.info(f"✅ Found patient: {stored_name} ({patient_id})")
                    
                    if patient_docs:
                        # Sortiere nach Datentyp
                        patient_info = None
                        medication = None
                        treatments = []
                        
                        for doc in patient_docs:
                            payload = doc.get("payload", {})
                            content = payload.get("content", "")
                            data_type = payload.get("data_type", "")
                            
                            if data_type == "patient_info":
                                patient_info = content
                            elif data_type == "medication":
                                medication = content
                            elif data_type == "treatment":
                                treatments.append(content)
                        
                        # Erstelle formatierte Antwort
                        result = f"=== PATIENTENDATEN FÜR {name.upper()} ===\n\n"
                        
                        if patient_info:
                            result += f"📋 STAMMDATEN:\n{patient_info}\n\n"
                        
                        if medication:
                            result += f"💊 AKTUELLE MEDIKATION:\n{medication}\n\n"
                        
                        if treatments:
                            result += "🏥 LETZTE BEHANDLUNGEN:\n"
                            # Sortiere nach Datum (neueste zuerst)
                            sorted_treatments = sorted(treatments, reverse=True)
                            for treatment in sorted_treatments[:5]:  # Max 5 neueste
                                result += f"{treatment}\n"
                        
                        result += "\n=== ENDE DER PATIENTENDATEN ==="
                        
                        context.userdata.patient_loaded = True
                        logger.info(f"✅ Successfully loaded data for {name}")
                        
                        return result
                    
                    else:
                        logger.info(f"❌ No patient found: {name}, {birth_year}")
                        return f"Ich habe keinen Patienten namens {name} mit Geburtsjahr {birth_year} gefunden. Bitte überprüfen Sie die Angaben."
                
                else:
                    logger.error(f"Qdrant error: {response.status_code}")
                    return "Die Datenbank ist momentan nicht erreichbar."
                    
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."


async def request_handler(ctx: JobContext):
    """Request handler"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point"""
    session_id = f"{ctx.room.name}_{int(asyncio.get_event_loop().time())}"
    logger.info(f"🏥 Starting session: {session_id}")

    session = None

    try:
        # Connect to room
        await ctx.connect()
        logger.info(f"✅ Connected to room")

        # Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ Participant joined: {participant.identity}")

        # Configure Ollama
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,
        )

        # Create session
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                qdrant_url=QDRANT_URL,
                greeting_sent=False,
                patient_name=None,
                patient_birth_year=None,
                patient_loaded=False
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

        # Create and start agent
        agent = MedicalAssistant()
        
        await session.start(
            room=ctx.room,
            agent=agent
        )

        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"🎤 User: {event.transcript}")

        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"🤖 Agent state: {event}")

        # Initial greeting
        await asyncio.sleep(1.5)

        greeting_text = """Guten Tag! Ich bin Lisa von der Klinik St. Anna.

Ich werde Ihnen die kompletten Patientendaten vorlesen, inklusive der aktuellen Medikation und der letzten Behandlungen.

Dafür benötige ich:
1. Den vollständigen Namen des Patienten
2. Das Geburtsjahr

Wie heißt der Patient?"""

        await session.say(
            greeting_text,
            allow_interruptions=True,
            add_to_chat_ctx=True
        )

        logger.info(f"✅ Agent ready!")

        # Wait for disconnect
        disconnect_event = asyncio.Event()

        def handle_disconnect():
            disconnect_event.set()

        ctx.room.on("disconnected", handle_disconnect)
        await disconnect_event.wait()

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise

    finally:
        if session:
            try:
                await session.aclose()
                logger.info(f"✅ Session closed")
            except:
                pass


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
