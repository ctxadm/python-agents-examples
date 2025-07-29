# LiveKit Agents - Medical Agent mit direkter Qdrant-Abfrage
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

@dataclass
class MedicalUserData:
    """User data context für den Medical Agent"""
    rag_url: str = "http://localhost:8000"
    qdrant_url: str = QDRANT_URL
    greeting_sent: bool = False
    # Vereinfachte Speicherung
    active_patient: Optional[str] = None
    patient_name: Optional[str] = None
    patient_info: Optional[str] = None
    patient_medication: Optional[str] = None
    patient_treatments: List[str] = field(default_factory=list)
    patient_allergies: Optional[str] = None
    patient_chronic_conditions: Optional[str] = None


class MedicalAssistant(Agent):
    """Medical Assistant für Patientenverwaltung"""

    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Lisa, die digitale Assistentin der Klinik St. Anna. ANTWORTE NUR AUF DEUTSCH.

WICHTIGE REGELN:
1. Verwende 'search_patient_data' für neue Patientensuchen
2. Verwende 'get_current_patient_details' für Details des geladenen Patienten
3. Erfinde NIEMALS Daten - sage wenn keine Daten gefunden wurden
4. Melde IMMER genau was die Funktionen zurückgeben""")

        logger.info("✅ MedicalAssistant initialized")

    @function_tool
    async def search_patient_data(self,
                                 context: RunContext[MedicalUserData],
                                 query: str) -> str:
        """
        Sucht nach Patientendaten - akzeptiert Patienten-ID (P001-P005) oder Namen.
        
        Args:
            query: Suchbegriff (ID oder Name)
            
        Returns:
            Formatierte Patientendaten oder Fehlermeldung
        """
        logger.info(f"🔍 Searching for: {query}")
        
        # Reset previous data
        context.userdata.active_patient = None
        context.userdata.patient_name = None
        context.userdata.patient_info = None
        context.userdata.patient_medication = None
        context.userdata.patient_treatments = []
        context.userdata.patient_allergies = None
        context.userdata.patient_chronic_conditions = None
        
        try:
            # IMMER direkte Qdrant-Abfrage!
            async with httpx.AsyncClient() as client:
                # Erste Abfrage: Scroll durch ALLE Dokumente
                response = await client.post(
                    f"{context.userdata.qdrant_url}/collections/{MEDICAL_COLLECTION}/points/scroll",
                    json={
                        "limit": 100,  # Mehr Ergebnisse holen
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    qdrant_data = response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    logger.info(f"📊 Got {len(points)} total documents from Qdrant")
                    
                    # Suche in allen Dokumenten
                    matching_docs = []
                    found_patient_id = None
                    found_patient_name = None
                    
                    # Normalisiere Suchanfrage
                    search_normalized = query.upper().strip()
                    
                    for point in points:
                        content = point.get("payload", {}).get("content", "")
                        metadata = point.get("payload", {}).get("metadata", {})
                        
                        # Suche nach Patient ID
                        if re.match(r'^P\d{3}$', search_normalized):
                            # Suche nach exakter Patient ID
                            if search_normalized in content:
                                matching_docs.append(point)
                                found_patient_id = search_normalized
                        else:
                            # Suche nach Namen (case-insensitive)
                            if query.lower() in content.lower():
                                matching_docs.append(point)
                                # Extrahiere Patient ID aus Content
                                id_match = re.search(r'(P\d{3})', content)
                                if id_match and not found_patient_id:
                                    found_patient_id = id_match.group(1)
                                # Extrahiere Namen
                                if not found_patient_name and "Patient:" in content:
                                    name_match = re.search(r'Patient:\s*([^\n]+)', content)
                                    if name_match:
                                        found_patient_name = name_match.group(1).strip()
                    
                    if matching_docs:
                        logger.info(f"✅ Found {len(matching_docs)} matching documents")
                        
                        # Setze aktiven Patienten
                        if found_patient_id:
                            context.userdata.active_patient = found_patient_id
                            logger.info(f"✅ Set active patient: {found_patient_id}")
                        
                        if found_patient_name:
                            context.userdata.patient_name = found_patient_name
                            logger.info(f"✅ Found patient name: {found_patient_name}")
                        
                        # Verarbeite Dokumente nach Typ
                        patient_info = None
                        medication = None
                        treatments = []
                        
                        for doc in matching_docs:
                            content = doc.get("payload", {}).get("content", "")
                            metadata = doc.get("payload", {}).get("metadata", {})
                            data_type = metadata.get("data_type", "")
                            
                            # Bestimme Typ aus Content wenn metadata leer
                            if not data_type:
                                if "Geburtsdatum:" in content or "Blutgruppe:" in content:
                                    data_type = "patient_info"
                                elif "Medikation" in content:
                                    data_type = "medication"
                                elif "Behandlung:" in content:
                                    data_type = "treatment"
                            
                            if data_type == "patient_info" and not patient_info:
                                patient_info = content
                                context.userdata.patient_info = content
                                # Extrahiere Details
                                if "Allergien:" in content:
                                    allergies_match = re.search(r'Allergien:\s*([^\n]+)', content)
                                    if allergies_match:
                                        context.userdata.patient_allergies = allergies_match.group(1)
                                if "Chronische Erkrankungen:" in content:
                                    chronic_match = re.search(r'Chronische Erkrankungen:\s*([^\n]+)', content)
                                    if chronic_match:
                                        context.userdata.patient_chronic_conditions = chronic_match.group(1)
                            
                            elif data_type == "medication" and not medication:
                                medication = content
                                context.userdata.patient_medication = content
                            
                            elif data_type == "treatment":
                                treatments.append(content)
                        
                        context.userdata.patient_treatments = treatments
                        
                        # Erstelle Antwort
                        response_text = "Ich habe folgende Patientendaten gefunden:\n\n"
                        
                        if patient_info:
                            response_text += f"**Patientendaten:**\n{patient_info}\n\n"
                        
                        if medication:
                            response_text += f"**{medication}\n\n"
                        
                        if treatments:
                            response_text += "**Behandlungen:**\n"
                            for treatment in treatments[:3]:
                                response_text += f"{treatment}\n\n"
                        
                        return response_text.strip()
                    
                    else:
                        logger.info("❌ No matching documents found")
                        return f"Ich habe keine Patientendaten für '{query}' gefunden. Bitte überprüfen Sie die Eingabe."
                
                else:
                    logger.error(f"Qdrant error: {response.status_code}")
                    return "Die Datenbank ist momentan nicht erreichbar. Bitte versuchen Sie es später erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Es gab einen Fehler bei der Suche. Bitte versuchen Sie es erneut."

    @function_tool
    async def get_current_patient_details(self,
                                        context: RunContext[MedicalUserData],
                                        detail_type: str) -> str:
        """
        Gibt Details des aktuell geladenen Patienten zurück.
        
        Args:
            detail_type: "medication", "treatments", "allergies", "chronic_conditions", oder "all"
            
        Returns:
            Die angeforderten Details
        """
        logger.info(f"📋 Getting details: {detail_type} for patient: {context.userdata.active_patient}")
        
        if not context.userdata.active_patient and not context.userdata.patient_name:
            return "Es ist kein Patient geladen. Bitte suchen Sie zuerst nach einem Patienten."
        
        patient_id = context.userdata.active_patient or "Unbekannt"
        patient_name = context.userdata.patient_name or "Unbekannt"
        
        response_text = f"Details für {patient_name} ({patient_id}):\n\n"
        
        if detail_type == "medication" or detail_type == "all":
            if context.userdata.patient_medication:
                response_text += f"**{context.userdata.patient_medication}\n\n"
            else:
                response_text += "**Medikation:** Keine Daten vorhanden.\n\n"
        
        if detail_type == "treatments" or detail_type == "all":
            if context.userdata.patient_treatments:
                response_text += "**Letzte Behandlungen:**\n"
                for treatment in context.userdata.patient_treatments:
                    response_text += f"{treatment}\n\n"
            else:
                response_text += "**Behandlungen:** Keine Daten vorhanden.\n\n"
        
        if detail_type == "allergies" or detail_type == "all":
            if context.userdata.patient_allergies:
                response_text += f"**Allergien:** {context.userdata.patient_allergies}\n\n"
            else:
                response_text += "**Allergien:** Keine Daten vorhanden.\n\n"
        
        if detail_type == "chronic_conditions" or detail_type == "all":
            if context.userdata.patient_chronic_conditions:
                response_text += f"**Chronische Erkrankungen:** {context.userdata.patient_chronic_conditions}\n\n"
            else:
                response_text += "**Chronische Erkrankungen:** Keine Daten vorhanden.\n\n"
        
        return response_text.strip()


async def request_handler(ctx: JobContext):
    """Request handler"""
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

    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")

        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")

        # 3. Configure Ollama
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        qdrant_url = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")

        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2")

        # 4. Create session
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                rag_url=rag_url,
                qdrant_url=qdrant_url,
                greeting_sent=False,
                active_patient=None,
                patient_name=None,
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

        # 5. Create and start agent
        agent = MedicalAssistant()
        
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

        # 6. Initial greeting
        await asyncio.sleep(1.5)

        greeting_text = """Guten Tag und herzlich willkommen bei der Klinik St. Anna!
Ich bin Lisa, Ihre digitale medizinische Assistentin.

Bitte nennen Sie mir die Patienten-ID (z.B. P001) oder den Namen des Patienten."""

        await session.say(
            greeting_text,
            allow_interruptions=True,
            add_to_chat_ctx=True
        )

        logger.info(f"✅ [{session_id}] Agent ready!")

        # Wait for disconnect
        disconnect_event = asyncio.Event()

        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()

        ctx.room.on("disconnected", handle_disconnect)
        await disconnect_event.wait()

    except Exception as e:
        logger.error(f"❌ [{session_id}] Error: {e}", exc_info=True)
        raise

    finally:
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass

        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
