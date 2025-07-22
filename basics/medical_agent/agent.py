import asyncio
import logging
import os
import httpx
import re
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero
from typing import Optional
import json

load_dotenv()

# Logging
logger = logging.getLogger("medical-assistant")
logger.setLevel(logging.INFO)

# Global variables for workaround
medical_assistant = None
current_session = None
monitor_task = None

# Medical Assistant with RAG
class MedicalAssistant:
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info("Medical Assistant initialized with RAG service")
        self.processed_count = 0
        
    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "agent_type": "medical",
                        "top_k": 3,
                        "collection": "medical_nutrition"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"RAG search successful: {len(results)} results")
                        formatted_results = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted_results.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted_results)
                else:
                    logger.error(f"RAG search failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return None

    def process_patient_id(self, query_text: str) -> str:
        """Process patient ID from speech to text"""
        pattern = r'p\s*null\s*null\s*(\w+)'
        match = re.search(pattern, query_text.lower())
        if match:
            number = match.group(1)
            number_map = {
                'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                'neun': '9', 'null': '0'
            }
            if number in number_map:
                number = number_map[number]
            
            corrected_id = f"P00{number}"
            logger.info(f"Corrected patient ID to '{corrected_id}'")
            return f"Patienten-ID {corrected_id}"
        
        return query_text

# Workaround: Listen to room events for transcriptions
async def monitor_transcriptions(ctx: JobContext, session: AgentSession):
    """Monitor room for transcriptions and inject RAG results"""
    global medical_assistant
    
    processed_transcripts = set()
    
    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket, participant: rtc.RemoteParticipant):
        """Handle transcription data"""
        try:
            if data.topic == "transcription":
                transcript_data = json.loads(data.data.decode())
                transcript_text = transcript_data.get("text", "")
                transcript_id = transcript_data.get("id", "")
                
                if transcript_id not in processed_transcripts and transcript_text:
                    processed_transcripts.add(transcript_id)
                    logger.info(f"=== WORKAROUND: Detected transcript: {transcript_text}")
                    
                    # Schedule RAG processing
                    asyncio.create_task(process_with_rag(transcript_text, session))
                    
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    # Alternative: Monitor participant speaking events
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
    
    logger.info("Transcription monitor started")

async def process_with_rag(text: str, session: AgentSession):
    """Process text with RAG and respond"""
    global medical_assistant
    
    try:
        # Skip if it's likely an agent response
        if any(phrase in text.lower() for phrase in ["guten tag", "herr doktor", "agent"]):
            return
            
        logger.info(f"Processing with RAG: {text}")
        
        # Process patient IDs
        processed_text = medical_assistant.process_patient_id(text)
        
        # Search RAG
        rag_results = await medical_assistant.search_knowledge(processed_text)
        
        if rag_results:
            # Create response with RAG context
            response = f"""Basierend auf Ihrer Anfrage zu '{processed_text}' habe ich folgende Informationen gefunden:

{rag_results}

Diese Daten stammen aus unserer Patientendatenbank."""
            
            # Send response
            await session.say(response, allow_interruptions=True)
            logger.info("RAG-enhanced response sent")
        else:
            logger.info("No RAG results found, letting normal flow handle it")
            
    except Exception as e:
        logger.error(f"Error in process_with_rag: {e}")

async def entrypoint(ctx: JobContext):
    global medical_assistant, current_session
    
    logger.info("=== Medical Agent Starting (1.2.1 Fixed) ===")
    
    # Initialize assistant
    medical_assistant = MedicalAssistant()
    
    # Check RAG service health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{medical_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed")
    except Exception as e:
        logger.error(f"Failed to check RAG service health: {e}")
    
    # Connect to room first
    await ctx.connect()
    logger.info("Connected to room")
    
    # Create session with 1.2.1 API
    session = AgentSession(
        instructions="""Du bist ein Agent mit Zugriff auf die Patientendatenbank.
            
ERSTE ANTWORT: "Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?"

WICHTIG - Datenbank:
- Wenn Informationen aus der Datenbank kommen, nutze sie IMMER
- Sage NIE "nicht gefunden" wenn Daten vorhanden sind
- Patienten-IDs: "p null null fünf" = "P005"

Währungen: "15 Franken 50" statt "15.50" """,
        llm=openai.LLM(
            model="llama3.1:8b",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            temperature=0.7,
        ),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(model="tts-1", voice="shimmer"),
        vad=silero.VAD.load(
            min_silence_duration=0.8,
            min_speech_duration=0.3,
            activation_threshold=0.5
        ),
    )
    
    current_session = session
    
    # Start the transcription monitor
    await monitor_transcriptions(ctx, session)
    
    # Start session  
    await session.start(ctx.room)
    logger.info("Session started")
    
    # Initial greeting
    await session.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", allow_interruptions=True)
    
    # Alternative workaround: Periodic check for new messages
    async def periodic_check():
        last_check = ""
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                # Try to access session state if available
                if hasattr(session, '_chat_ctx') or hasattr(session, 'chat_ctx'):
                    chat_ctx = getattr(session, '_chat_ctx', None) or getattr(session, 'chat_ctx', None)
                    if chat_ctx and hasattr(chat_ctx, 'messages'):
                        messages = chat_ctx.messages
                        if messages:
                            last_msg = messages[-1]
                            msg_text = getattr(last_msg, 'content', '') or getattr(last_msg, 'text', '')
                            if msg_text and msg_text != last_check:
                                last_check = msg_text
                                if last_msg.role == "user":
                                    logger.info(f"=== Periodic check found: {msg_text}")
                                    await process_with_rag(msg_text, session)
                                    
            except Exception as e:
                logger.debug(f"Periodic check error (expected): {e}")
                
    # Start periodic checker
    check_task = asyncio.create_task(periodic_check())
    logger.info("Started periodic message checker")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
