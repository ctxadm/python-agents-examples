import asyncio
import logging
import os
import httpx
import re
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RunContext,
    llm,
    stt,
    tts,
    vad,
)
from livekit.plugins import openai, silero
from typing import Optional
import time

load_dotenv()

# Logging
logger = logging.getLogger("medical-assistant")
logger.setLevel(logging.INFO)

# Medical Assistant with RAG
class MedicalAssistant:
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info("Medical Assistant initialized with RAG service")
        
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
                        logger.info(f"RAG search successful: {len(results)} results for query: {query}")
                        # Format results for LLM context
                        formatted_results = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted_results.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted_results)
                    else:
                        logger.info(f"No RAG results found for query: {query}")
                        return None
                else:
                    logger.error(f"RAG search failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return None

    def process_patient_id(self, query_text: str) -> str:
        """Process patient ID from speech to text"""
        # Konvertiere "p null null X" zu "P00X"
        pattern = r'p\s*null\s*null\s*(\w+)'
        match = re.search(pattern, query_text.lower())
        if match:
            number = match.group(1)
            # Konvertiere Wörter zu Zahlen wenn nötig
            number_map = {
                'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                'neun': '9', 'null': '0'
            }
            if number in number_map:
                number = number_map[number]
            
            corrected_id = f"P00{number}"
            query_text = f"Patienten-ID {corrected_id}"
            logger.info(f"Corrected patient ID from '{match.group(0)}' to '{corrected_id}'")
        
        return query_text

# Custom Agent class that handles RAG
class MedicalAgent(Agent):
    def __init__(self, assistant: MedicalAssistant, **kwargs):
        super().__init__(**kwargs)
        self.assistant = assistant
        self._processing_lock = asyncio.Lock()
        
    async def on_user_turn_completed(self, turn_ctx, user_message):
        """This should be called when user finishes speaking"""
        logger.info("=== on_user_turn_completed CALLED ===")
        
        if user_message and user_message.text:
            logger.info(f"Processing user query: {user_message.text}")
            
            # Process patient IDs
            processed_text = self.assistant.process_patient_id(user_message.text)
            
            # Search RAG
            rag_results = await self.assistant.search_knowledge(processed_text)
            
            if rag_results:
                # Update the user message with RAG results
                enhanced_text = f"{processed_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                user_message.text = enhanced_text
                logger.info(f"Enhanced query with RAG results")
            else:
                logger.warning(f"No RAG results found for: {processed_text}")

async def entrypoint(ctx: agents.JobContext):
    logger.info("=== Medical Agent Starting ===")
    
    # Initialize assistant
    medical_assistant = MedicalAssistant()
    
    # Check RAG service health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{medical_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to check RAG service health: {e}")
    
    # Create agent
    agent = MedicalAgent(
        assistant=medical_assistant,
        instructions="""Du bist ein Agent mit Zugriff auf die Patientendatenbank.
            
            ERSTE ANTWORT (IMMER):
            "Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?"
            
            EXTREM WICHTIG - So liest du die Datenbank:
            1. Die Daten stehen IMMER unter "Relevante Informationen aus der Datenbank"
            2. Wenn dort "Patient: Emma Fischer" steht, dann EXISTIERT Emma Fischer zu 100%!
            3. Wenn dort "patient_id": "P005" steht, dann ist P005 eine GÜLTIGE ID!
            4. Du MUSST die Daten die du bekommst als WAHR akzeptieren!
            
            Patienten-IDs verstehen:
            - "p null null fünf" bedeutet wahrscheinlich "P005" 
            - "p null null drei" bedeutet wahrscheinlich "P003"
            - Buchstabe "P" gefolgt von Zahlen ist eine Patienten-ID
            
            REGEL: Wenn ein Patient in den "Relevanten Informationen" steht, dann:
            - Sage NIEMALS "nicht gefunden" oder "keine Daten"
            - Der Patient IST in der Datenbank
            - Gib die Informationen aus den relevanten Daten wieder
            
            Datenschutz:
            - Frage nach dem Namen des Patienten oder der Patienten-ID
            - Gib NUR Informationen zum bestätigten Patienten heraus
            - Antworte professionell und präzise
            
            Bei unklaren Eingaben:
            - "p null null X" → interpretiere als "P00X" 
            - Frage nach: "Meinen Sie die Patienten-ID P00X, Herr Doktor?"
            
            Nenne dich selbst nur "Agent" und duze niemals.
            
            Währungsangaben - WICHTIG für korrekte Aussprache:
            - Schreibe Beträge IMMER als "X Franken" aus
            - NIEMALS "15.50" sondern "15 Franken 50"
            - NIEMALS "CHF" oder "€" verwenden
            - Dosierungen klar aussprechen:
              - "10mg" → "zehn Milligramm"
              - "200µg" → "zweihundert Mikrogramm"
              - "5ml" → "fünf Milliliter"
            - Telefonnummern mit Pausen:
              - "+41 79 123 4567" → "plus 41... 79... 123... 45... 67" """,
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
    
    # Create session
    session = AgentSession(
        agent=agent,
    )
    
    # Connect to room
    await ctx.connect()
    
    # Start session
    await session.start(room=ctx.room)
    
    # Initial greeting
    await session.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", allow_interruptions=True)
    logger.info("Medical agent started successfully")
    
    # Log for debugging
    @ctx.room.on("track_published")
    def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"Track published: {publication.sid} from {participant.identity}")
    
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"Track subscribed: {track.sid} from {participant.identity}")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
