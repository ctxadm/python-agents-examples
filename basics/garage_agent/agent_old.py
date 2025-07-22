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
logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.INFO)

# Garage Assistant with RAG
class GarageAssistant:
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info("Garage Assistant initialized with RAG service")
        
    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"
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

    def process_vehicle_id(self, query_text: str) -> str:
        """Process vehicle ID from speech to text"""
        # Konvertiere "f null null X" zu "F00X"
        pattern = r'f\s*null\s*null\s*(\w+)'
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
            
            corrected_id = f"F00{number}"
            query_text = f"Fahrzeug-ID {corrected_id}"
            logger.info(f"Corrected vehicle ID from '{match.group(0)}' to '{corrected_id}'")
        
        return query_text

# Custom Agent class that handles RAG
class GarageAgent(Agent):
    def __init__(self, assistant: GarageAssistant, **kwargs):
        super().__init__(**kwargs)
        self.assistant = assistant
        self._processing_lock = asyncio.Lock()
        
    async def on_user_turn_completed(self, turn_ctx, user_message):
        """This should be called when user finishes speaking"""
        logger.info("=== on_user_turn_completed CALLED ===")
        
        if user_message and user_message.text:
            logger.info(f"Processing user query: {user_message.text}")
            
            # Process vehicle IDs
            processed_text = self.assistant.process_vehicle_id(user_message.text)
            
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
    logger.info("=== Garage Agent Starting ===")
    
    # Initialize assistant
    garage_assistant = GarageAssistant()
    
    # Check RAG service health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{garage_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to check RAG service health: {e}")
    
    # Create agent
    agent = GarageAgent(
        assistant=garage_assistant,
        instructions="""Du bist der Garage Agent der Firma AutoService Müller.
            
            ERSTE ANTWORT (IMMER):
            "Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann."
            
            EXTREM WICHTIG - So liest du die Datenbank:
            1. Die Daten stehen IMMER unter "Relevante Informationen aus der Datenbank"
            2. Wenn dort "Besitzer: Marco Rossi" steht, dann IST Marco Rossi zu 100% Kunde!
            3. Wenn dort "fahrzeug_id": "F003" steht, dann ist F003 eine GÜLTIGE ID!
            4. Du MUSST die Daten die du bekommst als WAHR akzeptieren!
            
            Fahrzeug-IDs verstehen:
            - "f null null drei" bedeutet wahrscheinlich "F003" 
            - "f null null fünf" bedeutet wahrscheinlich "F005"
            - Buchstabe "F" gefolgt von Zahlen ist eine Fahrzeug-ID
            
            REGEL: Wenn ein Kunde in den "Relevanten Informationen" steht, dann:
            - Sage NIEMALS "nicht gefunden" oder "keine Daten"
            - Der Kunde IST in der Datenbank
            - Gib NUR die relevanten Fahrzeugdaten aus
            
            Antwortverhalten - KURZ UND PRÄZISE:
            - Nenne nur die angefragten Informationen
            - Keine langen Erklärungen oder Geschichten
            - Fokus auf: Fahrzeugdaten, Probleme, anstehende Arbeiten
            - Beispiel: "Ihr Audi A4, Kennzeichen LU 234567, hat folgende Probleme: ..."
            
            Datenschutz:
            - Gib NUR Informationen zum bestätigten Kunden heraus
            - Bei Unklarheiten nachfragen
            
            Bei unklaren Eingaben:
            - "f null null X" → interpretiere als "F00X" 
            - Frage nach: "Meinen Sie die Fahrzeug-ID F00X?"
            
            KEINE unnötigen Floskeln, KEINE langen Sätze, NUR relevante Informationen!
            
            Währungsangaben - WICHTIG für korrekte Aussprache:
            - Schreibe Beträge IMMER als "X Franken" aus
            - NIEMALS "420.00" sondern "420 Franken"
            - NIEMALS "CHF" oder "€" verwenden
            - Bei Kommabeträgen: "180 Franken 50" statt "180.50"
            - Große Beträge ausschreiben für bessere Aussprache:
              - 420 → "vierhundertzwanzig Franken"
              - 1850 → "eintausendachthundertfünfzig Franken" """,
        llm=openai.LLM(
            model="llama3.1:8b",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            temperature=0.3,
        ),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(model="tts-1", voice="onyx"),
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
    await session.say("Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann.", allow_interruptions=True)
    logger.info("Garage agent started successfully")
    
    # Log for debugging
    @ctx.room.on("track_published")
    def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"Track published: {publication.sid} from {participant.identity}")
    
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"Track subscribed: {track.sid} from {participant.identity}")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
