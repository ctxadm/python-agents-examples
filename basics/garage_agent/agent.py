# ========================================
# GARAGE AGENT (basics/garage_agent/agent.py) - FIXED
# ========================================
import os
import logging
import httpx
import json
import asyncio
import re
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import ChatContext, ChatMessage, ChatContent
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

logger = logging.getLogger("garage-assistant")

class GarageAssistant:
    """Garage Assistant with RAG integration"""
    
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        self.http_client = None
        
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()

    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            response = await self.http_client.post(
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting garage agent entrypoint")
    
    # Initialize the garage assistant
    async with GarageAssistant() as garage_assistant:
        
        # Create the agent (NOT inheriting from Agent)
        agent = Agent(
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
              - 1850 → "eintausendachthundertfünfzig Franken"
            """,
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            llm=openai.LLM(
                model="llama3.1:8b",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.3
            ),
            tts=openai.TTS(model="tts-1", voice="onyx"),
            vad=silero.VAD.load(
                min_silence_duration=0.8,    # Erhöht für Session-Kontinuität
                min_speech_duration=0.3,
                activation_threshold=0.5     # Existiert in deiner Version!
            ),
            allow_interruptions=True,  # Hier ist es richtig!
            min_consecutive_speech_delay=0.0
        )
        
        # Check RAG service health first
        try:
            response = await garage_assistant.http_client.get(f"{garage_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to check RAG service health: {e}")
        
        # WICHTIG: Warte auf den Participant
        logger.info("Waiting for participant...")
        participant = await ctx.wait_for_participant()
        logger.info(f"Participant joined: {participant.identity}")
        
        # Create session with participant
        session = AgentSession(
            agent=agent,
            participant=participant,
            min_interruption_words=2  # Hier ist es richtig für AgentSession!
        )
        
        # Custom message handler for RAG enhancement
        original_on_message = session._on_participant_message if hasattr(session, '_on_participant_message') else None
        
        async def enhanced_message_handler(message: ChatMessage):
            """Enhance messages with RAG before processing"""
            if message.content:
                query_text = str(message.content[0]) if isinstance(message.content, list) and len(message.content) > 0 else str(message.content)
                
                # Process vehicle IDs
                query_text = garage_assistant.process_vehicle_id(query_text)
                
                # Search RAG
                rag_results = await garage_assistant.search_knowledge(query_text)
                
                if rag_results:
                    enhanced_content = f"{query_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                    message.content = [enhanced_content]
                    logger.info(f"Enhanced query with RAG results")
            
            # Call original handler if exists
            if original_on_message:
                await original_on_message(message)
        
        # Override message handler if possible
        if hasattr(session, '_on_participant_message'):
            session._on_participant_message = enhanced_message_handler
        
        # Start the session
        logger.info("Starting agent session...")
        await session.start()
        
        # Initial greeting
        await session.say("Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann.", 
                         allow_interruptions=True)
        
        logger.info("Garage agent session started successfully")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
