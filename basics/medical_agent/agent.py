# ========================================
# MEDICAL AGENT (basics/medical_agent/agent.py) - FIXED
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

logger = logging.getLogger("medical-assistant")

class MedicalAssistant:
    """Medical Assistant with RAG integration"""
    
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting medical agent entrypoint")
    
    # Initialize the medical assistant
    async with MedicalAssistant() as medical_assistant:
        
        # Create the agent (NOT inheriting from Agent)
        agent = Agent(
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
              - "+41 79 123 4567" → "plus 41... 79... 123... 45... 67"
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
                temperature=0.7
            ),
            tts=openai.TTS(model="tts-1", voice="shimmer"),
            vad=silero.VAD.load(
                min_silence_duration=0.8,    # Erhöht für Session-Kontinuität
                min_speech_duration=0.3,
                activation_threshold=0.5,    # NEU
                deactivation_threshold=0.3   # NEU
            ),
            interrupt_min_words=2,  # Mindestens 2 Wörter für Unterbrechung
        )
        
        # Check RAG service health first
        try:
            response = await medical_assistant.http_client.get(f"{medical_assistant.base_url}/health")
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
            participant=participant
        )
        
        # Custom message handler for RAG enhancement
        original_on_message = session._on_participant_message if hasattr(session, '_on_participant_message') else None
        
        async def enhanced_message_handler(message: ChatMessage):
            """Enhance messages with RAG before processing"""
            if message.content:
                query_text = str(message.content[0]) if isinstance(message.content, list) and len(message.content) > 0 else str(message.content)
                
                # Process patient IDs
                query_text = medical_assistant.process_patient_id(query_text)
                
                # Search RAG
                rag_results = await medical_assistant.search_knowledge(query_text)
                
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
        await session.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", 
                         allow_interruptions=True)
        
        logger.info("Medical agent session started successfully")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
