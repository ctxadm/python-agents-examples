# ========================================
# MEDICAL AGENT (basics/medical_agent/agent.py)
# ========================================
import os
import logging
import httpx
import json
import asyncio
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import ChatContext, ChatMessage, ChatContent
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

logger = logging.getLogger("medical-assistant")

class MedicalAgent(Agent):
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        
        super().__init__(
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
            stt=openai.STT(  # Wechsel zu Whisper für bessere Erkennung
                model="whisper-1",
                language="de"
            ),
            llm=openai.LLM(
                model="llama3.1:8b",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
                timeout=120.0,
                temperature=0.7
            ),
            tts=openai.TTS(model="tts-1", voice="shimmer"),  # OpenAI TTS
            vad=silero.VAD.load(
                min_silence_duration=0.6,    # Noch höher für bessere Trennung
                min_speech_duration=0.3      # Länger für vollständige Wörter
            )
        )
        logger.info("Medical assistant starting with RAG support, Whisper STT and local Ollama LLM")

    async def on_enter(self):
        """Called when the agent enters the conversation"""
        logger.info("Medical assistant ready with RAG support")
        
        # Check RAG service health
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to check RAG service health: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - here we can enhance with RAG"""
        user_query = new_message.content
        
        if user_query and isinstance(user_query, list) and len(user_query) > 0:
            # Extract text content from the message
            query_text = str(user_query[0]) if hasattr(user_query[0], '__str__') else ""
            
            # Spezielle Behandlung für Patienten-IDs
            if query_text:
                # Konvertiere "p null null X" zu "P00X"
                import re
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
                
                # Search RAG for relevant information
                rag_results = await self.search_knowledge(query_text)
                
                if rag_results:
                    # Create enhanced content with RAG results
                    enhanced_content = f"{query_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                    
                    # Update the message content directly
                    new_message.content = [enhanced_content]
                    
                    logger.info(f"Enhanced query with RAG results for: {query_text}")

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

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting medical agent entrypoint")
    
    # NOTE: ctx.connect() is already called in simple_multi_agent_fixed.py
    # Do NOT call it again here!
    
    # Create and start the agent session
    session = AgentSession()
    agent = MedicalAgent()
    
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("Medical agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
