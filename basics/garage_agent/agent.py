# ========================================
# GARAGE AGENT (basics/garage_agent/agent.py)
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

logger = logging.getLogger("garage-assistant")

class GarageAgent(Agent):
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        
        super().__init__(
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
              - 1850 → "eintausendachthundertfünfzig Franken"""",
            stt=openai.STT(  # Wechsel zu Whisper für bessere Erkennung
                model="whisper-1",
                language="de"
            ),
            llm=openai.LLM(
                model="llama3.1:8b",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
                timeout=120.0,
                temperature=0.3
            ),
            tts=openai.TTS(model="tts-1", voice="onyx"),  # OpenAI TTS
            vad=silero.VAD.load(
                min_silence_duration=0.6,    # Höher für bessere Trennung
                min_speech_duration=0.3      # Länger für vollständige Wörter
            )
        )
        logger.info("Garage assistant starting with RAG support, Whisper STT and local Ollama LLM")

    async def on_enter(self):
        """Called when the agent enters the conversation"""
        logger.info("Garage assistant ready with RAG support")
        
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
            
            # Spezielle Behandlung für Fahrzeug-IDs
            if query_text:
                # Konvertiere "f null null X" zu "F00X"
                import re
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

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting garage agent entrypoint")
    
    # NOTE: ctx.connect() is already called in simple_multi_agent_fixed.py
    # Do NOT call it again here!
    
    # Create and start the agent session
    session = AgentSession()
    agent = GarageAgent()
    
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("Garage agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
