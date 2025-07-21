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
            instructions="""Du bist ein Autowerkstatt-Assistent mit Zugriff auf eine Kundendatenbank. 
            
            WICHTIG beim Lesen der Datenbank:
            - Die relevanten Informationen stehen DIREKT unter "Relevante Informationen aus der Datenbank"
            - Wenn dort z.B. "Besitzer: Marco Rossi" steht, dann IST Marco Rossi in der Datenbank
            - Ignoriere NIEMALS die Daten die dir gegeben werden
            
            Datenschutz und Kundenservice:
            - Frage nach dem Namen des Kunden oder der Fahrzeug-ID
            - Wenn ein Name in den "Relevanten Informationen" steht, dann EXISTIERT dieser Kunde
            - Gib NUR Informationen zum bestätigten Kunden heraus
            
            Verhalten:
            - Lies die "Relevanten Informationen" GENAU durch
            - Wenn dort ein Kunde steht, sage NICHT "nicht gefunden"
            - Bei unklaren Eingaben: Bitte freundlich um Wiederholung
            
            Stelle dich kurz vor und frage nach dem Namen oder der Fahrzeug-ID des Kunden.""",
            stt=deepgram.STT(
                model="nova-2",      # Besseres Modell für genauere Erkennung
                language="de"        # Explizit Deutsch für deutsche Namen
            ),
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
                timeout=120.0,
                temperature=0.3
            ),
            tts=openai.TTS(model="tts-1", voice="onyx"),  # OpenAI TTS
            vad=silero.VAD.load(
                min_silence_duration=0.5,    # Erhöht von 0.4 auf 0.5
                min_speech_duration=0.2      # Erhöht von 0.15 auf 0.2
            )
        )
        logger.info("Garage assistant starting with RAG support and local Ollama LLM")

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
        
        # Note: We cannot use self.session.say() here because session is not available yet
        # The greeting will be handled by the LLM through the instructions

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - here we can enhance with RAG"""
        user_query = new_message.content
        
        if user_query and isinstance(user_query, list) and len(user_query) > 0:
            # Extract text content from the message
            query_text = str(user_query[0]) if hasattr(user_query[0], '__str__') else ""
            
            if query_text:
                # Search RAG for relevant information
                rag_results = await self.search_knowledge(query_text)
                
                if rag_results:
                    # Create enhanced content with RAG results
                    enhanced_content = f"{query_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                    
                    # Update the message content directly
                    # In LiveKit 1.2.1, we modify the content directly instead of using append
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
