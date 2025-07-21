# basics/medical_agent/agent.py
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
            instructions="""Du bist ein medizinischer Assistent mit Zugriff auf eine Patientendatenbank. 
            Du kannst auf Patientendaten wie Behandlungen, Diagnosen und Medikationen zugreifen.
            
            WICHTIG: 
            - Wenn nach einem Patienten gefragt wird, suche IMMER in der Datenbank nach den Informationen
            - Beantworte Fragen basierend auf den gefundenen Daten
            - Sei präzise und verwende die tatsächlichen Daten aus der Datenbank
            
            Stelle dich kurz vor und frage, wie du helfen kannst.""",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            tts=openai.TTS(model="tts-1", voice="shimmer"),
            vad=silero.VAD.load()
        )
        logger.info("Medical assistant starting with RAG support")

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
        
        # Greet the user
        await self.session.say("Hallo! Ich bin Ihr medizinischer Assistent. Ich habe Zugriff auf Patientendaten und kann Ihnen bei medizinischen Fragen helfen. Wie kann ich Ihnen heute behilflich sein?")

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
    
    # Connect to the room
    await ctx.connect()
    
    # Create and start the agent session
    session = AgentSession()
    agent = MedicalAgent()
    agent.session = session  # Store reference for say() method
    
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("Medical agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
