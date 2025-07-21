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
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import function_tool, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("medical-assistant")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

class MedicalAgent(Agent):
    """Medical Agent with RAG integration"""
    
    def __init__(self) -> None:
        # Initialize parent with instructions
        super().__init__(
            instructions="""Du bist eine kompetente und erfahrene Arztsekretärin in einer medizinischen Praxis mit Zugriff auf die elektronische Patientenakte.

DEINE ROLLE:
- Du unterstützt Ärzte bei der Vorbereitung von Patientengesprächen
- Du gibst schnell und präzise Auskunft über Patientendaten aus der Datenbank
- Du bist professionell, diskret und effizient

DATENBANK-ZUGRIFF:
- Du hast vollen Zugriff auf die elektronische Patientenakte mit allen relevanten Patientendaten
- Bei Anfragen zu spezifischen Patienten (z.B. "Emma Fischer") durchsuchst du sofort die Datenbank
- Du gibst alle relevanten Informationen strukturiert wieder:
  * Stammdaten (Name, Geburtsdatum, Blutgruppe)
  * Bekannte Allergien und Unverträglichkeiten
  * Chronische Erkrankungen
  * Aktuelle Medikation mit Dosierung
  * Letzte Behandlungen und Befunde
  * Notfallkontakte

ARBEITSWEISE:
- Antworte kurz, präzise und strukturiert
- Hebe wichtige Informationen hervor (z.B. Allergien, Wechselwirkungen)
- Weise auf auffällige oder kritische Befunde hin
- Erinnere an anstehende Kontrollen oder Folgebehandlungen

Bereite die Informationen so auf, dass der Arzt optimal auf das Patientengespräch vorbereitet ist.""",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            tts=openai.TTS(model="tts-1", voice="shimmer"),
            vad=silero.VAD.load()
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "medical_nutrition")
        logger.info(f"MedicalAgent initialized with collection: {self.rag_collection}")
    
    async def on_enter(self):
        """Called when the agent session starts"""
        logger.info("Medical agent entered the session")
        
        # Test RAG connection
        try:
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{self.rag_url}/health")
                if health_response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service unhealthy: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Could not reach RAG service: {e}")
        
        # Initial greeting
        await self.session.say(
            "Guten Tag Herr Doktor! Ich bin Ihre digitale Praxisassistentin. "
            "Ich kann Ihnen sofort alle relevanten Patientendaten aus unserer elektronischen Akte zur Verfügung stellen. "
            "Welchen Patienten möchten Sie besprechen?"
        )
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - here we can enhance with RAG"""
        user_query = new_message.content[0] if new_message.content else ""
        
        if user_query:
            # Search RAG for relevant information
            rag_results = await self.search_knowledge(str(user_query))
            
            if rag_results:
                # Add RAG context to the conversation
                rag_context = "\n\nRelevante Informationen aus der Patientendatenbank:\n"
                for idx, result in enumerate(rag_results, 1):
                    rag_context += f"\n{idx}. {result['content']}"
                    if result.get('metadata'):
                        rag_context += f"\n   Details: {json.dumps(result['metadata'], ensure_ascii=False)}"
                
                # Add system message with RAG context
                turn_ctx.append(
                    role="system",
                    text=f"Nutze diese Datenbankinformationen für deine Antwort:{rag_context}"
                )
    
    async def search_knowledge(self, query: str) -> Optional[List[Dict]]:
        """Sucht in der medizinischen Wissensdatenbank"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": query,
                        "agent_type": "medical",
                        "top_k": 3,
                        "collection": self.rag_collection
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"RAG search successful: {len(data['results'])} results for query: {query}")
                    return data['results']
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return None

async def entrypoint(ctx: JobContext):
    """Main entry point for the medical assistant agent"""
    logger.info("Medical assistant starting with RAG support")
    
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create and start session
    session = AgentSession()
    await session.start(
        agent=MedicalAgent(),
        room=ctx.room
    )
    
    logger.info("Medical assistant ready with RAG support")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

# Ensure the module can be imported
__all__ = ['entrypoint']
