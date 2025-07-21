# basics/garage_agent/agent.py
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

logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

class GarageAgent(Agent):
    """Garage Agent with RAG integration"""
    
    def __init__(self) -> None:
        # Initialize parent with instructions
        super().__init__(
            instructions="""Du bist ein freundlicher und kompetenter Kundenservice-Mitarbeiter einer Autowerkstatt mit Zugriff auf die komplette Fahrzeugdatenbank.

DEINE ROLLE:
- Du hilfst Kunden bei allen Fragen zu ihren Fahrzeugen
- Du gibst Auskunft über Service-Historie, anstehende Wartungen und aktuelle Probleme
- Du bist hilfsbereit, verständnisvoll und lösungsorientiert

DATENBANK-ZUGRIFF:
- Du hast Zugriff auf alle Fahrzeugdaten unserer Kunden
- Bei Anfragen zu spezifischen Fahrzeugen (z.B. "VW Golf mit Kennzeichen ZH 123456") durchsuchst du die Datenbank
- Du kannst folgende Informationen bereitstellen:
  * Fahrzeugdaten (Marke, Modell, Baujahr, Kilometerstand)
  * Service-Historie mit durchgeführten Arbeiten
  * Aktuelle Probleme und deren Status
  * Anstehende Wartungen und Reparaturen
  * Kostenvoranschläge
  * Garantiestatus

KUNDENSERVICE:
- Erkläre technische Sachverhalte verständlich
- Gib transparente Auskunft über Kosten
- Empfehle notwendige Wartungen basierend auf Herstellervorgaben
- Weise auf dringende Reparaturen hin (besonders sicherheitsrelevante)
- Biete Terminvereinbarungen an
- Zeige Verständnis für die Sorgen der Kunden

KOMMUNIKATIONSSTIL:
- Freundlich und professionell
- Vermeide zu viel Fachjargon
- Sei ehrlich über notwendige Reparaturen
- Betone Sicherheit und Zuverlässigkeit

Hilf den Kunden, ihr Fahrzeug optimal zu warten und sicher zu fahren.""",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            tts=openai.TTS(model="tts-1", voice="nova"),  # Männliche, freundliche Stimme
            vad=silero.VAD.load()
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "automotive_docs")
        logger.info(f"GarageAgent initialized with collection: {self.rag_collection}")
    
    async def on_enter(self):
        """Called when the agent session starts"""
        logger.info("Garage agent entered the session")
        
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
            "Guten Tag! Willkommen beim Kundenservice Ihrer Autowerkstatt. "
            "Ich kann Ihnen alle Informationen zu Ihrem Fahrzeug geben - von der Service-Historie "
            "bis zu anstehenden Wartungen. Wie kann ich Ihnen heute helfen?"
        )
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - here we can enhance with RAG"""
        user_query = new_message.content[0] if new_message.content else ""
        
        if user_query:
            # Search RAG for relevant information
            rag_results = await self.search_knowledge(str(user_query))
            
            if rag_results:
                # Add RAG context to the conversation
                rag_context = "\n\nRelevante Informationen aus der Fahrzeugdatenbank:\n"
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
        """Sucht in der Fahrzeug-Wissensdatenbank"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": query,
                        "agent_type": "garage",
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
    """Main entry point for the garage assistant agent"""
    logger.info("Garage assistant starting with RAG support")
    
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create and start session
    session = AgentSession()
    await session.start(
        agent=GarageAgent(),
        room=ctx.room
    )
    
    logger.info("Garage assistant ready with RAG support")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

# Ensure the module can be imported
__all__ = ['entrypoint']
