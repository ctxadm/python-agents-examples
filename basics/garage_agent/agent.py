# basics/garage_agent/agent.py
import os
import logging
import httpx
import json
import asyncio
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe
from livekit.agents.voice import Agent, AgentSession
from livekit.agents import llm
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("garage-assistant")

class GarageAgent(Agent):
    """Garage Agent with RAG integration"""
    
    def __init__(self):
        # System prompt
        system_prompt = """Du bist ein freundlicher und kompetenter Kundenservice-Mitarbeiter einer Autowerkstatt mit Zugriff auf die komplette Fahrzeugdatenbank.

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

Hilf den Kunden, ihr Fahrzeug optimal zu warten und sicher zu fahren."""
        
        # Initialize the parent Agent
        super().__init__(
            instructions=system_prompt,
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"  # Männliche, freundliche Stimme
            ),
            vad=silero.VAD.load(),
            allow_interruptions=True
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "automotive_docs")
        logger.info(f"GarageAgent initialized with collection: {self.rag_collection}")
    
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
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Test RAG connection
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        try:
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{rag_url}/health")
                if health_response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service unhealthy: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Could not reach RAG service: {e}")
        
        # Create agent and session
        agent = GarageAgent()
        session = AgentSession()
        
        # Start the session with the agent
        await session.start(
            agent=agent,
            room=ctx.room
        )
        
        # Send initial greeting through the session
        await session.say(
            "Guten Tag! Willkommen beim Kundenservice Ihrer Autowerkstatt. "
            "Ich kann Ihnen alle Informationen zu Ihrem Fahrzeug geben - von der Service-Historie "
            "bis zu anstehenden Wartungen. Wie kann ich Ihnen heute helfen?",
            allow_interruptions=True
        )
        
        logger.info("Garage assistant ready with RAG support")
        
    except Exception as e:
        logger.error(f"Error in garage assistant: {e}", exc_info=True)
        raise

# Ensure the module can be imported
__all__ = ['entrypoint']
