# basics/medical_agent/agent.py
import os
import logging
import httpx
import json
import asyncio
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe
from livekit.agents.voice import Agent
from livekit.agents import llm
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("medical-assistant")

class MedicalAgent(Agent):
    """Medical Agent with RAG integration"""
    
    def __init__(self):
        # System prompt
        system_prompt = """Du bist eine kompetente und erfahrene Arztsekretärin in einer medizinischen Praxis mit Zugriff auf die elektronische Patientenakte.

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

Bereite die Informationen so auf, dass der Arzt optimal auf das Patientengespräch vorbereitet ist."""
        
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
                voice="shimmer"
            ),
            vad=silero.VAD.load(),
            allow_interruptions=True
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "medical_nutrition")
        logger.info(f"MedicalAgent initialized with collection: {self.rag_collection}")
    
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
        
        # Create and start the agent
        agent = MedicalAgent()
        agent.start(ctx.room, ctx.local_participant)
        
        # Send initial greeting through the agent session
        await agent.session.say(
            "Guten Tag Herr Doktor! Ich bin Ihre digitale Praxisassistentin. "
            "Ich kann Ihnen sofort alle relevanten Patientendaten aus unserer elektronischen Akte zur Verfügung stellen. "
            "Welchen Patienten möchten Sie besprechen?",
            allow_interruptions=True
        )
        
        logger.info("Medical assistant ready with RAG support")
        
    except Exception as e:
        logger.error(f"Error in medical assistant: {e}", exc_info=True)
        raise

# Ensure the module can be imported
__all__ = ['entrypoint']
