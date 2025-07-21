# basics/garage_agent.py
import logging
import httpx
import json
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import (
    JobContext, 
    Agent, 
    AgentSession, 
    RoomInputOptions,
    AutoSubscribe
)
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("garage-assistant")

class GarageAssistant(Agent):
    def __init__(self):
        # RAG Service URL aus Environment
        self.rag_url = "http://localhost:8000"
        
        super().__init__(
            instructions="""Du bist ein hilfreicher Kundenservice-Assistent für eine professionelle Autowerkstatt mit Zugriff auf eine technische Fahrzeugdatenbank.
            
            DATENBANK-ZUGRIFF:
            - Du hast Zugriff auf technische Fahrzeuginformationen
            - Nutze die Datenbank für spezifische Reparaturanleitungen und Wartungspläne
            - Greife auf Herstellerspezifikationen und Service-Bulletins zu
            
            Du kannst Kunden helfen bei:
            - Terminvereinbarungen für Service und Wartung
            - Erklärung von Autoreparaturen und technischen Problemen in einfachen Worten
            - Kostenvoranschlägen für gängige Dienstleistungen (basierend auf Datenbank)
            - Fragen zu Wartungsintervallen und Wartungsplänen (aus der Datenbank)
            - Überprüfung von Service-Historie und Garantieinformationen
            - Empfehlungen für vorbeugende Wartung basierend auf Fahrzeugdaten
            
            Sei professionell, sachkundig und ehrlich bezüglich Reparaturbedarf.
            Betone immer Sicherheit und ordnungsgemäße Wartung.
            Nutze die Datenbank für präzise technische Informationen.""",
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.7,
            ),
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
                        "collection": "automotive_knowledge"  # Falls spezifische Collection existiert
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"RAG search successful: {len(data['results'])} results")
                    return data['results']
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return None
    
    async def on_user_message(self, message: str) -> str:
        """Verarbeitet Nutzernachrichten mit RAG-Unterstützung"""
        # Suche relevante Informationen in der Datenbank
        knowledge = await self.search_knowledge(message)
        
        # Erstelle erweiterten Kontext für das LLM
        context = f"Kundenanfrage: {message}\n\n"
        
        if knowledge:
            context += "Relevante technische Informationen aus der Datenbank:\n"
            for idx, result in enumerate(knowledge, 1):
                context += f"\n{idx}. {result['content']}\n"
                if result.get('metadata'):
                    context += f"   Quelle: {result['metadata'].get('source', 'Unbekannt')}\n"
                    if result['metadata'].get('vehicle_make'):
                        context += f"   Fahrzeug: {result['metadata']['vehicle_make']} {result['metadata'].get('vehicle_model', '')}\n"
        
        # Lasse das LLM mit dem erweiterten Kontext antworten
        return context

async def entrypoint(ctx: JobContext):
    """Main entry point for the garage assistant agent"""
    logger.info("Garage assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Create the agent
        agent = GarageAssistant()
        
        # Test RAG connection
        test_results = await agent.search_knowledge("service maintenance")
        if test_results:
            logger.info("RAG service is accessible")
        else:
            logger.warning("RAG service might be unavailable")
        
        # Create the session with all components
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
        )
        
        # Start the session
        await session.start(
            agent=agent, 
            room=ctx.room,
            room_input_options=RoomInputOptions()
        )
        
        # Generate initial greeting
        await session.generate_reply(
            instructions="Greet the customer warmly. Say: 'Hello! Welcome to our automotive service center. I have access to our comprehensive vehicle database and can help with technical information, service scheduling, and maintenance recommendations. How can I assist you with your vehicle today?'"
        )
        
        logger.info("Garage assistant ready with RAG support")
        
    except Exception as e:
        logger.error(f"Error in garage assistant: {e}", exc_info=True)
        raise

__all__ = ['entrypoint']
