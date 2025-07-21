# basics/medical_agent.py
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

logger = logging.getLogger("medical-assistant")

class MedicalAssistant(Agent):
    def __init__(self):
        # RAG Service URL aus Environment
        self.rag_url = "http://localhost:8000"
        
        super().__init__(
            instructions="""Du bist ein hilfreicher Assistent für medizinische Informationen mit Zugriff auf eine medizinische Wissensdatenbank.

WICHTIGER HINWEIS: Du stellst nur allgemeine Gesundheitsinformationen zur Verfügung.
Erinnere die Nutzer immer daran, dass:
- Dies KEIN Ersatz für professionellen medizinischen Rat ist
- Sie bei medizinischen Anliegen qualifizierte Gesundheitsfachkräfte konsultieren sollten
- Sie in Notfällen sofort den Notdienst anrufen sollten

DATENBANK-ZUGRIFF:
- Du hast Zugriff auf eine medizinische Wissensdatenbank
- Nutze diese für präzise, evidenzbasierte Informationen
- Zitiere relevante Quellen aus der Datenbank wenn möglich

Du kannst helfen bei:
- Allgemeinen Gesundheitsinformationen und Wellness-Tipps
- Erklärung gängiger medizinischer Begriffe in einfacher Sprache
- Grundlegenden Erste-Hilfe-Informationen
- Empfehlungen für einen gesunden Lebensstil
- Verständnis häufiger Symptome (mit Haftungsausschluss)
- Medikamenten-Erinnerungen und allgemeinen Informationen

Sei präzise, einfühlsam und klar in deinen Antworten.
Im Zweifelsfall solltest du immer zur Vorsicht raten und eine professionelle Beratung empfehlen.""",
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.7,
            ),
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
                        "collection": "medical_knowledge"  # Falls spezifische Collection existiert
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
        context = f"Nutzeranfrage: {message}\n\n"
        
        if knowledge:
            context += "Relevante Informationen aus der Wissensdatenbank:\n"
            for idx, result in enumerate(knowledge, 1):
                context += f"\n{idx}. {result['content']}\n"
                if result.get('metadata'):
                    context += f"   Quelle: {result['metadata'].get('source', 'Unbekannt')}\n"
        
        # Lasse das LLM mit dem erweiterten Kontext antworten
        return context

async def entrypoint(ctx: JobContext):
    """Main entry point for the medical assistant agent"""
    logger.info("Medical assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Create the agent
        agent = MedicalAssistant()
        
        # Test RAG connection
        test_results = await agent.search_knowledge("health check")
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
                voice="shimmer"
            ),
        )
        
        # Start the session
        await session.start(
            agent=agent, 
            room=ctx.room,
            room_input_options=RoomInputOptions()
        )
        
        # Generate initial greeting with medical disclaimer
        await session.generate_reply(
            instructions="""Greet the user professionally. Say: 
            'Hello! I'm your medical information assistant with access to a comprehensive medical knowledge database. 
            I can help you understand general health topics and medical terms. 
            Please remember that I provide general information only and cannot replace professional medical advice. 
            For any medical concerns, please consult with a qualified healthcare provider. 
            How can I help you today?'"""
        )
        
        logger.info("Medical assistant ready with RAG support")
        
    except Exception as e:
        logger.error(f"Error in medical assistant: {e}", exc_info=True)
        raise

__all__ = ['entrypoint']
