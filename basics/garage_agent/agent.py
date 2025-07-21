# basics/garage_agent/agent.py
import os
import logging
import httpx
import json
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import (
    JobContext, 
    AutoSubscribe,
    WorkerOptions,
    cli
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    LLM,
    LLMOptions
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("garage-assistant")

class GarageAssistantWithRAG:
    """Garage Assistant with RAG integration"""
    
    def __init__(self, rag_url: str = None):
        self.rag_url = rag_url or os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "automotive_docs")
        logger.info(f"GarageAssistant initialized with collection: {self.rag_collection}")
        
        # System prompt
        self.system_prompt = """Du bist ein freundlicher und kompetenter Kundenservice-Mitarbeiter einer Autowerkstatt mit Zugriff auf die komplette Fahrzeugdatenbank.

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
    
    def create_enhanced_prompt(self, user_message: str, rag_results: List[Dict]) -> str:
        """Erstellt einen erweiterten Prompt mit RAG-Ergebnissen"""
        if not rag_results:
            return user_message
        
        enhanced = f"Kundenanfrage: {user_message}\n\n"
        enhanced += "Relevante Informationen aus der Fahrzeugdatenbank:\n\n"
        
        for idx, result in enumerate(rag_results, 1):
            enhanced += f"{idx}. {result['content']}\n"
            if result.get('metadata'):
                enhanced += f"   Details: {json.dumps(result['metadata'], ensure_ascii=False)}\n"
            enhanced += "\n"
        
        enhanced += "\nBitte beantworte die Anfrage basierend auf diesen Datenbankinformationen. Erkläre technische Details verständlich."
        return enhanced

async def entrypoint(ctx: JobContext):
    """Main entry point for the garage assistant agent"""
    logger.info("Garage assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Test RAG connection
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        assistant = GarageAssistantWithRAG(rag_url)
        
        try:
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{rag_url}/health")
                if health_response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service unhealthy: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Could not reach RAG service: {e}")
        
        # Initial chat context
        initial_ctx = ChatContext().append(
            role="system",
            text=assistant.system_prompt
        )
        
        # Create voice assistant
        voice_assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="gpt-4o-mini",  # Verwende OpenAI direkt statt Ollama für bessere Stabilität
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"  # Männliche, freundliche Stimme
            ),
            chat_ctx=initial_ctx
        )
        
        # Custom message handler für RAG integration
        @voice_assistant.on("user_speech_committed")
        async def on_user_speech(user_message: str):
            logger.info(f"Customer said: {user_message}")
            
            # Search in RAG
            rag_results = await assistant.search_knowledge(user_message)
            
            if rag_results:
                # Enhance the context with RAG results
                enhanced_prompt = assistant.create_enhanced_prompt(user_message, rag_results)
                
                # Update the chat context
                voice_assistant._chat_ctx.append(
                    role="system",
                    text=f"Nutze diese Informationen für deine Antwort:\n{enhanced_prompt}"
                )
        
        # Start the assistant
        voice_assistant.start(ctx.room)
        
        # Generate initial greeting
        await voice_assistant.say(
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
