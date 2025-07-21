# basics/medical_agent/agent.py
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

logger = logging.getLogger("medical-assistant")

class MedicalAssistantWithRAG:
    """Medical Assistant with RAG integration"""
    
    def __init__(self, rag_url: str = None):
        self.rag_url = rag_url or os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "medical_nutrition")
        logger.info(f"MedicalAssistant initialized with collection: {self.rag_collection}")
        
        # System prompt
        self.system_prompt = """Du bist eine kompetente und erfahrene Arztsekretärin in einer medizinischen Praxis mit Zugriff auf die elektronische Patientenakte.

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
    
    def create_enhanced_prompt(self, user_message: str, rag_results: List[Dict]) -> str:
        """Erstellt einen erweiterten Prompt mit RAG-Ergebnissen"""
        if not rag_results:
            return user_message
        
        enhanced = f"Nutzeranfrage: {user_message}\n\n"
        enhanced += "Relevante Informationen aus der Patientendatenbank:\n\n"
        
        for idx, result in enumerate(rag_results, 1):
            enhanced += f"{idx}. {result['content']}\n"
            if result.get('metadata'):
                enhanced += f"   Details: {json.dumps(result['metadata'], ensure_ascii=False)}\n"
            enhanced += "\n"
        
        enhanced += "\nBitte beantworte die Anfrage basierend auf diesen Datenbankinformationen."
        return enhanced

async def entrypoint(ctx: JobContext):
    """Main entry point for the medical assistant agent"""
    logger.info("Medical assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Test RAG connection
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        assistant = MedicalAssistantWithRAG(rag_url)
        
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
                model="gpt-4o-mini",  # Verwende OpenAI direkt statt Ollama
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"
            ),
            chat_ctx=initial_ctx
        )
        
        # Custom message handler für RAG integration
        @voice_assistant.on("user_speech_committed")
        async def on_user_speech(user_message: str):
            logger.info(f"User said: {user_message}")
            
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
