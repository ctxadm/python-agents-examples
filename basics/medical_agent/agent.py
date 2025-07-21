# basics/medical_agent.py
import os
import logging
import httpx
import json
from typing import Optional, List, Dict
from livekit.agents import (
    JobContext,
    AutoSubscribe,
    llm,
    multimodal
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("medical-assistant")

class RAGEnabledLLM(llm.LLM):
    """Custom LLM that integrates RAG search before generating responses"""
    
    def __init__(self, base_llm: llm.LLM, rag_url: str = "http://localhost:8000"):
        super().__init__()
        self._base_llm = base_llm
        self._rag_url = rag_url
        self._rag_collection = os.getenv("RAG_COLLECTION", "medical_nutrition")
        logger.info(f"RAGEnabledLLM initialized with collection: {self._rag_collection}")
    
    async def _search_knowledge(self, query: str) -> Optional[List[Dict]]:
        """Sucht in der medizinischen Wissensdatenbank"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._rag_url}/search",
                    json={
                        "query": query,
                        "agent_type": "medical",
                        "top_k": 3,
                        "collection": self._rag_collection
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
    
    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: llm.LLMOptions = llm.LLMOptions(),
        fnc_ctx: Optional[llm.FunctionContext] = None,
    ) -> llm.LLMStream:
        """Enhanced chat that includes RAG search results"""
        
        # Get the last user message
        last_message = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user":
                last_message = msg.content
                break
        
        # Search knowledge base if we have a user message
        if last_message:
            logger.info(f"Searching RAG for: {last_message}")
            knowledge = await self._search_knowledge(last_message)
            
            if knowledge:
                # Inject RAG results into the context
                rag_context = "Relevante Informationen aus der Patientendatenbank:\n\n"
                for idx, result in enumerate(knowledge, 1):
                    rag_context += f"{idx}. {result['content']}\n"
                    if result.get('metadata'):
                        rag_context += f"   Details: {json.dumps(result['metadata'], ensure_ascii=False)}\n"
                    rag_context += "\n"
                
                # Add system message with RAG results
                enhanced_messages = chat_ctx.messages.copy()
                enhanced_messages.append(llm.ChatMessage(
                    role="system",
                    content=f"Nutze diese Informationen aus der Datenbank für deine Antwort:\n\n{rag_context}\n\nBeantworte basierend auf diesen Daten die Frage des Nutzers."
                ))
                
                # Create new context with enhanced messages
                enhanced_ctx = llm.ChatContext(messages=enhanced_messages)
                
                # Call base LLM with enhanced context
                return await self._base_llm.chat(
                    chat_ctx=enhanced_ctx,
                    conn_options=conn_options,
                    fnc_ctx=fnc_ctx
                )
        
        # No user message or no RAG results, just use base LLM
        return await self._base_llm.chat(
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            fnc_ctx=fnc_ctx
        )

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
        
        # Create base LLM
        base_llm = openai.LLM(
            model="llama3.2:latest",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.7,
        )
        
        # Wrap with RAG functionality
        rag_llm = RAGEnabledLLM(base_llm, rag_url)
        
        # Initial instructions
        initial_ctx = llm.ChatContext().append(
            role="system",
            text="""Du bist eine kompetente und erfahrene Arztsekretärin in einer medizinischen Praxis mit Zugriff auf die elektronische Patientenakte.

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

DATENSCHUTZ:
- Alle Informationen sind streng vertraulich
- Nur für autorisiertes medizinisches Personal bestimmt
- Schweigepflicht gilt uneingeschränkt

Bereite die Informationen so auf, dass der Arzt optimal auf das Patientengespräch vorbereitet ist."""
        )
        
        # Create voice assistant with RAG-enabled LLM
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=rag_llm,
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"
            ),
            chat_ctx=initial_ctx
        )
        
        # Start the assistant
        assistant.start(ctx.room)
        
        # Generate initial greeting
        await assistant.say(
            "Guten Tag Herr Doktor! Ich bin Ihre digitale Praxisassistentin. "
            "Ich kann Ihnen sofort alle relevanten Patientendaten aus unserer elektronischen Akte zur Verfügung stellen. "
            "Welchen Patienten möchten Sie besprechen?",
            allow_interruptions=True
        )
        
        logger.info("Medical assistant ready with RAG support")
        
    except Exception as e:
        logger.error(f"Error in medical assistant: {e}", exc_info=True)
        raise

__all__ = ['entrypoint']
