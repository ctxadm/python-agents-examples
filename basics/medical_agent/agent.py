# basics/medical_agent/agent.py
import os
import logging
import httpx
import json
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe
from livekit.plugins import deepgram, openai, silero
from livekit.agents.voice_assistant import VoiceAssistant

logger = logging.getLogger("medical-assistant")

class MedicalAssistantWithRAG:
    """Medical Assistant with RAG integration"""
    
    def __init__(self, rag_url: str = None):
        self.rag_url = rag_url or os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "medical_nutrition")
        self.chat_history = []
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
    
    def create_enhanced_messages(self, user_message: str, rag_results: List[Dict]) -> List[dict]:
        """Erstellt Chat-Nachrichten mit RAG-Ergebnissen"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Füge Chat-Historie hinzu
        messages.extend(self.chat_history)
        
        # Füge RAG-Kontext hinzu wenn vorhanden
        if rag_results:
            rag_context = "Relevante Informationen aus der Patientendatenbank:\n\n"
            for idx, result in enumerate(rag_results, 1):
                rag_context += f"{idx}. {result['content']}\n"
                if result.get('metadata'):
                    rag_context += f"   Details: {json.dumps(result['metadata'], ensure_ascii=False)}\n"
                rag_context += "\n"
            
            messages.append({
                "role": "system",
                "content": f"Nutze diese Datenbankinformationen für deine Antwort:\n\n{rag_context}"
            })
        
        # Füge die aktuelle Nutzeranfrage hinzu
        messages.append({"role": "user", "content": user_message})
        
        return messages

async def entrypoint(ctx: JobContext):
    """Main entry point for the medical assistant agent"""
    logger.info("Medical assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Initialize assistant
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        assistant = MedicalAssistantWithRAG(rag_url)
        
        # Test RAG connection
        try:
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{rag_url}/health")
                if health_response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service unhealthy: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Could not reach RAG service: {e}")
        
        # Create initial messages for the LLM
        initial_messages = [
            {"role": "system", "content": assistant.system_prompt}
        ]
        
        # Custom LLM wrapper that integrates RAG
        class RAGIntegratedLLM:
            def __init__(self, base_llm, assistant):
                self.base_llm = base_llm
                self.assistant = assistant
            
            async def chat(self, messages):
                # Extract last user message
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break
                
                if last_user_msg:
                    # Search RAG
                    rag_results = await self.assistant.search_knowledge(last_user_msg)
                    
                    # Create enhanced messages
                    if rag_results:
                        enhanced_messages = self.assistant.create_enhanced_messages(last_user_msg, rag_results)
                        return await self.base_llm.chat(enhanced_messages)
                
                # No RAG results, use original messages
                return await self.base_llm.chat(messages)
        
        # Create base LLM
        base_llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        
        # Wrap with RAG integration
        rag_llm = RAGIntegratedLLM(base_llm, assistant)
        
        # Create voice assistant with custom LLM
        voice_assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=rag_llm,
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"
            )
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
        
        # Keep the assistant running
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"Error in medical assistant: {e}", exc_info=True)
        raise

# Import asyncio at the top if not already done
import asyncio

# Ensure the module can be imported
__all__ = ['entrypoint']
