# basics/garage_agent/agent.py
import os
import logging
import httpx
import json
from typing import Optional, List, Dict
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe
from livekit.plugins import deepgram, openai, silero
from livekit.agents.voice_assistant import VoiceAssistant

logger = logging.getLogger("garage-assistant")

class GarageAssistantWithRAG:
    """Garage Assistant with RAG integration"""
    
    def __init__(self, rag_url: str = None):
        self.rag_url = rag_url or os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.rag_collection = os.getenv("RAG_COLLECTION", "automotive_docs")
        self.chat_history = []
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
    
    def create_enhanced_messages(self, user_message: str, rag_results: List[Dict]) -> List[dict]:
        """Erstellt Chat-Nachrichten mit RAG-Ergebnissen"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Füge Chat-Historie hinzu
        messages.extend(self.chat_history)
        
        # Füge RAG-Kontext hinzu wenn vorhanden
        if rag_results:
            rag_context = "Relevante Informationen aus der Fahrzeugdatenbank:\n\n"
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
    """Main entry point for the garage assistant agent"""
    logger.info("Garage assistant starting with RAG support")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Initialize assistant
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        assistant = GarageAssistantWithRAG(rag_url)
        
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
                voice="nova"  # Männliche, freundliche Stimme
            )
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
        
        # Keep the assistant running
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"Error in garage assistant: {e}", exc_info=True)
        raise

# Import asyncio at the top if not already done
import asyncio

# Ensure the module can be imported
__all__ = ['entrypoint']
