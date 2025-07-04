# complex-agents/agent-selector/agent.py

import asyncio
import logging
import os
import sys
from typing import Optional, Dict, Annotated
from livekit import agents
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, openai, silero, ollama
import httpx

# Korrekte Imports für VoiceAssistant
try:
    from livekit.agents.voice_assistant import VoiceAssistant
except ImportError:
    # Fallback für ältere Versionen
    from livekit.agents import VoiceAssistant

logger = logging.getLogger("agent-selector")
logger.setLevel(logging.INFO)

class AgentSelector:
    def __init__(self):
        self.active_sessions: Dict[str, asyncio.Task] = {}
        
    async def entrypoint(self, ctx: JobContext):
        """Haupteinstiegspunkt - wählt Agent basierend auf Room-Namen"""
        
        room_name = ctx.room.name.lower()
        room_id = ctx.room.sid
        
        logger.info(f"=== Neuer Job für Room: '{ctx.room.name}' (ID: {room_id}) ===")
        
        # Verbindung aufbauen
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        
        # Prüfe ob bereits ein Agent für diesen Room läuft
        if room_id in self.active_sessions:
            logger.warning(f"Agent bereits aktiv für Room {room_id}")
            return
        
        # Agent-Auswahl basierend auf Room-Namen
        if any(keyword in room_name for keyword in ['vision', 'ollama', 'bild', 'image', 'visual']):
            logger.info("→ Starte Vision-Ollama Agent")
            agent_task = asyncio.create_task(self.run_vision_agent(ctx, room_id))
            
        elif any(keyword in room_name for keyword in ['rag', 'knowledge', 'wissen', 'datenbank', 'qdrant', 'search']):
            logger.info("→ Starte RAG Agent")
            agent_task = asyncio.create_task(self.run_rag_agent(ctx, room_id))
            
        else:
            # Standard: Vision Agent
            logger.info("→ Kein spezifisches Keyword, starte Vision-Ollama Agent als Standard")
            agent_task = asyncio.create_task(self.run_vision_agent(ctx, room_id))
        
        # Task speichern
        self.active_sessions[room_id] = agent_task
        
        try:
            await agent_task
        except Exception as e:
            logger.error(f"Agent Error in Room {room_id}: {e}")
        finally:
            if room_id in self.active_sessions:
                del self.active_sessions[room_id]
                logger.info(f"Agent für Room {room_id} beendet")
    
    async def run_vision_agent(self, ctx: JobContext, room_id: str):
        """Vision-Ollama Agent"""
        logger.info(f"Vision-Ollama Agent initialisiert für Room {room_id}")
        
        try:
            # Import aus dem Repository
            sys.path.insert(0, '/app')
            from complex_agents.vision_ollama.agent import entrypoint as vision_entrypoint
            
            # Rufe die Original-Entrypoint-Funktion auf
            await vision_entrypoint(ctx)
            
        except ImportError:
            logger.error("Vision-Ollama Agent nicht gefunden, verwende Fallback")
            
            # Fallback Implementation
            vad = silero.VAD.load()
            
            initial_ctx = llm.ChatContext().append(
                role="system",
                text=(
                    "Du bist ein hilfreicher KI-Assistent mit Vision-Fähigkeiten. "
                    "Du kannst Bilder analysieren und Fragen dazu beantworten."
                )
            )
            
            assistant = VoiceAssistant(
                vad=vad,
                stt=deepgram.STT(),
                llm=ollama.LLM(
                    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
                ),
                tts=openai.TTS(voice="alloy"),
                chat_ctx=initial_ctx,
            )
            
            assistant.start(ctx.room)
            await asyncio.Future()
    
    async def run_rag_agent(self, ctx: JobContext, room_id: str):
        """RAG Agent mit Qdrant Integration"""
        logger.info(f"RAG Agent initialisiert für Room {room_id}")
        
        try:
            # Import aus dem Repository
            sys.path.insert(0, '/app')
            from rag.agent import entrypoint as rag_entrypoint
            
            # Rufe die Original-Entrypoint-Funktion auf
            await rag_entrypoint(ctx)
            
        except ImportError:
            logger.error("RAG Agent nicht gefunden, verwende Fallback")
            
            # Fallback Implementation
            vad = silero.VAD.load()
            
            # RAG Service URL
            rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
            
            class RagFunctions(llm.FunctionContext):
                @llm.ai_callable()
                async def search_knowledge_base(
                    self,
                    query: Annotated[str, llm.TypeInfo(description="Suchanfrage")]
                ):
                    """Durchsucht die Wissensdatenbank"""
                    logger.info(f"RAG Suche: {query}")
                    
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.post(
                                f"{rag_service_url}/search",
                                json={
                                    "query": query,
                                    "agent_type": "general",
                                    "top_k": 3
                                }
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                results = data.get("results", [])
                                
                                if results:
                                    formatted = "\n\n".join([
                                        f"[Score: {r['score']:.2f}] {r['content']}"
                                        for r in results
                                    ])
                                    return f"Ergebnisse:\n\n{formatted}"
                                else:
                                    return "Keine Ergebnisse gefunden."
                            else:
                                return f"Fehler: Status {response.status_code}"
                                
                    except Exception as e:
                        logger.error(f"RAG Error: {e}")
                        return f"Fehler: {str(e)}"
            
            initial_ctx = llm.ChatContext().append(
                role="system",
                text=(
                    "Du bist ein KI-Assistent mit Zugriff auf eine Wissensdatenbank. "
                    "Nutze search_knowledge_base() für Informationen."
                )
            )
            
            fnc_ctx = RagFunctions()
            
            assistant = VoiceAssistant(
                vad=vad,
                stt=deepgram.STT(),
                llm=openai.LLM(model="gpt-4-turbo-preview"),
                tts=openai.TTS(voice="alloy"),
                chat_ctx=initial_ctx,
                fnc_ctx=fnc_ctx,
            )
            
            assistant.start(ctx.room)
            await asyncio.Future()

# Hauptprogramm
if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("=== Agent Selector gestartet ===")
    logger.info(f"RAG Service URL: {os.getenv('RAG_SERVICE_URL', 'http://localhost:8000')}")
    logger.info(f"Ollama Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    
    selector = AgentSelector()
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=selector.entrypoint,
        )
    )
