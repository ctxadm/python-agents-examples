# complex-agents/agent-selector/agent_selector.py

import asyncio
import logging
import os
import sys
from typing import Optional, Dict
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, ollama
import httpx
from typing import Annotated

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
        
        # VAD
        vad = silero.VAD.load()
        
        # Vision Agent Context
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "Du bist ein hilfreicher KI-Assistent mit Vision-Fähigkeiten. "
                "Du kannst Bilder analysieren und Fragen dazu beantworten. "
                "Sei freundlich, präzise und hilfreich. "
                "Antworte auf Deutsch, wenn du auf Deutsch angesprochen wirst."
            )
        )
        
        # Ollama Konfiguration
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        logger.info(f"Verwende Ollama Model: {ollama_model} auf {ollama_host}")
        
        assistant = VoiceAssistant(
            vad=vad,
            stt=deepgram.STT(
                language="de-DE"  # Deutsch als primäre Sprache
            ),
            llm=ollama.LLM(
                model=ollama_model,
                base_url=ollama_host
            ),
            tts=openai.TTS(
                voice="alloy",
                model="tts-1",
                language="de"  # Deutsche Stimme
            ),
            chat_ctx=initial_ctx,
        )
        
        # Starte den Assistant
        assistant.start(ctx.room)
        
        # Warte auf Room-Ende
        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            logger.info(f"Participant {participant.identity} hat Room verlassen")
        
        # Agent am Leben halten
        await asyncio.Future()
    
    async def run_rag_agent(self, ctx: JobContext, room_id: str):
        """RAG Agent mit Qdrant Integration"""
        logger.info(f"RAG Agent initialisiert für Room {room_id}")
        
        # VAD
        vad = silero.VAD.load()
        
        # RAG Service URL
        rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # Function Context für RAG
        class RagFunctions(llm.FunctionContext):
            @llm.ai_callable()
            async def search_knowledge_base(
                self,
                query: Annotated[str, llm.TypeInfo(description="Die Suchanfrage für die Wissensdatenbank")]
            ):
                """Durchsucht die Qdrant Wissensdatenbank nach relevanten Informationen"""
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
                            collection = data.get("collection_used", "unknown")
                            
                            logger.info(f"RAG gefunden: {len(results)} Ergebnisse aus Collection '{collection}'")
                            
                            if results:
                                formatted = "\n\n".join([
                                    f"[Relevanz: {r['score']:.2f}]\n{r['content']}"
                                    for r in results
                                ])
                                return f"Gefundene Informationen aus {collection}:\n\n{formatted}"
                            else:
                                return "Keine relevanten Informationen in der Wissensdatenbank gefunden."
                        else:
                            logger.error(f"RAG Service Error: {response.status_code}")
                            return f"Fehler beim Zugriff auf die Wissensdatenbank (Status: {response.status_code})"
                            
                except Exception as e:
                    logger.error(f"RAG Service Exception: {e}")
                    return f"Verbindungsfehler zur Wissensdatenbank: {str(e)}"
            
            @llm.ai_callable()
            async def list_available_collections(self):
                """Zeigt alle verfügbaren Sammlungen in der Wissensdatenbank"""
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(f"{rag_service_url}/collections")
                        
                        if response.status_code == 200:
                            data = response.json()
                            collections = data.get("collections", [])
                            
                            if collections:
                                formatted = "\n".join([
                                    f"- {c['name']} ({c['documents']} Dokumente)"
                                    for c in collections
                                ])
                                return f"Verfügbare Wissenssammlungen:\n{formatted}"
                            else:
                                return "Keine Sammlungen in der Wissensdatenbank gefunden."
                        else:
                            return "Fehler beim Abrufen der Sammlungen"
                            
                except Exception as e:
                    logger.error(f"Collections Error: {e}")
                    return "Fehler beim Verbinden zur Wissensdatenbank"
        
        # RAG Context
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "Du bist ein intelligenter KI-Assistent mit Zugriff auf eine umfangreiche Wissensdatenbank. "
                "Nutze die Funktion 'search_knowledge_base' um nach relevanten Informationen zu suchen. "
                "Beantworte Fragen präzise basierend auf den gefundenen Informationen. "
                "Wenn du dir unsicher bist, suche in der Wissensdatenbank nach. "
                "Du kannst auch 'list_available_collections' nutzen um zu sehen, welche Wissensgebiete verfügbar sind. "
                "Antworte auf Deutsch, wenn du auf Deutsch angesprochen wirst."
            )
        )
        
        # Function Context
        fnc_ctx = RagFunctions()
        
        assistant = VoiceAssistant(
            vad=vad,
            stt=deepgram.STT(
                language="de-DE"
            ),
            llm=openai.LLM(
                model="gpt-4-turbo-preview",
                temperature=0.7
            ),
            tts=openai.TTS(
                voice="alloy",
                model="tts-1",
                language="de"
            ),
            chat_ctx=initial_ctx,
            fnc_ctx=fnc_ctx,
        )
        
        # Starte den Assistant
        assistant.start(ctx.room)
        
        # Agent am Leben halten
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
    
    # Umgebungsvariablen prüfen
    logger.info("=== Agent Selector gestartet ===")
    logger.info(f"RAG Service URL: {os.getenv('RAG_SERVICE_URL', 'http://localhost:8000')}")
    logger.info(f"Ollama Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    logger.info(f"Ollama Model: {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}")
    
    selector = AgentSelector()
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=selector.entrypoint,
        )
    )
