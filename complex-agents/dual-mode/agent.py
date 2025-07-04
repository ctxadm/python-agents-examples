import asyncio
import logging
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import function_tool, ImageContent, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero
import aiohttp
from typing import Optional
import io
from PIL import Image

logger = logging.getLogger("dual-model-vision-agent")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.146:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:latest")
FUNCTION_MODEL = os.getenv("FUNCTION_MODEL", "llama3.1")

# RAG Service
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

class DualModelVisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        self._last_image_context = ""

        # Function Model (llama3.1) für LiveKit
        function_llm = openai.LLM(
            model=FUNCTION_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0
        )

        super().__init__(
            instructions="""You are an assistant with vision capabilities and access to a knowledge base.
            
            You work with two AI models:
            1. A vision model that analyzes images
            2. Your main model that can search the knowledge base
            
            When users ask about:
            - BAKOM, Funkkonzession, Swiss regulations -> search the knowledge base
            - What you see in an image -> use the vision analysis
            - Combine both when needed
            
            Always be accurate and helpful.""",
            stt=deepgram.STT(),
            llm=function_llm,
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )

    async def _analyze_frame_with_vision(self, frame) -> str:
        """Analysiert Frame mit Vision Model (llava)"""
        try:
            # Frame zu PIL Image konvertieren
            pil_image = Image.frombytes('RGB', 
                                      (frame.width, frame.height), 
                                      frame.data)
            
            # Zu Base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info(f"Analyzing image with {VISION_MODEL}")
            
            # Ollama API direkt aufrufen
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": VISION_MODEL,
                    "prompt": "Describe what you see in this image. Be specific about any text, logos, or documents visible.",
                    "images": [img_base64],
                    "stream": False
                }
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        description = result.get('response', 'Could not analyze image')
                        logger.info(f"Vision analysis: {description[:100]}...")
                        return description
                    else:
                        logger.error(f"Vision API error: {resp.status}")
                        return "Error analyzing image"
                        
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return f"Could not analyze image: {str(e)}"

    @function_tool(description="Search the knowledge base for information about regulations, BAKOM, frequencies, etc.")
    async def search_knowledge(
        self,
        query: str,
        use_image_context: bool = False
    ) -> str:
        """Search the knowledge base"""
        
        # Erweitere Query mit Bildkontext wenn gewünscht
        if use_image_context and self._last_image_context:
            query = f"{query} (Image shows: {self._last_image_context[:200]})"
        
        logger.info(f"RAG Search: {query}")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "top_k": 5
                }
                
                # Collection-Auswahl basierend auf Keywords
                query_lower = query.lower()
                if any(word in query_lower for word in ["bakom", "funk", "konzession", "radio", "frequenz"]):
                    payload["collection"] = "scrape-data"
                elif any(word in query_lower for word in ["qdrant", "vector", "database"]):
                    payload["collection"] = "Qdrant-Documentation"
                else:
                    payload["agent_type"] = "general"
                
                async with session.post(
                    f"{RAG_SERVICE_URL}/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('results'):
                            results = []
                            for i, res in enumerate(data['results'][:3], 1):
                                content = res.get('content', '')
                                score = res.get('score', 0)
                                
                                # Kürzen für bessere Lesbarkeit
                                if len(content) > 400:
                                    content = content[:400] + "..."
                                
                                results.append(f"Result {i} (Score: {score:.2f}):\n{content}")
                            
                            return f"Found {len(data['results'])} results:\n\n" + "\n\n---\n\n".join(results)
                        else:
                            return "No results found in the knowledge base."
                    else:
                        return f"Search error: HTTP {resp.status}"
                        
        except Exception as e:
            logger.error(f"RAG Search error: {e}")
            return f"Could not search knowledge base: {str(e)}"

    async def on_enter(self):
        room = get_job_context().room
        
        logger.info(f"Dual Model Agent started")
        logger.info(f"Vision Model: {VISION_MODEL} at {OLLAMA_HOST}")
        logger.info(f"Function Model: {FUNCTION_MODEL} at {OLLAMA_HOST}")
        logger.info(f"RAG Service: {RAG_SERVICE_URL}")
        
        # Test connections
        await self._test_connections()
        
        # Video handling
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])

        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def _test_connections(self):
        """Test alle Services beim Start"""
        # RAG Service
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{RAG_SERVICE_URL}/health") as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        logger.info(f"✅ RAG Service: {health}")
        except Exception as e:
            logger.error(f"❌ RAG Service error: {e}")
        
        # Ollama
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_HOST}/api/tags") as resp:
                    if resp.status == 200:
                        models = await resp.json()
                        logger.info(f"✅ Ollama models: {[m['name'] for m in models.get('models', [])]}")
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Wird aufgerufen wenn User spricht"""
        
        # Wenn ein Frame vorhanden ist, analysiere es
        if self._latest_frame:
            logger.info(f"Frame available: {self._latest_frame.width}x{self._latest_frame.height}")
            
            # Analysiere mit Vision Model
            vision_result = await self._analyze_frame_with_vision(self._latest_frame)
            self._last_image_context = vision_result
            
            # Füge Kontext zur Nachricht hinzu
            enhanced_content = f"{new_message.content}\n\n[Vision Context: I can see {vision_result[:200]}...]"
            new_message.content = enhanced_content
            
            # Frame für LiveKit hinzufügen (optional für Logging)
            new_message.content.append(ImageContent(image=self._latest_frame))
        else:
            logger.info("No frame available")

    def _create_video_stream(self, track: rtc.Track):
        if self._video_stream is not None:
            self._video_stream = None

        self._video_stream = rtc.VideoStream(track)

        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                frame_count += 1
                self._latest_frame = event.frame

                if frame_count % 30 == 0:
                    logger.info(f"Received {frame_count} frames")

        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    logger.info("Starting Dual Model Vision Agent")
    
    session = AgentSession()
    await session.start(
        agent=DualModelVisionAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
