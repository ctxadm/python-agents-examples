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
from typing import Optional, Union
import io
from PIL import Image

logger = logging.getLogger("dual-model-vision-agent")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.146:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava-llama3:latest")
FUNCTION_MODEL = os.getenv("FUNCTION_MODEL", "llama3.2:latest")  # Updated to llama3.2

# RAG Service
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

class DualModelVisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        self._last_image_context = ""
        self._current_loaded_model = None  # Track which model is currently loaded
        self._model_lock = asyncio.Lock()  # Prevent concurrent model operations

        # Function Model (llama3.2) für LiveKit
        function_llm = openai.LLM(
            model=FUNCTION_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.3  # Niedrigere Temperatur für konsistentere Antworten
        )

        super().__init__(
            instructions="""You are an assistant with vision capabilities and access to a knowledge base.
            
            You can handle TWO types of queries:
            
            1. VISION QUERIES (about what you see):
               - "What do you see?"
               - "What's on the screen?"
               - "Describe the image"
               → Use vision analysis, do NOT search the knowledge base
            
            2. KNOWLEDGE QUERIES (about Swiss regulations, BAKOM, frequencies):
               - Questions about licenses, permits, frequencies
               - BAKOM, Funkkonzession, regulations
               → Use search_knowledge function
            
            IMPORTANT RULES:
            - First determine if it's a VISION or KNOWLEDGE query
            - For VISION queries: Describe what you see, do NOT mention BAKOM
            - For KNOWLEDGE queries: Search the knowledge base first
            - Only talk about BAKOM when specifically asked about Swiss regulations
            
            When searching the knowledge base:
            - Items under "Konzessionsfreie Funkdienste" = no license required
            - Items under "Verbotene Funkdienste" = prohibited
            - Be accurate, don't invent details
            
            Be helpful and respond appropriately to the type of query.""",
            stt=deepgram.STT(),
            llm=function_llm,
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )

    async def _get_loaded_models_info(self) -> dict:
        """Get information about currently loaded models from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try to get running models (ps endpoint)
                async with session.get(f"{OLLAMA_HOST}/api/ps") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get('models', [])
                        logger.info(f"Currently loaded models: {[m.get('name', 'unknown') for m in models]}")
                        return {m['name']: m for m in models}
                    else:
                        logger.debug("No models currently loaded or ps endpoint not available")
                        return {}
        except Exception as e:
            logger.debug(f"Could not get loaded models info: {e}")
            return {}

    async def _unload_model(self, model_name: str):
        """Request Ollama to unload a model by setting keep_alive to 0"""
        try:
            logger.info(f"Requesting to unload model: {model_name}")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0  # This tells Ollama to unload the model
                }
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ Model {model_name} unload requested")
                    else:
                        logger.warning(f"Could not unload model {model_name}: {resp.status}")
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")

    async def _preload_model(self, model_name: str, keep_alive: str = "5m"):
        """Preload a model in Ollama memory"""
        try:
            logger.info(f"Preloading model {model_name} with keep_alive={keep_alive}")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": keep_alive
                }
                
                # For vision models, add a dummy image
                if "llava" in model_name.lower():
                    img = Image.new('RGB', (1, 1), color='black')
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    payload["images"] = [img_base64]
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        await resp.json()
                        logger.info(f"✅ Model {model_name} preloaded successfully")
                        self._current_loaded_model = model_name
                    else:
                        logger.error(f"Failed to preload model {model_name}: {resp.status}")
        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {e}")

    async def _ensure_model_loaded(self, model_name: str):
        """Ensure the required model is loaded, managing GPU memory efficiently"""
        async with self._model_lock:
            # Check what's currently loaded
            loaded_models = await self._get_loaded_models_info()
            
            # If the model is already loaded, just update keep_alive
            if model_name in loaded_models:
                logger.info(f"Model {model_name} already loaded, updating keep_alive")
                await self._preload_model(model_name, "5m")
                return
            
            # If a different model is loaded, unload it first to free GPU memory
            for loaded_model in loaded_models:
                if loaded_model != model_name:
                    logger.info(f"Unloading {loaded_model} to make room for {model_name}")
                    await self._unload_model(loaded_model)
                    await asyncio.sleep(1)  # Give Ollama time to free memory
            
            # Load the requested model
            await self._preload_model(model_name, "5m")

    async def _analyze_frame_with_vision(self, frame) -> str:
        """Analysiert Frame mit Vision Model (llava)"""
        try:
            # Ensure vision model is loaded
            await self._ensure_model_loaded(VISION_MODEL)
            
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
                    "stream": False,
                    "keep_alive": "2m"  # Keep vision model loaded for 2 minutes
                }
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        description = result.get('response', 'Could not analyze image')
                        logger.info(f"Vision analysis completed: {description[:100]}...")
                        
                        # After vision analysis, we could switch back to function model
                        # But let's keep it loaded for a bit in case of follow-up questions
                        
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
        use_image_context: Union[bool, str, None] = False
    ) -> str:
        """Search the knowledge base for specific information"""
        
        # Note: The function model (llama3.2) should already be loaded by the main LLM
        # But we can ensure it stays loaded
        await self._ensure_model_loaded(FUNCTION_MODEL)
        
        # Robust handling of use_image_context parameter
        if isinstance(use_image_context, str):
            use_image_context = use_image_context.lower() not in ['false', 'null', 'none', '']
        elif use_image_context is None:
            use_image_context = False
        
        # Log the received parameters for debugging
        logger.info(f"search_knowledge called with query='{query}', use_image_context={use_image_context} (type: {type(use_image_context)})")
        
        # Erweitere Query mit Bildkontext wenn gewünscht
        if use_image_context and self._last_image_context:
            query = f"{query} (Image shows: {self._last_image_context[:200]})"
        
        logger.info(f"RAG Search initiated: {query}")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "top_k": 5
                }
                
                # Collection-Auswahl basierend auf Keywords
                query_lower = query.lower()
                if any(word in query_lower for word in ["bakom", "funk", "konzession", "radio", "frequenz", 
                                                         "schweiz", "swiss", "pmr", "cb-funk", "freenet", 
                                                         "konzessionsfrei", "verboten"]):
                    payload["collection"] = "scrape-data"
                    logger.info("Using scrape-data collection")
                elif any(word in query_lower for word in ["qdrant", "vector", "database"]):
                    payload["collection"] = "Qdrant-Documentation"
                    logger.info("Using Qdrant-Documentation collection")
                else:
                    payload["collection"] = "scrape-data"  # Default to scrape-data
                    logger.info("Defaulting to scrape-data collection")
                
                logger.info(f"Sending search request: {payload}")
                
                async with session.post(
                    f"{RAG_SERVICE_URL}/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    response_text = await resp.text()
                    logger.info(f"RAG Response status: {resp.status}")
                    
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('results'):
                            results = []
                            for i, res in enumerate(data['results'][:3], 1):
                                content = res.get('content', '')
                                score = res.get('score', 0)
                                metadata = res.get('metadata', {})
                                
                                # Kürzen für bessere Lesbarkeit
                                if len(content) > 400:
                                    content = content[:400] + "..."
                                
                                result_text = f"Result {i} (Score: {score:.2f}):\n{content}"
                                if metadata.get('topic'):
                                    result_text += f"\nTopic: {metadata['topic']}"
                                
                                results.append(result_text)
                            
                            response = f"Found {len(data['results'])} results in knowledge base:\n\n"
                            response += "\n\n---\n\n".join(results)
                            logger.info(f"Returning {len(results)} results to agent")
                            return response
                        else:
                            logger.warning("No results found in knowledge base")
                            return "No results found in the knowledge base for this query."
                    else:
                        logger.error(f"Search error: {response_text}")
                        return f"Search error: HTTP {resp.status}"
                        
        except Exception as e:
            logger.error(f"RAG Search error: {e}")
            return f"Could not search knowledge base: {str(e)}"

    async def on_enter(self):
        room = get_job_context().room
        
        logger.info("=" * 50)
        logger.info("Dual Model Agent starting...")
        logger.info(f"Vision Model: {VISION_MODEL} (5.5 GB) at {OLLAMA_HOST}")
        logger.info(f"Function Model: {FUNCTION_MODEL} (2.0 GB) at {OLLAMA_HOST}")
        logger.info(f"RAG Service: {RAG_SERVICE_URL}")
        logger.info("=" * 50)
        
        # Test connections
        await self._test_connections()
        
        # Preload the function model as it's used most often
        logger.info("Preloading function model for faster responses...")
        await self._ensure_model_loaded(FUNCTION_MODEL)
        
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
                logger.info("New video track subscribed")
                self._create_video_stream(track)

    async def _test_connections(self):
        """Test alle Services beim Start"""
        # RAG Service
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{RAG_SERVICE_URL}/health") as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        logger.info(f"✅ RAG Service connected: {health}")
                        
                        # Collections anzeigen
                        async with session.get(f"{RAG_SERVICE_URL}/collections") as resp2:
                            if resp2.status == 200:
                                collections = await resp2.json()
                                logger.info(f"Available collections: {collections}")
        except Exception as e:
            logger.error(f"❌ RAG Service error: {e}")
        
        # Ollama
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_HOST}/api/tags") as resp:
                    if resp.status == 200:
                        models = await resp.json()
                        available_models = [m['name'] for m in models.get('models', [])]
                        logger.info(f"✅ Ollama connected. Available models: {available_models}")
                        
                        # Check if required models are available
                        vision_found = False
                        function_found = False
                        
                        for model in available_models:
                            if VISION_MODEL in model:
                                vision_found = True
                                logger.info(f"✅ Vision model found: {model}")
                            if FUNCTION_MODEL in model:
                                function_found = True
                                logger.info(f"✅ Function model found: {model}")
                        
                        if not vision_found:
                            logger.error(f"❌ Vision model {VISION_MODEL} not found!")
                        if not function_found:
                            logger.error(f"❌ Function model {FUNCTION_MODEL} not found!")
                            
                        # Show what's currently loaded
                        loaded = await self._get_loaded_models_info()
                        if loaded:
                            logger.info(f"Currently loaded in Ollama: {list(loaded.keys())}")
                        
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Wird aufgerufen wenn User spricht"""
        
        # Wenn ein Frame vorhanden ist, analysiere es
        if self._latest_frame:
            logger.info(f"Analyzing frame: {self._latest_frame.width}x{self._latest_frame.height}")
            
            # Analysiere mit Vision Model
            vision_result = await self._analyze_frame_with_vision(self._latest_frame)
            self._last_image_context = vision_result
            
            # Erweitere die Nachricht mit Vision-Kontext
            original_content = str(new_message.content)
            enhanced_content = f"{original_content}\n\n[Vision Context: {vision_result}]"
            
            # Erstelle neue Nachricht mit erweitertem Inhalt
            new_message.content = enhanced_content
            
            logger.info(f"Enhanced message with vision context")
        else:
            logger.info("No frame available for analysis")
            self._last_image_context = ""

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
                    logger.info(f"Video stream active: {frame_count} frames received")

        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info("Dual Model Vision Agent - Entrypoint reached")
    
    session = AgentSession()
    await session.start(
        agent=DualModelVisionAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
