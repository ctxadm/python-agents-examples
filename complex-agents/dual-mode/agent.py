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
import numpy as np

logger = logging.getLogger("dual-model-vision-agent")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.146:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava-llama3:latest")
FUNCTION_MODEL = os.getenv("FUNCTION_MODEL", "llama3.2:latest")

# RAG Service
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

class DualModelVisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._stream_task = None  # Track the stream task
        self._last_image_context = ""
        self._current_loaded_model = None
        self._model_lock = asyncio.Lock()

        # Function Model (llama3.2) für LiveKit
        function_llm = openai.LLM(
            model=FUNCTION_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.3
        )

        super().__init__(
            instructions="""You are an assistant with vision capabilities and access to a knowledge base.

            CRITICAL RULES:
            1. If you get a vision error, say "I cannot see the image right now due to a technical issue"
            2. NEVER make up or imagine what's on screen if vision analysis fails
            3. Keep vision and knowledge base answers separate
            
            You can handle TWO types of queries:
            
            1. VISION QUERIES (about what you see):
               - "What do you see?"
               - "What's on the screen?"
               - "Describe the image"
               → Use vision analysis, do NOT search the knowledge base
               → If vision fails: Say "I cannot analyze the image at the moment"
            
            2. KNOWLEDGE QUERIES (about Swiss regulations, BAKOM, frequencies):
               - Questions about licenses, permits, frequencies
               - BAKOM, Funkkonzession, regulations
               → Use search_knowledge function
            
            IMPORTANT RULES:
            - First determine if it's a VISION or KNOWLEDGE query
            - For VISION queries: Describe ONLY what you actually see
            - For KNOWLEDGE queries: Search the knowledge base first
            - Only talk about BAKOM when specifically asked about Swiss regulations
            
            When searching the knowledge base:
            - Items under "Konzessionsfreie Funkdienste" = no license required
            - Items under "Verbotene Funkdienste" = prohibited
            - Be accurate, don't invent details
            
            Be honest about technical limitations.""",
            stt=deepgram.STT(),
            llm=function_llm,
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )

    async def _get_loaded_models_info(self) -> dict:
        """Get information about currently loaded models from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
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
                    "keep_alive": 0
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

    async def _preload_model(self, model_name: str, keep_alive: str = "5m") -> bool:
        """Preload a model in Ollama memory and wait for it to be ready"""
        try:
            logger.info(f"Preloading model {model_name} with keep_alive={keep_alive}")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": keep_alive
                }
                
                if "llava" in model_name.lower():
                    img = Image.new('RGB', (1, 1), color='black')
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    payload["images"] = [img_base64]
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✅ Model {model_name} preloaded successfully")
                        self._current_loaded_model = model_name
                        return True
                    else:
                        logger.error(f"Failed to preload model {model_name}: {resp.status}")
                        return False
        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {e}")
            return False

    async def _ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure the required model is loaded, managing GPU memory efficiently"""
        async with self._model_lock:
            loaded_models = await self._get_loaded_models_info()
            
            if model_name in loaded_models:
                logger.info(f"Model {model_name} already loaded, updating keep_alive")
                return await self._preload_model(model_name, "5m")
            
            for loaded_model in loaded_models:
                if loaded_model != model_name:
                    logger.info(f"Unloading {loaded_model} to make room for {model_name}")
                    await self._unload_model(loaded_model)
                    await asyncio.sleep(2)
            
            logger.info(f"Loading {model_name}, this may take a moment...")
            success = await self._preload_model(model_name, "5m")
            
            if success:
                if "llava" in model_name:
                    logger.info("Waiting for vision model to fully initialize...")
                    await asyncio.sleep(3)
                logger.info(f"Model {model_name} is ready")
            else:
                logger.error(f"Failed to load model {model_name}")
            
            return success

    async def _analyze_frame_with_vision(self, frame) -> str:
        """Analysiert Frame mit Vision Model (llava)"""
        try:
            logger.info("Preparing vision model...")
            model_ready = await self._ensure_model_loaded(VISION_MODEL)
            
            if not model_ready:
                logger.error("Vision model failed to load")
                return "Vision model is not available at the moment. Please try again later."
            
            await asyncio.sleep(1)
            
            # Enhanced frame conversion with better error handling
            try:
                # Get frame data based on format
                if hasattr(frame, 'data') and hasattr(frame, 'width') and hasattr(frame, 'height'):
                    # Direct frame data
                    frame_data = frame.data
                    width = frame.width
                    height = frame.height
                elif hasattr(frame, 'frame'):
                    # Wrapped frame
                    frame_data = frame.frame.data
                    width = frame.frame.width
                    height = frame.frame.height
                else:
                    logger.error(f"Unknown frame format: {type(frame)}")
                    return "Error: Unknown frame format"
                
                # Validate frame data
                expected_size = width * height * 3  # RGB
                if len(frame_data) < expected_size:
                    logger.error(f"Frame data too small: {len(frame_data)} < {expected_size}")
                    # Try to handle different pixel formats
                    if len(frame_data) == width * height * 4:  # RGBA
                        # Convert RGBA to RGB
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
                        frame_array = frame_array[:, :, :3]  # Drop alpha channel
                        pil_image = Image.fromarray(frame_array, 'RGB')
                    else:
                        return "Error: Invalid frame data size"
                else:
                    # Standard RGB conversion
                    pil_image = Image.frombytes('RGB', (width, height), frame_data)
                
                logger.info(f"Successfully converted frame: {width}x{height}")
                
            except Exception as e:
                logger.error(f"Frame conversion error: {e}")
                return "Error processing the image frame"
            
            # Convert to Base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info(f"Sending image to {VISION_MODEL} for analysis...")
            
            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": VISION_MODEL,
                    "prompt": "Describe what you see in this image. Be specific about any text, logos, or documents visible.",
                    "images": [img_base64],
                    "stream": False,
                    "keep_alive": "2m"
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
                        return description
                    else:
                        error_text = await resp.text()
                        logger.error(f"Vision API error {resp.status}: {error_text}")
                        return "Error analyzing image - vision service returned an error"
                        
        except asyncio.TimeoutError:
            logger.error("Vision analysis timed out")
            return "Vision analysis timed out. The image might be too complex or the model is busy."
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            return f"Could not analyze image due to technical error: {str(e)}"

    @function_tool(description="Search the knowledge base for information about regulations, BAKOM, frequencies, etc.")
    async def search_knowledge(
        self,
        query: str,
        use_image_context: Union[bool, str, None] = False
    ) -> str:
        """Search the knowledge base for specific information"""
        
        await self._ensure_model_loaded(FUNCTION_MODEL)
        
        if isinstance(use_image_context, str):
            use_image_context = use_image_context.lower() not in ['false', 'null', 'none', '']
        elif use_image_context is None:
            use_image_context = False
        
        logger.info(f"search_knowledge called with query='{query}', use_image_context={use_image_context} (type: {type(use_image_context)})")
        
        if use_image_context and self._last_image_context:
            query = f"{query} (Image shows: {self._last_image_context[:200]})"
        
        logger.info(f"RAG Search initiated: {query}")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "top_k": 5
                }
                
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
                    payload["collection"] = "scrape-data"
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
        
        await self._test_connections()
        
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
                            
                        loaded = await self._get_loaded_models_info()
                        if loaded:
                            logger.info(f"Currently loaded in Ollama: {list(loaded.keys())}")
                        
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Wird aufgerufen wenn User spricht"""
        
        if self._latest_frame:
            logger.info(f"Analyzing frame: {self._latest_frame.width}x{self._latest_frame.height}")
            
            vision_result = await self._analyze_frame_with_vision(self._latest_frame)
            self._last_image_context = vision_result
            
            original_content = str(new_message.content)
            enhanced_content = f"{original_content}\n\n[Vision Context: {vision_result}]"
            
            new_message.content = enhanced_content
            
            logger.info(f"Enhanced message with vision context")
        else:
            logger.info("No frame available for analysis")
            self._last_image_context = ""

    def _create_video_stream(self, track: rtc.Track):
        # Cancel existing stream task if any
        if self._stream_task and not self._stream_task.done():
            logger.info("Cancelling existing video stream task")
            self._stream_task.cancel()
        
        if self._video_stream is not None:
            self._video_stream = None

        self._video_stream = rtc.VideoStream(track)

        async def read_stream():
            frame_count = 0
            try:
                async for event in self._video_stream:
                    frame_count += 1
                    self._latest_frame = event.frame

                    if frame_count % 30 == 0:
                        logger.info(f"Video stream active: {frame_count} frames received")
            except asyncio.CancelledError:
                logger.info(f"Video stream cancelled after {frame_count} frames")
                raise
            except Exception as e:
                logger.error(f"Video stream error: {e}")

        self._stream_task = asyncio.create_task(read_stream())

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
