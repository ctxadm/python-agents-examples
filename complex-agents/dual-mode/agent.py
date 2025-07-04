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
        self._processing_vision = False  # Add flag to prevent concurrent vision processing

        # Function Model (llama3.2) für LiveKit
        function_llm = openai.LLM(
            model=FUNCTION_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.3
        )

        super().__init__(
            instructions="""Ich bin ein KI-Assistent mit Bilderkennungs-Fähigkeiten und Zugriff auf eine Wissensdatenbank.

            WICHTIG: Wenn meine Eingabe bereits eine Bildschirmbeschreibung enthält (z.B. "Ja, ich kann deinen Bildschirm sehen..."), 
            dann gebe ich diese Information DIREKT und UNVERÄNDERT wieder.
            
            Wenn ich nach dem Bildschirm oder was ich sehe gefragt werde:
            - Gebe ich die bereitgestellte Bildschirmbeschreibung direkt wieder
            - Ich füge keine zusätzlichen Erklärungen hinzu
            
            Für WISSENS-ANFRAGEN (über Schweizer Regulierungen, BAKOM, Frequenzen):
            - Nutze die search_knowledge Funktion für genaue Informationen
            - Fokus auf Regulierungen, Lizenzen und Frequenzzuteilungen
            - Artikel unter "Konzessionsfreie Funkdienste" = keine Lizenz erforderlich
            - Artikel unter "Verbotene Funkdienste" = verboten
            
            Wenn Bildschirm-Kontext in eckigen Klammern verfügbar ist, nutze ich diesen für relevante Antworten.""",
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
                return "Vision Model ist momentan nicht verfügbar. Bitte versuche es später erneut."
            
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
                    return "Fehler: Unbekanntes Frame-Format"
                
                # Check data size to determine format
                actual_size = len(frame_data)
                rgb_size = width * height * 3
                rgba_size = width * height * 4
                yuv420_size = int(width * height * 1.5)
                
                logger.info(f"Frame info: {width}x{height}, data size: {actual_size} bytes")
                logger.info(f"Expected sizes - RGB: {rgb_size}, RGBA: {rgba_size}, YUV420: {yuv420_size}")
                
                if actual_size == rgb_size:
                    # Standard RGB
                    pil_image = Image.frombytes('RGB', (width, height), frame_data)
                    logger.info("Detected RGB format")
                
                elif actual_size == rgba_size:
                    # RGBA format
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
                    frame_array = frame_array[:, :, :3]  # Drop alpha channel
                    pil_image = Image.fromarray(frame_array, 'RGB')
                    logger.info("Detected RGBA format, converted to RGB")
                
                elif actual_size == yuv420_size:
                    # YUV420/I420 format - most common in video streaming
                    logger.info("Detected YUV420/I420 format, converting to RGB")
                    
                    # Convert YUV420 to RGB
                    y_size = width * height
                    uv_size = (width // 2) * (height // 2)
                    
                    # Extract Y, U, V planes
                    y_data = np.frombuffer(frame_data[:y_size], dtype=np.uint8).reshape((height, width))
                    u_data = np.frombuffer(frame_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
                    v_data = np.frombuffer(frame_data[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))
                    
                    # Upsample U and V to full resolution
                    u_data = np.repeat(np.repeat(u_data, 2, axis=0), 2, axis=1)
                    v_data = np.repeat(np.repeat(v_data, 2, axis=0), 2, axis=1)
                    
                    # Convert YUV to RGB using standard conversion formula
                    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Vectorized conversion
                    y = y_data.astype(np.float32)
                    u = u_data.astype(np.float32) - 128
                    v = v_data.astype(np.float32) - 128
                    
                    rgb_array[:, :, 0] = np.clip(y + 1.402 * v, 0, 255).astype(np.uint8)  # R
                    rgb_array[:, :, 1] = np.clip(y - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)  # G
                    rgb_array[:, :, 2] = np.clip(y + 1.772 * u, 0, 255).astype(np.uint8)  # B
                    
                    pil_image = Image.fromarray(rgb_array, 'RGB')
                    logger.info("Successfully converted YUV420 to RGB")
                
                else:
                    logger.error(f"Unknown frame format. Size: {actual_size}, expected RGB: {rgb_size}, RGBA: {rgba_size}, YUV420: {yuv420_size}")
                    return "Fehler: Unbekanntes Video-Frame-Format"
                
                logger.info(f"Successfully processed frame: {width}x{height}")
                
            except Exception as e:
                logger.error(f"Frame conversion error: {e}", exc_info=True)
                return "Fehler beim Verarbeiten des Bildes"
            
            # Convert to Base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info(f"Sending image to {VISION_MODEL} for analysis...")
            
            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": VISION_MODEL,
                    "prompt": "Beschreibe was du in diesem Bild siehst. Sei spezifisch über sichtbare Texte, Logos oder Dokumente.",
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
                        description = result.get('response', 'Konnte das Bild nicht analysieren')
                        logger.info(f"Vision analysis completed: {description[:100]}...")
                        return description
                    else:
                        error_text = await resp.text()
                        logger.error(f"Vision API error {resp.status}: {error_text}")
                        return "Fehler beim Analysieren des Bildes - Vision Service hat einen Fehler zurückgegeben"
                        
        except asyncio.TimeoutError:
            logger.error("Vision analysis timed out")
            return "Vision-Analyse hat zu lange gedauert. Das Bild könnte zu komplex sein oder das Modell ist beschäftigt."
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            return f"Konnte das Bild nicht analysieren aufgrund eines technischen Fehlers: {str(e)}"

    @function_tool(description="Durchsuche die Wissensdatenbank nach Informationen über Regulierungen, BAKOM, Frequenzen, etc.")
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
            query = f"{query} (Bild zeigt: {self._last_image_context[:200]})"
        
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
                                
                                result_text = f"Resultat {i} (Score: {score:.2f}):\n{content}"
                                if metadata.get('topic'):
                                    result_text += f"\nThema: {metadata['topic']}"
                                
                                results.append(result_text)
                            
                            response = f"Fand {len(data['results'])} Resultate in der Wissensdatenbank:\n\n"
                            response += "\n\n---\n\n".join(results)
                            logger.info(f"Returning {len(results)} results to agent")
                            return response
                        else:
                            logger.warning("No results found in knowledge base")
                            return "Keine Resultate in der Wissensdatenbank für diese Anfrage gefunden."
                    else:
                        logger.error(f"Search error: {response_text}")
                        return f"Suchfehler: HTTP {resp.status}"
                        
        except Exception as e:
            logger.error(f"RAG Search error: {e}")
            return f"Konnte Wissensdatenbank nicht durchsuchen: {str(e)}"

    @function_tool(description="Analysiere den aktuellen Bildschirm/Kamera-Feed")
    async def analyze_screen(self) -> str:
        """Analysiert das aktuelle Bild vom Bildschirm oder der Kamera"""
        
        if not self._latest_frame:
            return "Kein Bildschirm oder Kamera-Feed verfügbar. Bitte teile deinen Bildschirm."
        
        if self._processing_vision:
            return "Vision-Analyse läuft bereits, bitte warte einen Moment."
        
        self._processing_vision = True
        try:
            logger.info(f"Analyzing frame via function tool: {self._latest_frame.width}x{self._latest_frame.height}")
            
            vision_result = await self._analyze_frame_with_vision(self._latest_frame)
            self._last_image_context = vision_result
            
            if any(error_indicator in vision_result for error_indicator in 
                  ["Fehler", "nicht analysieren", "nicht verfügbar", "zu lange", "technischen Fehlers"]):
                return f"Fehler bei der Bildanalyse: {vision_result}"
            
            return f"Bildschirmanalyse: {vision_result}"
            
        finally:
            self._processing_vision = False

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
        
        original_content = str(new_message.content)
        logger.info(f"User message: {original_content}")
        
        # Check if this is a vision-related query
        vision_keywords = ['see', 'screen', 'show', 'display', 'image', 'picture', 'what do you see', 
                          'was siehst du', 'bildschirm', 'zeig', 'sehen', 'erkennen', 'siehst du']
        is_vision_query = any(keyword in original_content.lower() for keyword in vision_keywords)
        
        if is_vision_query and self._latest_frame:
            # Check if already processing vision
            if self._processing_vision:
                logger.warning("Already processing vision query, skipping")
                return
                
            self._processing_vision = True
            try:
                logger.info(f"Vision query detected. Analyzing frame: {self._latest_frame.width}x{self._latest_frame.height}")
                
                vision_result = await self._analyze_frame_with_vision(self._latest_frame)
                self._last_image_context = vision_result
                
                # Check if vision analysis was successful
                if not any(error_indicator in vision_result for error_indicator in 
                          ["Fehler", "nicht analysieren", "nicht verfügbar", "zu lange", "technischen Fehlers"]):
                    # Erfolg - Vision-Ergebnis in die Nachricht einfügen
                    logger.info("Vision analysis successful, injecting result into message")
                    
                    # Modifiziere die User-Nachricht, um die Vision-Antwort zu erzwingen
                    vision_response = f"Ja, ich kann deinen Bildschirm sehen. {vision_result}"
                    new_message.content = vision_response
                    logger.info(f"Replaced message content with vision result")
                    
                else:
                    # Fehler bei der Vision-Analyse
                    logger.error(f"Vision analysis failed: {vision_result}")
                    new_message.content = vision_result
                    
            finally:
                self._processing_vision = False
                
        elif is_vision_query and not self._latest_frame:
            logger.warning("Vision query but no frame available")
            # Ersetze die Nachricht mit der Fehlermeldung
            new_message.content = "Ich kann deinen Bildschirm gerade nicht sehen. Bitte stelle sicher, dass du deinen Bildschirm oder deine Kamera teilst."
            
        else:
            # Für Nicht-Vision-Anfragen - füge Kontext hinzu falls verfügbar
            if self._last_image_context:
                # Erweitere die Nachricht mit Vision-Kontext
                enhanced_content = f"{original_content}\n\n[Verfügbarer Bildschirm-Kontext: {self._last_image_context[:500]}]"
                new_message.content = enhanced_content
                logger.info("Enhanced non-vision query with available context")

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
