# LiveKit Agents 1.0.x Version
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("vision-agent-ollama")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")  # Updated model name

class VisionAgent(Agent):
    def __init__(self) -> None:
        # Ollama über OpenAI-kompatible API
        super().__init__(
            instructions="""You are a vision assistant that can see and describe what's in the video.
            
            FIRST RESPONSE: "Hallo! Ich kann sehen, was Sie mir zeigen. Wie kann ich helfen?"
            
            IMPORTANT:
            - Keep responses under 50 words
            - Only answer what is directly asked
            - Be accurate when describing what you see
            - Respond in German
            - Do not offer additional help or ask follow-up questions
            """,
            llm=openai.LLM(
                model=OLLAMA_MODEL,
                base_url=f"{OLLAMA_HOST}/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.3,
            ),
            stt=deepgram.STT(language="de"),
            tts=openai.TTS(model="tts-1", voice="nova"),
            vad=silero.VAD.load(
                min_silence_duration=0.8,
                min_speech_duration=0.3,
                activation_threshold=0.5
            )
        )
        
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        self._room = None
        
        logger.info(f"Vision Agent initialized with Ollama at {OLLAMA_HOST} using model {OLLAMA_MODEL}")
    
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("Vision Agent entering session")
        
        # Get room from context
        from livekit.agents import get_job_context
        self._room = get_job_context().room
        
        # Setup video tracking
        await self._setup_video_tracking()
        
        # Greet user
        await self.session.say(
            "Hallo! Ich kann sehen, was Sie mir zeigen. Wie kann ich helfen?",
            allow_interruptions=True
        )
    
    async def on_user_message(self, message: ChatMessage) -> None:
        """Process user message with latest video frame"""
        if self._latest_frame:
            logger.info(f"Adding frame to message: {self._latest_frame.width}x{self._latest_frame.height}")
            # Add image to the message content
            if isinstance(message.content, str):
                message.content = [message.content, ImageContent(image=self._latest_frame)]
            elif isinstance(message.content, list):
                message.content.append(ImageContent(image=self._latest_frame))
        else:
            logger.warning("No frame available for vision analysis")
    
    async def _setup_video_tracking(self):
        """Setup video stream tracking"""
        # Find existing video tracks
        if self._room.remote_participants:
            for participant in self._room.remote_participants.values():
                for publication in participant.track_publications.values():
                    if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                        self._create_video_stream(publication.track)
                        break
        
        # Watch for new video tracks
        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track, 
            publication: rtc.RemoteTrackPublication, 
            participant: rtc.RemoteParticipant
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"New video track subscribed from {participant.identity}")
                self._create_video_stream(track)
    
    def _create_video_stream(self, track: rtc.Track):
        """Create video stream to capture frames"""
        # Clean up old stream
        if self._video_stream is not None:
            self._video_stream = None
        
        # Create new stream
        self._video_stream = rtc.VideoStream(track)
        logger.info("Video stream created")
        
        async def read_stream():
            frame_count = 0
            try:
                async for event in self._video_stream:
                    frame_count += 1
                    self._latest_frame = event.frame
                    
                    # Log every 30 frames
                    if frame_count % 30 == 0:
                        logger.info(f"Captured {frame_count} frames, latest: {event.frame.width}x{event.frame.height}")
            except Exception as e:
                logger.error(f"Error reading video stream: {e}")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)
    
    async def on_leave(self):
        """Cleanup when agent leaves"""
        logger.info("Vision Agent leaving session")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        
        # Clean up video stream
        if self._video_stream:
            self._video_stream = None

async def entrypoint(ctx: JobContext):
    """Entry point for vision agent"""
    logger.info("=== Vision Agent Starting (1.0.x) ===")
    
    # Connect to room
    await ctx.connect()
    logger.info("Connected to room")
    
    # Create session with new 1.0.x API
    session = AgentSession()
    
    # Start session with agent instance
    await session.start(
        room=ctx.room,
        agent=VisionAgent()
    )
    
    logger.info("Vision Agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
