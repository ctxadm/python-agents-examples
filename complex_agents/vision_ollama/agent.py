# LiveKit Agents - Vision Agent (Production Ready)
import logging
import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.llm import function_tool, ImageContent, ChatContext, ChatMessage
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "vision-1")
WORKER_ID = os.getenv("WORKER_ID", "vision-worker-1")

class ConversationState(Enum):
    """State Machine für Konversationsphasen"""
    GREETING = "greeting"
    AWAITING_FRAME = "awaiting_frame"
    ANALYZING = "analyzing"
    PROVIDING_ANALYSIS = "providing_analysis"

@dataclass
class VisionContext:
    """Kontext für Vision-Analyse"""
    latest_frame: Optional[rtc.VideoFrame] = None
    frame_timestamp: Optional[float] = None
    video_stream: Optional[rtc.VideoStream] = None
    video_task: Optional[asyncio.Task] = None
    frame_count: int = 0
    analysis_count: int = 0
    
    def has_recent_frame(self, max_age_seconds: float = 2.0) -> bool:
        """Prüft ob ein aktueller Frame vorhanden ist"""
        if not self.latest_frame or not self.frame_timestamp:
            return False
        age = asyncio.get_event_loop().time() - self.frame_timestamp
        return age <= max_age_seconds
    
    def reset(self):
        """Reset des Kontexts"""
        self.latest_frame = None
        self.frame_timestamp = None
        self.frame_count = 0

@dataclass  
class VisionUserData:
    """User data context für den Vision Agent"""
    authenticated_user: Optional[str] = None
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    vision_context: VisionContext = field(default_factory=VisionContext)
    last_analysis: Optional[str] = None
    session_start_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class VisionAssistant(Agent):
    """Vision Assistant für Bildanalyse"""
    
    def __init__(self) -> None:
        # Instructions OHNE Function Tools  
        super().__init__(instructions="""Du bist ein Vision-Assistent der sehen und beschreiben kann. ANTWORTE NUR AUF DEUTSCH.

KRITISCHE REGELN:
1. Du KANNST das Kamerabild sehen - es wird automatisch hinzugefügt wenn verfügbar
2. Beschreibe NUR was du tatsächlich siehst - NICHTS erfinden!
3. Wenn kein Bild im Context → "Ich kann gerade kein Bild sehen"
4. Halte Antworten kurz und präzise (max. 50 Wörter)
5. NIEMALS Entschuldigungen - nutze "Leider" statt "Sorry"

ANTWORT-STRUKTUR bei Bildanalyse:
"Ich sehe: [Kurze, präzise Beschreibung des Bildinhalts]"

Bei Fragen zum Bild:
- Beantworte NUR was direkt gefragt wird
- Sei spezifisch und akkurat
- Keine Spekulationen über nicht sichtbare Dinge

VERBOTENE WÖRTER: "Entschuldigung", "Es tut mir leid", "Sorry" """)
        
        self._vision_context = None  # Will be set from session
        logger.info(f"✅ VisionAssistant initialized for worker {WORKER_ID}")
    
    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Vision Agent on_enter called")
        
        # Get vision context from session
        from livekit.agents import get_job_context
        ctx = get_job_context()
        
        # Find the session userdata
        if hasattr(self, '_session') and hasattr(self._session, 'userdata'):
            self._vision_context = self._session.userdata.vision_context
            logger.info("✅ Vision context linked to agent")
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called after user finishes speaking - add vision frame to message"""
        if self._vision_context and self._vision_context.has_recent_frame():
            logger.info(f"📸 Adding frame to user message: {self._vision_context.latest_frame.width}x{self._vision_context.latest_frame.height}")
            
            # Add frame to the message content
            if isinstance(new_message.content, str):
                new_message.content = [new_message.content, ImageContent(image=self._vision_context.latest_frame)]
            elif isinstance(new_message.content, list):
                new_message.content.append(ImageContent(image=self._vision_context.latest_frame))
            
            # Update analysis count
            self._vision_context.analysis_count += 1
        else:
            logger.warning("⚠️ No recent frame available for vision analysis")


async def request_handler(ctx: JobContext):
    """Request handler mit Room-Filter"""
    room_name = ctx.room.name if ctx.room else "unknown"
    logger.info(f"[{AGENT_NAME}] 📥 Request received for room: {room_name} on worker {WORKER_ID}")
    
    # Accept vision rooms
    if room_name.startswith("vision_room_"):
        logger.info(f"[{AGENT_NAME}] ✅ Accepting - correct type")
        await ctx.accept()
    else:
        logger.info(f"[{AGENT_NAME}] ❌ Rejecting - wrong type (expected: vision_room_*)")


async def entrypoint(ctx: JobContext):
    """Entry point für den Vision Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🏁 Starting Vision Agent Session: {session_id}")
    logger.info(f"🤖 Worker: {WORKER_ID}")
    logger.info("="*50)
    
    session = None
    session_closed = False
    
    # Register disconnect handler FIRST
    def on_disconnect():
        nonlocal session_closed
        logger.info(f"[{session_id}] Room disconnected event received")
        session_closed = True
    
    if ctx.room:
        ctx.room.on("disconnected", on_disconnect)
    
    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10
        
        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                    logger.info(f"✅ [{session_id}] Audio track found and subscribed")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
            
            await asyncio.sleep(1)
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
        
        if not audio_track_received:
            logger.warning(f"⚠️ [{session_id}] No audio track found after {max_wait_time}s, continuing anyway...")
        
        # 4. Configure LLM with Ollama Vision
        ollama_host = os.getenv("OLLAMA_HOST", "http://172.16.0.136:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")
        
        llm = openai.LLM(
            model=ollama_model,
            base_url=f"{ollama_host}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.0,  # Deterministisch wie Garage Agent
        )
        logger.info(f"🤖 [{session_id}] Using Ollama Vision: {ollama_model} at {ollama_host}")
        
        # 5. Create session
        session = AgentSession[VisionUserData](
            userdata=VisionUserData(
                authenticated_user=None,
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                vision_context=VisionContext(),
                last_analysis=None,
                session_start_time=asyncio.get_event_loop().time()
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,
                min_speech_duration=0.2,
                activation_threshold=0.5
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )
        
        # 6. Setup video tracking
        await setup_video_tracking(ctx.room, session)
        
        # 7. Create agent
        agent = VisionAssistant()
        
        # 8. Start session (GENAU WIE GARAGE AGENT!)
        logger.info(f"🏁 [{session_id}] Starting session...")
        
        # Link session to agent for vision context access
        agent._session = session
        
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Warte auf Audio-Stabilisierung
        await asyncio.sleep(2.0)
        
        # Event handlers (wie Garage Agent)
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
        
        @session.on("agent_state_changed") 
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("function_call")
        def on_function_call(event):
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
        
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Generated response preview: {response_preview}...")
        
        # 9. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = """Hallo! Ich bin Ihr Vision-Assistent.

Ich kann sehen was Sie mir über Ihre Kamera zeigen und Ihre Fragen dazu beantworten.

Aktivieren Sie bitte Ihre Kamera und fragen Sie mich dann, was ich sehe!"""
            
            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_FRAME
            
            # Retry-Mechanismus wie Garage Agent
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await session.say(
                        greeting_text,
                        allow_interruptions=True,
                        add_to_chat_ctx=True
                    )
                    logger.info(f"✅ [{session_id}] Initial greeting sent successfully")
                    break
                except Exception as e:
                    logger.warning(f"[{session_id}] Greeting attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error after all retries: {e}", exc_info=True)
        
        logger.info(f"✅ [{session_id}] Vision Agent ready with Ollama Vision!")
        
        # Log video status
        if session.userdata.vision_context.video_stream:
            logger.info(f"📹 [{session_id}] Video stream active")
        else:
            logger.warning(f"⚠️ [{session_id}] No video stream detected yet")
        
        # Wait for disconnect
        disconnect_event = asyncio.Event()
        
        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()
        
        ctx.room.on("disconnected", handle_disconnect)
        
        await disconnect_event.wait()
        logger.info(f"[{session_id}] Session ending...")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup (wie Garage Agent)
        if session:
            # Cancel video task if exists
            if session.userdata.vision_context.video_task:
                session.userdata.vision_context.video_task.cancel()
                try:
                    await session.userdata.vision_context.video_task
                except asyncio.CancelledError:
                    pass
        
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass
        
        logger.info(f"✅ [{session_id}] Cleanup complete")
        logger.info("="*50)


async def setup_video_tracking(room: rtc.Room, session: AgentSession[VisionUserData]):
    """Setup video stream tracking"""
    logger.info("📹 Setting up video tracking...")
    
    # Find existing video tracks
    video_found = False
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 Found existing video track from {participant.identity}")
                await create_video_stream(publication.track, session)
                video_found = True
                break
        if video_found:
            break
    
    # Watch for new video tracks
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"📹 New video track subscribed from {participant.identity}")
            asyncio.create_task(create_video_stream(track, session))


async def create_video_stream(track: rtc.Track, session: AgentSession[VisionUserData]):
    """Create video stream to capture frames"""
    vision_ctx = session.userdata.vision_context
    
    # Cancel old task if exists
    if vision_ctx.video_task and not vision_ctx.video_task.done():
        logger.info("🔄 Cancelling old video task")
        vision_ctx.video_task.cancel()
        try:
            await vision_ctx.video_task
        except asyncio.CancelledError:
            pass
    
    # Clean up old stream
    if vision_ctx.video_stream is not None:
        vision_ctx.video_stream = None
    
    # Create new stream
    vision_ctx.video_stream = rtc.VideoStream(track)
    logger.info("✅ Video stream created")
    
    async def read_stream():
        """Read frames from video stream"""
        try:
            async for event in vision_ctx.video_stream:
                vision_ctx.frame_count += 1
                vision_ctx.latest_frame = event.frame
                vision_ctx.frame_timestamp = asyncio.get_event_loop().time()
                
                # Log every 30 frames
                if vision_ctx.frame_count % 30 == 0:
                    logger.info(f"📸 Captured {vision_ctx.frame_count} frames, latest: {event.frame.width}x{event.frame.height}")
                
                # Update conversation state
                if session.userdata.conversation_state == ConversationState.AWAITING_FRAME:
                    session.userdata.conversation_state = ConversationState.ANALYZING
                    
        except asyncio.CancelledError:
            logger.info("📹 Video stream reading cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Error reading video stream: {e}")
    
    # Store the task
    vision_ctx.video_task = asyncio.create_task(read_stream())
    logger.info("✅ Video stream reading task started")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
