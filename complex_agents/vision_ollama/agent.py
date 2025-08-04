# LiveKit Agents - Vision Agent (FIXED)
import asyncio
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from livekit import rtc, agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("vision-agent-ollama")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")
AGENT_NAME = os.getenv("AGENT_NAME", "vision-1")

@dataclass
class VisionUserData:
    """User data context für den Vision Agent"""
    latest_frame: Optional[rtc.VideoFrame] = None
    video_stream: Optional[rtc.VideoStream] = None
    greeting_sent: bool = False

class VisionAssistant(Agent):
    def __init__(self) -> None:
        # Simplified instructions - no STT/TTS/VAD here as they belong to the session
        super().__init__(instructions="""Du bist ein Vision-Assistent der sehen und beschreiben kann, was im Video gezeigt wird.
            
WICHTIG:
- Halte Antworten unter 50 Wörtern
- Beantworte nur was direkt gefragt wird
- Sei präzise beim Beschreiben was du siehst
- Antworte auf Deutsch
- Biete keine zusätzliche Hilfe an oder stelle Nachfragen
""")
        logger.info(f"VisionAssistant initialized for Ollama at {OLLAMA_HOST} using model {OLLAMA_MODEL}")
    
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("Vision Agent entering session")

async def request_handler(ctx: JobContext):
    """Request handler für Room-Filter"""
    room_name = ctx.room.name if ctx.room else "unknown"
    logger.info(f"[{AGENT_NAME}] 📨 Job request received for room: {room_name}")
    
    # Accept any room starting with "vision_room_"
    if room_name.startswith("vision_room_"):
        logger.info(f"[{AGENT_NAME}] ✅ Accepting room: {room_name}")
        await ctx.accept()
    else:
        logger.info(f"[{AGENT_NAME}] ❌ Rejecting room: {room_name} (expected: vision_room_*)")
        # Don't accept - let another agent handle it

async def entrypoint(ctx: JobContext):
    """Entry point für den Vision Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🏁 Starting Vision Agent Session: {session_id}")
    logger.info("="*50)
    
    session = None
    session_closed = False
    
    # Register disconnect handler
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
        
        # 2. Wait for participant (CRITICAL!)
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. Wait for audio track (CRITICAL!)
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
        
        # 4. Configure LLM with Ollama for vision
        llm = openai.LLM(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.3,
        )
        logger.info(f"🤖 [{session_id}] Using Ollama vision model: {OLLAMA_MODEL}")
        
        # 5. Create session with all necessary components
        session = AgentSession[VisionUserData](
            userdata=VisionUserData(
                latest_frame=None,
                video_stream=None,
                greeting_sent=False
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
        
        # 8. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Wait for audio stabilization
        await asyncio.sleep(2.0)
        
        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Generated response preview: {response_preview}...")
        
        # Hook into message processing to add video frames
        original_chat_method = session._fnc_ctx._fnc
        
        async def chat_with_vision(chat_ctx: ChatContext) -> None:
            # Add latest frame to the last user message if available
            if session.userdata.latest_frame and chat_ctx.messages:
                last_msg = chat_ctx.messages[-1]
                if last_msg.role == "user":
                    logger.info(f"📸 Adding frame to message: {session.userdata.latest_frame.width}x{session.userdata.latest_frame.height}")
                    if isinstance(last_msg.content, str):
                        last_msg.content = [last_msg.content, ImageContent(image=session.userdata.latest_frame)]
                    elif isinstance(last_msg.content, list):
                        last_msg.content.append(ImageContent(image=session.userdata.latest_frame))
            
            # Call original method
            return await original_chat_method(chat_ctx)
        
        # Replace the method
        session._fnc_ctx._fnc = chat_with_vision
        
        # 9. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = "Hallo! Ich kann sehen, was Sie mir zeigen. Wie kann ich helfen?"
            
            session.userdata.greeting_sent = True
            
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )
            logger.info(f"✅ [{session_id}] Initial greeting sent successfully")
            
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}", exc_info=True)
        
        logger.info(f"✅ [{session_id}] Vision Agent ready!")
        
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
        # Cleanup
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
    logger.info("Setting up video tracking...")
    
    # Find existing video tracks
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                create_video_stream(publication.track, session)
                break
    
    # Watch for new video tracks
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"New video track subscribed from {participant.identity}")
            create_video_stream(track, session)

def create_video_stream(track: rtc.Track, session: AgentSession[VisionUserData]):
    """Create video stream to capture frames"""
    # Clean up old stream
    if session.userdata.video_stream is not None:
        session.userdata.video_stream = None
    
    # Create new stream
    session.userdata.video_stream = rtc.VideoStream(track)
    logger.info("Video stream created")
    
    async def read_stream():
        frame_count = 0
        try:
            async for event in session.userdata.video_stream:
                frame_count += 1
                session.userdata.latest_frame = event.frame
                
                # Log every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Captured {frame_count} frames, latest: {event.frame.width}x{event.frame.height}")
        except Exception as e:
            logger.error(f"Error reading video stream: {e}")
    
    # Start the stream reading task
    asyncio.create_task(read_stream())

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
