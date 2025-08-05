#!/usr/bin/env python3
"""
Vision Agent für LiveKit Multi-Agent System
Kompatibel mit LiveKit Agents 1.0.23 (NEUE API)
Pfad: python-agents/complex_agents/vision_ollama/agent.py
"""

import asyncio
import logging
import os
from typing import Optional
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    JobContext, 
    WorkerOptions,
    cli
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero

# Setup logging
logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

# Environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.136:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-vision-1")


@dataclass
class VisionUserData:
    """User data context für den Vision Agent"""
    latest_frame: Optional[rtc.VideoFrame] = None
    video_stream: Optional[rtc.VideoStream] = None
    frame_count: int = 0
    tasks: list = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []


async def request_handler(ctx: JobContext):
    """Request handler"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the vision agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("=" * 60)
    logger.info(f"🚀 Vision Agent Starting (LiveKit Agents 1.0.23)")
    logger.info(f"📍 Room: {room_name}")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
    logger.info(f"🆔 Session: {session_id}")
    logger.info("=" * 60)
    
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
        
        # 3. Wait for audio track (WICHTIG für STT!)
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
        
        # 4. Configure LLM with Ollama
        llm = openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            temperature=0.0
        )
        logger.info(f"🤖 [{session_id}] Using {OLLAMA_MODEL}")
        
        # 5. Create session with UserData
        userdata = VisionUserData()
        
        session = AgentSession[VisionUserData](
            userdata=userdata,
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,
                min_speech_duration=0.2
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova",
                speed=1.0
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )
        
        # 6. Setup instructions
        instructions = """Du bist ein Python Fehler-Finder. Antworte IMMER auf Deutsch!

DEINE AUFGABE: Finde Tippfehler in Python-Code.

WENN DU CODE IM BILD SIEHST:
1. Schaue dir jede Zeile genau an
2. Suche nach falsch geschriebenen Python-Keywords
3. Melde den Fehler sofort

HÄUFIGE TIPPFEHLER:
- 'trom' statt 'from'
- 'imoprt' statt 'import'
- 'defn' statt 'def'
- 'retrun' statt 'return'

ANTWORT-FORMAT:
"Ich sehe Python-Code. Fehler in Zeile [NUMMER]: '[TIPPFEHLER]' muss '[RICHTIG]' sein."

BEISPIEL:
Wenn du siehst: trom math import sqrt
Sage: "Ich sehe Python-Code. Fehler in Zeile 15: 'trom' muss 'from' sein."

WICHTIG:
- Wenn kein Code sichtbar → "Bitte teilen Sie Ihren Bildschirm"
- Wenn Code sichtbar → Analysiere und finde den Fehler
- Sei präzise mit der Zeilennummer und dem Fehler"""
        
        # Set instructions on session
        await session.update_instructions(instructions)
        
        # 7. Setup video track handlers
        video_track_found = False
        
        # Check for existing video tracks
        for track_pub in participant.track_publications.values():
            if track_pub.kind == rtc.TrackKind.KIND_VIDEO and track_pub.track is not None:
                logger.info(f"📹 [{session_id}] Found existing video track")
                await setup_video_stream(track_pub.track, userdata, session_id)
                video_track_found = True
                break
        
        # Setup track subscription handlers
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant
        ):
            logger.info(f"🎬 [{session_id}] track_subscribed: {track.kind} from {participant.identity}")
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(setup_video_stream(track, userdata, session_id))
        
        @ctx.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 [{session_id}] Video track unsubscribed")
                cleanup_video_stream(userdata)
        
        # 8. Setup event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            
            # Attach frame to context if available
            if userdata.latest_frame:
                logger.info(f"[{session_id}] 📸 Attaching frame to context")
                # Add frame to chat context
                asyncio.create_task(add_frame_to_context(session, userdata))
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Response: {response_preview}...")
        
        # 9. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room
        )
        
        # Wait for audio stabilization
        await asyncio.sleep(2.0)
        
        # 10. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        greeting_text = """Hallo! Ich bin Ihr Python Code-Assistent.
        
Ich kann Tippfehler in Python-Code für Sie finden. Bitte teilen Sie Ihren Bildschirm und zeigen Sie mir den Code, den ich überprüfen soll.

Sagen Sie einfach "Prüfe meinen Code" oder ähnliches, wenn Sie bereit sind."""
        
        # Retry mechanism for greeting
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
                    logger.error(f"[{session_id}] Failed to send greeting after {max_retries} attempts")
        
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
        if userdata:
            cleanup_video_stream(userdata)
        
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass
        
        logger.info(f"✅ [{session_id}] Cleanup complete")
        logger.info("=" * 60)


async def setup_video_stream(track: rtc.Track, userdata: VisionUserData, session_id: str):
    """Setup video stream from track"""
    try:
        logger.info(f"🎥 [{session_id}] Setting up video stream...")
        
        # Cleanup existing stream
        cleanup_video_stream(userdata)
        
        # Create new video stream
        userdata.video_stream = rtc.VideoStream(track)
        logger.info(f"📹 [{session_id}] Video stream created successfully")
        
        # Start frame capture task
        async def capture_frames():
            frame_count = 0
            try:
                logger.info(f"🎥 [{session_id}] Starting frame capture loop")
                
                async for event in userdata.video_stream:
                    if hasattr(event, 'frame'):
                        userdata.latest_frame = event.frame
                        userdata.frame_count += 1
                        frame_count += 1
                        
                        # Log progress every 30 frames (~1 second)
                        if frame_count % 30 == 0:
                            logger.info(f"📸 [{session_id}] Captured {frame_count} frames")
                        
                        # Log first frame
                        if frame_count == 1:
                            logger.info(f"🎉 [{session_id}] First frame captured successfully!")
                            
            except asyncio.CancelledError:
                logger.info(f"🛑 [{session_id}] Frame capture cancelled after {frame_count} frames")
            except Exception as e:
                logger.error(f"❌ [{session_id}] Error capturing frames: {e}", exc_info=True)
        
        # Create and store the task
        task = asyncio.create_task(capture_frames())
        task.add_done_callback(lambda t: userdata.tasks.remove(t) if t in userdata.tasks else None)
        userdata.tasks.append(task)
        logger.info(f"🚀 [{session_id}] Frame capture task started")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error setting up video stream: {e}", exc_info=True)


def cleanup_video_stream(userdata: VisionUserData):
    """Cleanup video stream and tasks"""
    logger.info("🧹 Cleaning up video stream...")
    
    # Cancel all tasks
    for task in userdata.tasks:
        if not task.done():
            task.cancel()
    userdata.tasks.clear()
    
    if userdata.video_stream:
        try:
            if hasattr(userdata.video_stream, 'close'):
                userdata.video_stream.close()
        except Exception as e:
            logger.warning(f"⚠️ Error closing video stream: {e}")
        userdata.video_stream = None
        logger.info("✅ Video stream cleaned up")


async def add_frame_to_context(session: AgentSession, userdata: VisionUserData):
    """Add frame to chat context"""
    if userdata.latest_frame:
        try:
            # Create image content
            image_content = ImageContent(image=userdata.latest_frame)
            
            # Add to chat context
            chat_ctx = session.chat_ctx
            if chat_ctx and len(chat_ctx.messages) > 0:
                last_message = chat_ctx.messages[-1]
                
                # Add image to last user message
                if last_message.role == "user":
                    if isinstance(last_message.content, str):
                        last_message.content = [last_message.content, image_content]
                    elif isinstance(last_message.content, list):
                        last_message.content.append(image_content)
                    
                    logger.info("✅ Frame added to chat context")
            
            # Clear frame after use
            userdata.latest_frame = None
            
        except Exception as e:
            logger.error(f"❌ Error adding frame to context: {e}", exc_info=True)


# WICHTIG: Der folgende Code wird NUR ausgeführt wenn das Script direkt gestartet wird
if __name__ == "__main__":
    # Dieser Block wird NICHT vom Multi-Agent Wrapper ausgeführt!
    
    async def standalone_request_handler(request):
        """Accept vision room requests - NUR für Standalone Mode"""
        room_name = request.room.name if request.room else "unknown"
        logger.info(f"📥 Request for room: {room_name}")
        
        if room_name.startswith("vision_room"):
            logger.info(f"✅ Accepting vision room: {room_name}")
            await request.accept()
        else:
            logger.info(f"❌ Rejecting non-vision room: {room_name}")
            await request.reject()
    
    logger.info("🏁 Starting Vision Agent Worker (Standalone Mode)")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
    
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=standalone_request_handler
    ))
