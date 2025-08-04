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

from livekit import rtc
from livekit.agents import JobContext, get_job_context
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero

# Setup logging
logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

# Environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.136:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")


class VisionAgent(Agent):
    """Vision-enabled agent for code analysis"""
    
    def __init__(self) -> None:
        self._latest_frame: Optional[rtc.VideoFrame] = None
        self._video_stream: Optional[rtc.VideoStream] = None
        self._frame_task: Optional[asyncio.Task] = None
        
        # Initialize parent class with configuration
        super().__init__(
            instructions="""You are a PYTHON CODE ERROR DETECTOR. ALWAYS respond in German!

            WHEN YOU SEE CODE IN THE SCREENSHOT:
            1. IMMEDIATELY analyze it line by line
            2. DO NOT ask for code - you already see it!
            3. Find the error and report it
            
            SPECIFIC ERRORS TO FIND:
            - 'trom' instead of 'from' 
            - 'imoprt' instead of 'import'
            - Any typo in Python keywords
            
            YOUR RESPONSE MUST BE:
            "Ich sehe Python-Code. Fehler in Zeile [NUMBER]: '[TYPO]' muss '[CORRECT]' sein."
            
            EXAMPLE for line 15 with 'trom':
            "Ich sehe Python-Code. Fehler in Zeile 15: 'trom' muss 'from' sein."
            
            DO NOT SAY:
            - "Bitte geben Sie den Code an" (Wrong - you see it!)
            - "Ich werde analysieren" (Wrong - do it now!)
            
            ANALYZE THE CODE IN THE IMAGE NOW!""",
            
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            
            llm=openai.LLM.with_ollama(
                model=OLLAMA_MODEL,
                base_url=f"{OLLAMA_HOST}/v1",
                temperature=0.0
            ),
            
            tts=openai.TTS(
                model="tts-1",
                voice="nova",
                speed=1.0
            ),
            
            vad=silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=0.5
            )
        )
        
        logger.info(f"✅ VisionAgent initialized with Ollama at {OLLAMA_HOST}")
    
    async def on_enter(self) -> None:
        """Called when agent enters the room"""
        logger.info("🚪 Agent entering room - on_enter called")
        
        try:
            room = get_job_context().room
            logger.info(f"📍 Room name: {room.name}")
            logger.info(f"👥 Remote participants: {len(room.remote_participants)}")
            
            # Check for existing video tracks
            video_track_found = False
            for participant in room.remote_participants.values():
                logger.info(f"👤 Checking participant: {participant.identity}")
                
                for publication in participant.track_publications.values():
                    if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                        if publication.subscribed:
                            logger.info(f"📹 Found subscribed video track from {participant.identity}")
                            self._setup_video_stream(publication.track)
                            video_track_found = True
                            break
                        else:
                            logger.info(f"📹 Found unsubscribed video track from {participant.identity}")
                
                if video_track_found:
                    break
            
            if not video_track_found:
                logger.warning("⚠️ No video tracks found on enter")
            
            # Listen for new tracks
            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant
            ):
                logger.info(f"🎬 track_subscribed event: {track.kind} from {participant.identity}")
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info(f"📹 New video track subscribed from {participant.identity}")
                    self._setup_video_stream(track)
            
            # Listen for track unsubscribed
            @room.on("track_unsubscribed")
            def on_track_unsubscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant
            ):
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info(f"📹 Video track unsubscribed from {participant.identity}")
                    self._cleanup_video_stream()
                    
        except Exception as e:
            logger.error(f"❌ Error in on_enter: {e}", exc_info=True)
    
    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """Attach video frame to user's message"""
        logger.info("💬 on_user_turn_completed called")
        logger.info(f"📝 User message: {new_message.content}")
        
        if self._latest_frame:
            logger.info("📸 Attaching video frame to message")
            
            try:
                # Create image content from frame
                image_content = ImageContent(image=self._latest_frame)
                new_message.content.append(image_content)
                
                # Clear frame after use
                self._latest_frame = None
                logger.info("✅ Frame attached successfully to user message")
            except Exception as e:
                logger.error(f"❌ Error attaching frame: {e}", exc_info=True)
        else:
            logger.warning("⚠️ No video frame available for attachment")
    
    def _setup_video_stream(self, track: rtc.Track) -> None:
        """Setup video stream from track"""
        try:
            logger.info("🎥 Setting up video stream...")
            
            # Cleanup existing stream
            self._cleanup_video_stream()
            
            # Create new video stream
            self._video_stream = rtc.VideoStream(track)
            logger.info("📹 Video stream created successfully")
            
            # Start frame capture task
            self._frame_task = asyncio.create_task(self._capture_frames())
            logger.info("🚀 Frame capture task started")
            
        except Exception as e:
            logger.error(f"❌ Error setting up video stream: {e}", exc_info=True)
    
    def _cleanup_video_stream(self) -> None:
        """Cleanup video stream and tasks"""
        logger.info("🧹 Cleaning up video stream...")
        
        if self._frame_task and not self._frame_task.done():
            self._frame_task.cancel()
            self._frame_task = None
            logger.info("✅ Frame task cancelled")
        
        if self._video_stream:
            try:
                # Note: close() might not exist in some versions
                if hasattr(self._video_stream, 'close'):
                    self._video_stream.close()
            except Exception as e:
                logger.warning(f"⚠️ Error closing video stream: {e}")
            self._video_stream = None
            logger.info("✅ Video stream cleaned up")
    
    async def _capture_frames(self) -> None:
        """Capture frames from video stream"""
        frame_count = 0
        
        try:
            logger.info("🎥 Starting frame capture loop")
            
            async for event in self._video_stream:
                if hasattr(event, 'frame'):
                    self._latest_frame = event.frame
                    frame_count += 1
                    
                    # Log progress every 30 frames (~1 second)
                    if frame_count % 30 == 0:
                        logger.info(f"📸 Captured {frame_count} frames")
                    
                    # Log first frame
                    if frame_count == 1:
                        logger.info("🎉 First frame captured successfully!")
                        
        except asyncio.CancelledError:
            logger.info(f"🛑 Frame capture cancelled after {frame_count} frames")
        except Exception as e:
            logger.error(f"❌ Error capturing frames: {e}", exc_info=True)


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the vision agent"""
    logger.info("=" * 60)
    logger.info("🚀 Vision Agent Starting (LiveKit Agents 1.0.23)")
    logger.info(f"📍 Room: {ctx.room.name if ctx.room else 'Unknown'}")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
    logger.info("=" * 60)
    
    # Create agent
    agent = VisionAgent()
    logger.info("✅ Agent instance created")
    
    # Create session
    session = AgentSession()
    logger.info("✅ AgentSession created")
    
    # Start the session
    logger.info("🚀 Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("✅ Vision agent session started successfully")
    
    # WICHTIG: Warte bis die Session beendet wird!
    # Ohne das beendet sich die entrypoint sofort
    await asyncio.sleep(float('inf'))  # Warte für immer


# WICHTIG: Der folgende Code wird NUR ausgeführt wenn das Script direkt gestartet wird
# NICHT wenn es über den Multi-Agent Wrapper importiert wird!
# Für den Multi-Agent Wrapper wird NUR die entrypoint Funktion oben verwendet.

if __name__ == "__main__":
    # Dieser Block wird NICHT vom Multi-Agent Wrapper ausgeführt!
    from livekit.agents import JobRequest, WorkerOptions
    
    async def request_handler(request: JobRequest) -> None:
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
        request_fnc=request_handler
    ))
