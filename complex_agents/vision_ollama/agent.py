#!/usr/bin/env python3
"""
LiveKit Vision Agent - Standalone Version
Kompatibel mit LiveKit Agents SDK 1.0.x
"""

import asyncio
import logging
import os
from typing import Optional

from livekit import rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    get_job_context
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.voice import Agent
from livekit.plugins import openai, silero

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vision-agent")

# Environment variables with defaults
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
            instructions="""Du bist ein Code-Analyse-Spezialist mit Vision-Fähigkeiten.
            Du kannst sehen was der User dir über die Kamera zeigt.
            
            WICHTIGE REGELN:
            - Antworte IMMER auf Deutsch
            - Wenn du Code siehst, analysiere ihn genau
            - Finde Syntax-Fehler und nenne die Zeilennummer
            - Gib konkrete Korrekturen
            - Halte Antworten kurz und präzise (max 2-3 Sätze)
            
            Wenn du nach Code-Fehlern gefragt wirst, schaue genau hin!""",
            
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            
            llm=openai.LLM.with_ollama(
                model=OLLAMA_MODEL,
                base_url=f"{OLLAMA_HOST}/v1",
                temperature=0.0  # Deterministisch für Code-Analyse
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
        logger.info("🚪 Agent entering room")
        
        # Get current room from job context
        try:
            room = get_job_context().room
            logger.info(f"📍 Room name: {room.name}")
            logger.info(f"👥 Participants: {len(room.remote_participants)}")
            
            # Subscribe to existing video tracks
            for participant in room.remote_participants.values():
                logger.info(f"👤 Checking participant: {participant.identity}")
                
                for publication in participant.track_publications.values():
                    if (publication.track and 
                        publication.track.kind == rtc.TrackKind.KIND_VIDEO and
                        publication.subscribed):
                        logger.info(f"📹 Found subscribed video track from {participant.identity}")
                        self._setup_video_stream(publication.track)
                        break
            
            # Listen for new tracks
            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant
            ):
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
            logger.error(f"❌ Error in on_enter: {e}")
    
    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """Attach video frame to user's message"""
        logger.info("💬 on_user_turn_completed called")
        
        if self._latest_frame:
            logger.info("📸 Attaching video frame to message")
            
            # Create image content from frame
            image_content = ImageContent(image=self._latest_frame)
            new_message.content.append(image_content)
            
            # Clear frame after use
            self._latest_frame = None
            logger.info("✅ Frame attached successfully")
        else:
            logger.warning("⚠️ No video frame available")
    
    def _setup_video_stream(self, track: rtc.Track) -> None:
        """Setup video stream from track"""
        try:
            # Cleanup existing stream
            self._cleanup_video_stream()
            
            # Create new video stream
            self._video_stream = rtc.VideoStream(track)
            logger.info("📹 Video stream created")
            
            # Start frame capture task
            self._frame_task = asyncio.create_task(self._capture_frames())
            
        except Exception as e:
            logger.error(f"❌ Error setting up video stream: {e}")
    
    def _cleanup_video_stream(self) -> None:
        """Cleanup video stream and tasks"""
        if self._frame_task and not self._frame_task.done():
            self._frame_task.cancel()
            self._frame_task = None
        
        if self._video_stream:
            try:
                self._video_stream.close()
            except:
                pass
            self._video_stream = None
    
    async def _capture_frames(self) -> None:
        """Capture frames from video stream"""
        frame_count = 0
        
        try:
            logger.info("🎥 Starting frame capture")
            
            async for event in self._video_stream:
                if hasattr(event, 'frame'):
                    self._latest_frame = event.frame
                    frame_count += 1
                    
                    # Log progress every 30 frames (~1 second)
                    if frame_count % 30 == 0:
                        logger.info(f"📸 Captured {frame_count} frames")
                        
        except asyncio.CancelledError:
            logger.info(f"🛑 Frame capture stopped after {frame_count} frames")
        except Exception as e:
            logger.error(f"❌ Error capturing frames: {e}")


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the agent"""
    logger.info("=" * 50)
    logger.info("🚀 Vision Agent Starting")
    logger.info(f"📍 Room: {ctx.room.name if ctx.room else 'Unknown'}")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
    logger.info("=" * 50)
    
    # Create and start agent session
    agent = VisionAgent()
    session = AgentSession()
    
    # Start the session - it will manage itself
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("✅ Agent session started successfully")
    # Session manages its own lifecycle - no need to wait


def main():
    """Main entry point"""
    logger.info("🏁 Starting Vision Agent Worker")
    
    # Run the agent
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint
    ))


if __name__ == "__main__":
    main()
