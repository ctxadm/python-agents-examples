# LiveKit Agents - Vision Agent (LiveKit 1.1.0+ Compatible) - FIXED
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "vision-1")
WORKER_ID = os.getenv("WORKER_ID", "vision-worker-1")


class VisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        
        super().__init__(
            instructions="""Du bist ein Code-Analyse-Spezialist mit Vision-Fähigkeiten.
                Du kannst sehen was der User dir über die Kamera zeigt.
                Verwende keine unaussprechbaren Zeichen.
                
                WICHTIGE REGELN:
                - Antworte IMMER auf Deutsch
                - Wenn du Code siehst, finde Syntax-Fehler
                - Nenne immer die Zeilennummer des Fehlers
                - Gib eine konkrete Korrektur
                - Halte Antworten kurz (max 2 Sätze)""",
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            llm=openai.LLM.with_ollama(
                model=os.getenv("OLLAMA_MODEL", "llava-llama3:latest"),
                base_url=f"{os.getenv('OLLAMA_HOST', 'http://172.16.0.136:11434')}/v1",
                temperature=0.0
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
            vad=silero.VAD.load()
        )
        logger.info(f"✅ Vision Agent initialized")
    
    async def on_enter(self):
        """Called when agent enters the room"""
        logger.info("🚪 on_enter called - Agent entering room")
        room = get_job_context().room
        
        # Find the first video track (if any) from the remote participant
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            logger.info(f"👤 Found remote participant: {remote_participant.identity}")
            
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            
            if video_tracks:
                logger.info(f"📹 Found {len(video_tracks)} video track(s)")
                self._create_video_stream(video_tracks[0])
            else:
                logger.warning("⚠️ No video tracks found yet")
        else:
            logger.warning("⚠️ No remote participants yet")
        
        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 New video track subscribed from {participant.identity}")
                self._create_video_stream(track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Add the latest video frame to the user's message"""
        logger.info("📍 on_user_turn_completed called!")
        
        if self._latest_frame:
            logger.info("📸 Adding frame to message")
            new_message.content.append(ImageContent(image=self._latest_frame))
            # Clear frame after use (wie im Original)
            self._latest_frame = None
            logger.info("✅ Frame attached to message")
        else:
            logger.warning("⚠️ No frame available")
    
    def _create_video_stream(self, track: rtc.Track):
        """Create video stream to capture frames"""
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            try:
                self._video_stream.close()
            except:
                pass  # Ignore if close doesn't exist
        
        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)
        logger.info("✅ Video stream created")
        
        frame_count = 0
        
        async def read_stream():
            nonlocal frame_count
            try:
                async for event in self._video_stream:
                    # Store the latest frame for use later
                    self._latest_frame = event.frame
                    frame_count += 1
                    
                    # Log every 30 frames (about once per second)
                    if frame_count % 30 == 0:
                        logger.info(f"📸 Captured {frame_count} frames")
            except Exception as e:
                logger.error(f"❌ Error reading video stream: {e}")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


# Request handler entfernt - nicht nötig für einfache Setups


async def entrypoint(ctx: JobContext):
    """Main entrypoint - matches working example"""
    logger.info("🏁 Starting Vision Agent (LiveKit 1.1.0+)")
    logger.info(f"🏠 Room: {ctx.room.name if ctx.room else 'unknown'}")
    
    # Create session and start agent
    session = AgentSession()
    await session.start(
        agent=VisionAgent(),
        room=ctx.room
    )
    
    logger.info("✅ Agent session started")
    # WICHTIG: KEIN await danach - session managed sich selbst!


if __name__ == "__main__":
    logger.info(f"🏁 Starting Vision Worker: {WORKER_ID}")
    
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint
        # request_handler removed - wie im funktionierenden Beispiel
    ))
