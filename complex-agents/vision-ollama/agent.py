## This is a basic example of how to use function calling.
## To test the function, you can ask the agent to print to the console!
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import function_tool, ImageContent, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero, cartesia

logger = logging.getLogger("vision-agent-ollama")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")

class VisionAgent(Agent):
    def __init__(self) -> None:
        # Frame-Buffer statt einzelnem Frame
        self._frame_buffer = []
        self._max_frames = 5  # Anzahl der zu speichernden Frames
        self._video_stream = None
        self._tasks = []
        
        # Ollama über OpenAI-kompatible API
        ollama_llm = openai.LLM(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=60.0  # 60 Sekunden Timeout für lokale Modelle
        )
        
        super().__init__(
            instructions="""
                You are an assistant communicating through voice with vision capabilities.
                You can see what the user is showing you through their camera or screen.
                Don't use any unpronouncable characters.
                Be accurate in describing what you see, whether it's from a camera or screen share.
            """,
            stt=deepgram.STT(),
            llm=ollama_llm,
            oder openai.TTS(),
            vad=silero.VAD.load()
        )
    
    async def on_enter(self):
        room = get_job_context().room
        
        # Debug: Log all participants and tracks
        logger.info(f"Room participants: {len(room.remote_participants)}")
        
        # Find the first video track (if any) from the remote participant
        if room.remote_participants:
            for participant_id, remote_participant in room.remote_participants.items():
                logger.info(f"Participant {participant_id}: {remote_participant.name or 'Unknown'}")
                
                # Log all track publications
                for track_id, publication in remote_participant.track_publications.items():
                    logger.info(f"  Track {track_id}: kind={publication.kind}, subscribed={publication.subscribed}")
                
                # Get video tracks
                video_tracks = [
                    publication.track
                    for publication in list(remote_participant.track_publications.values())
                    if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
                ]
                
                if video_tracks:
                    logger.info(f"Found {len(video_tracks)} video track(s), using first one")
                    self._create_video_stream(video_tracks[0])
                else:
                    logger.warning("No video tracks found at startup")
        else:
            logger.warning("No remote participants found at startup")
        
        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"Track subscribed event: kind={track.kind}, sid={track.sid}, participant={participant.name}")
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"New video track subscribed: {track.sid}")
                self._create_video_stream(track)
        
        # Watch for track updates (wichtig für Screen Share!)
        @room.on("track_muted")
        def on_track_muted(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"Track muted: {publication.sid}")
        
        @room.on("track_unmuted")
        def on_track_unmuted(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"Track unmuted: {publication.sid}")
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"Video track unmuted, recreating stream")
                self._create_video_stream(publication.track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add a frame from the buffer to the new message
        if self._frame_buffer:
            # Mittleres Frame aus dem Buffer nehmen (stabiler als das letzte)
            middle_idx = len(self._frame_buffer) // 2
            selected_frame = self._frame_buffer[middle_idx]
            
            logger.info(f"Using frame {middle_idx+1} of {len(self._frame_buffer)} buffered frames")
            logger.info(f"Frame resolution: {selected_frame.width}x{selected_frame.height}")
            
            new_message.content.append(ImageContent(image=selected_frame))
            
            # Buffer leeren nach Verwendung
            self._frame_buffer.clear()
        else:
            logger.warning("No frames in buffer when user turn completed")
            logger.info(f"Video stream active: {self._video_stream is not None}")
            logger.info(f"Active tasks: {len(self._tasks)}")
    
    # Helper method to buffer video frames from the user's track
    def _create_video_stream(self, track: rtc.Track):
        logger.info(f"Creating video stream for track: {track.sid}")
        
        # Clear frame buffer when switching tracks
        self._frame_buffer.clear()
        logger.info("Cleared frame buffer for new track")
        
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            logger.info("Closing existing video stream")
            self._video_stream.close()
        
        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            frame_count = 0
            logger.info(f"Starting to read frames from track {track.sid}")
            try:
                async for event in self._video_stream:
                    frame_count += 1
                    
                    # Frame zum Buffer hinzufügen
                    self._frame_buffer.append(event.frame)
                    
                    # Buffer-Größe begrenzen
                    if len(self._frame_buffer) > self._max_frames:
                        self._frame_buffer.pop(0)  # Ältestes Frame entfernen
                    
                    # Debug-Info alle 30 Frames
                    if frame_count % 30 == 0:
                        logger.info(f"Frame buffer size: {len(self._frame_buffer)}, total frames: {frame_count}")
                        logger.info(f"Latest frame: {event.frame.width}x{event.frame.height}")
                        
                    # Log bei Auflösungsänderung (wichtig für Screen Share Erkennung)
                    if frame_count == 1 or (frame_count > 1 and len(self._frame_buffer) > 1):
                        prev_frame = self._frame_buffer[-2] if len(self._frame_buffer) > 1 else None
                        if prev_frame and (prev_frame.width != event.frame.width or prev_frame.height != event.frame.height):
                            logger.info(f"Resolution changed: {prev_frame.width}x{prev_frame.height} -> {event.frame.width}x{event.frame.height}")
            except Exception as e:
                logger.error(f"Error reading video stream: {e}")
            finally:
                logger.info(f"Video stream ended after {frame_count} frames")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    logger.info(f"Using Ollama at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
    logger.info(f"Frame buffering enabled with max {5} frames")
    
    session = AgentSession()
    await session.start(
        agent=VisionAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
