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
            tts=cartesia.TTS(),  # oder openai.TTS() falls kein Cartesia Key
            vad=silero.VAD.load()
        )
    
    async def on_enter(self):
        room = get_job_context().room
        
        # Find the first video track (if any) from the remote participant
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])
        
        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"New video track subscribed: {track.sid}")
                self._create_video_stream(track)
    
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
    
    # Helper method to buffer video frames from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            self._video_stream.close()
        
        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                frame_count += 1
                
                # Frame zum Buffer hinzufügen
                self._frame_buffer.append(event.frame)
                
                # Buffer-Größe begrenzen
                if len(self._frame_buffer) > self._max_frames:
                    self._frame_buffer.pop(0)  # Ältestes Frame entfernen
                
                # Debug-Info alle 30 Frames
                if frame_count % 30 == 0:
                    logger.debug(f"Frame buffer size: {len(self._frame_buffer)}, total frames: {frame_count}")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
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
