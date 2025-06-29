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
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("vision-agent-ollama")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")

class VisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
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
                You are an assistant with vision capabilities.
                Be very careful and accurate when describing what you see.
                If someone tells you that you're wrong about what you see, reconsider and look again.
                Common applications include YouTube, web browsers, IDEs, and other software.
                Never insist on something if the user corrects you.
            """,
            stt=deepgram.STT(),
            llm=ollama_llm,
            tts=openai.TTS(),  # OpenAI TTS verwenden
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
                self._create_video_stream(track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            logger.info(f"Adding frame to message: {self._latest_frame.width}x{self._latest_frame.height}")
            new_message.content.append(ImageContent(image=self._latest_frame))
            # Nicht löschen, damit wir kontinuierlich Frames haben
            # self._latest_frame = None
        else:
            logger.warning("No frame available when user asked question")
    
    # Helper method to buffer the latest video frame from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # VideoStream hat keine close() Methode - wir setzen es einfach auf None
        if self._video_stream is not None:
            self._video_stream = None
        
        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                frame_count += 1
                # Store the latest frame for use later
                self._latest_frame = event.frame
                
                # Log every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Received {frame_count} frames, latest: {event.frame.width}x{event.frame.height}")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    logger.info(f"Using Ollama at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
    
    session = AgentSession()
    await session.start(
        agent=VisionAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
