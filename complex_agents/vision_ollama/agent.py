# LiveKit Agents - Vision Agent (Production Ready) - OLLAMA FIXED
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
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage
from livekit.plugins import openai, silero
from PIL import Image
import io

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
    """Vision Assistant für Code-Analyse"""
    
    def __init__(self) -> None:
        # SUPER WICHTIG: Die Instructions müssen EXTREM deutlich und KURZ sein
        super().__init__(instructions="""SYSTEM: Du bist ein deutscher Code-Analyse-Experte. ANTWORTE IMMER AUF DEUTSCH!

WENN DU EIN BILD MIT CODE SIEHST:
1. FINDE DEN FEHLER
2. NENNE DIE ZEILENNUMMER
3. ERKLÄRE DAS PROBLEM
4. GEBE DIE LÖSUNG
5. FERTIG! NICHT MEHR SAGEN!

ANTWORT-FORMAT (MAXIMAL 3 SÄTZE):
"Ich sehe den Fehler in Zeile [X]: [Problem].
Der korrekte Code lautet: [Lösung].
Das behebt den Syntax-Fehler."

BEISPIEL:
"Ich sehe den Fehler in Zeile 15: 'trom' ist falsch geschrieben.
Der korrekte Code lautet: 'from livekit.plugins import openai, silero'.
Das behebt den Import-Fehler."

VERBOTEN:
- Lange Erklärungen
- Alternative Lösungswege vorschlagen
- Über Tools oder andere Methoden reden
- Mehr als 3 Sätze

REGEL: KURZ UND PRÄZISE!""")
        
        # Store frame directly in agent like original
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        logger.info(f"✅ Code-Vision-Assistant initialized for worker {WORKER_ID}")
    
    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Code Vision Agent on_enter called")
        
        # Get room from job context
        from livekit.agents import get_job_context
        room = get_job_context().room
        
        # Find video tracks from remote participants
        if room.remote_participants:
            for participant in room.remote_participants.values():
                for publication in participant.track_publications.values():
                    if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                        logger.info(f"📹 Found video track from {participant.identity}")
                        self._create_video_stream(publication.track)
                        break
        
        # Watch for new video tracks
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 New video track subscribed from {participant.identity}")
                self._create_video_stream(track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called after user finishes speaking - add vision frame to message"""
        if self._latest_frame:
            logger.info(f"📸 Adding frame to user message")
            
            # Convert content to list if it's a string
            if isinstance(new_message.content, str):
                user_text = new_message.content
                new_message.content = [new_message.content]
                
                # WICHTIG: Füge explizite Anweisung hinzu wenn es um Code geht
                if "code" in user_text.lower() or "fehler" in user_text.lower() or "problem" in user_text.lower():
                    new_message.content[0] = f"{user_text}\n\nBITTE ANALYSIERE DEN CODE IM BILD UND FINDE DEN SYNTAX-FEHLER! ANTWORTE AUF DEUTSCH!"
            
            # Log user message for debugging
            logger.info(f"🎯 User message: {new_message.content[0] if new_message.content else 'No content'}")
            
            # Append frame like in original agent
            new_message.content.append(ImageContent(image=self._latest_frame))
            
            # Don't clear the frame - keep it for next time
            # self._latest_frame = None  # REMOVED
        else:
            logger.warning("⚠️ No frame available for vision analysis")
    
    def _create_video_stream(self, track: rtc.Track):
        """Create video stream to capture frames"""
        # Close existing stream
        if self._video_stream is not None:
            self._video_stream.close()
        
        # Create new stream
        self._video_stream = rtc.VideoStream(track)
        logger.info("✅ Video stream created")
        
        async def read_stream():
            """Read frames from video stream"""
            frame_count = 0
            try:
                async for event in self._video_stream:
                    frame_count += 1
                    
                    # Store frame directly without resizing for now
                    self._latest_frame = event.frame
                    
                    # Log every 30 frames
                    if frame_count % 30 == 0:
                        logger.info(f"📸 Captured {frame_count} frames, latest: {event.frame.width}x{event.frame.height}")
                        # Log frame format for debugging
                        if hasattr(event.frame, 'type'):
                            logger.debug(f"📹 Frame format: {event.frame.type}")
                        
            except Exception as e:
                logger.error(f"❌ Error reading video stream: {e}")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


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
        
        # 4. Configure LLM - NUR llava-llama3 ist vision-fähig!
        ollama_host = os.getenv("OLLAMA_HOST", "http://172.16.0.136:11434")
        
        # WICHTIG: NUR llava-llama3 kann Bilder analysieren!
        ollama_model = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")
        
        # Create LLM with vision model
        llm = openai.LLM.with_ollama(
            model=ollama_model,
            base_url=f"{ollama_host}/v1",
            temperature=0.0,  # Deterministisch für konsistente Antworten
        )
        
        logger.info(f"🤖 [{session_id}] Using Ollama Vision: {ollama_model} at {ollama_host}")
        
        # 5. Create session with German-focused configuration
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
                language="de"  # Deutsch für STT
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"  # oder "alloy" für männliche Stimme
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )
        
        # 6. Create and start agent
        agent = VisionAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Warte auf Audio-Stabilisierung
        await asyncio.sleep(2.0)
        
        # Event handlers
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
            
            # Check if response is in English (hallucination detection)
            if hasattr(event, '__str__'):
                response_str = str(event)
                english_indicators = ["is a feature", "you will need", "can be useful", "this can be"]
                if any(indicator in response_str.lower() for indicator in english_indicators):
                    logger.error(f"⚠️ ENGLISH RESPONSE DETECTED! Model is ignoring German instructions!")
        
        # 8. Initial greeting - DEUTSCH!
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = """Hallo! Ich bin Ihr Code-Analyse-Spezialist.

Zeigen Sie mir Ihren Code über die Kamera, und ich helfe Ihnen dabei, Syntax-Fehler, Bugs oder Verbesserungsmöglichkeiten zu finden.

Welches Code-Problem kann ich für Sie lösen?"""
            
            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_FRAME
            
            # Retry-Mechanismus
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


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
