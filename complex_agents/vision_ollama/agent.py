#!/usr/bin/env python3
"""
Vision Agent für LiveKit Multi-Agent System
Kompatibel mit LiveKit Agents 1.0.23
Basiert auf dem funktionierenden Beispiel-Code
"""

import asyncio
import logging
import os
from typing import Optional
from livekit import rtc
from livekit.agents import (
    JobContext, 
    WorkerOptions,
    cli,
    get_job_context
)
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
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        
        super().__init__(
            instructions="""Du bist ein KI-Assistent mit Bildschirm-Sichtfähigkeiten für Python-Code-Analyse.
            
Du KANNST den Bildschirm des Nutzers sehen, wenn er seinen Bildschirm teilt!

DEINE AUFGABE: Finde Tippfehler in Python-Code auf dem geteilten Bildschirm.

WENN DER NUTZER NACH DEM BILDSCHIRM FRAGT:
- Sage: "Ja, ich kann Ihren Bildschirm sehen. Zeigen Sie mir bitte den Python-Code."

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

WICHTIG:
- Sage NIEMALS "Ich kann den Bildschirm nicht sehen" - Du KANNST ihn sehen!
- Wenn kein Code sichtbar → "Bitte zeigen Sie mir den Python-Code auf Ihrem Bildschirm"
- Wenn Code sichtbar → Analysiere und finde den Fehler""",
            
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
        
        logger.info(f"✅ VisionAgent initialized")
    
    async def on_enter(self):
        """Called when agent enters the room"""
        logger.info("🎯 Agent on_enter called")
        
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
                logger.info(f"📹 Found {len(video_tracks)} video track(s)")
                self._create_video_stream(video_tracks[0])
        
        # Watch for new video tracks
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info("📹 New video track subscribed")
                self._create_video_stream(track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Add the latest video frame to the new message"""
        logger.info("💬 on_user_turn_completed called")
        logger.info(f"🖼️ Latest frame available: {self._latest_frame is not None}")
        
        if self._latest_frame:
            logger.info("📸 Attaching video frame to message")
            try:
                # Ensure content is a list
                if not hasattr(new_message, 'content'):
                    new_message.content = []
                elif not isinstance(new_message.content, list):
                    new_message.content = []
                
                # Append the frame
                new_message.content.append(ImageContent(image=self._latest_frame))
                logger.info(f"✅ Frame attached! Content items: {len(new_message.content)}")
                
                # Keep frame for debugging (don't clear immediately)
                # self._latest_frame = None
                
            except Exception as e:
                logger.error(f"❌ Error attaching frame: {e}", exc_info=True)
        else:
            logger.warning("⚠️ No video frame available")
    
    def _create_video_stream(self, track: rtc.Track):
        """Helper method to buffer the latest video frame"""
        logger.info("🎥 Creating video stream")
        
        # Close any existing stream
        if self._video_stream is not None:
            self._video_stream.close()
        
        # Create a new stream
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                # Store the latest frame
                self._latest_frame = event.frame
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"📸 Captured {frame_count} frames")
                
                if frame_count == 1:
                    logger.info("🎉 First frame captured!")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


async def entrypoint(ctx: JobContext):
    """Main entrypoint"""
    logger.info("="*50)
    logger.info("🚀 Vision Agent Starting")
    logger.info(f"📍 Room: {ctx.room.name if ctx.room else 'Unknown'}")
    logger.info("="*50)
    
    # Connect to room
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Create session with agent
    session = AgentSession()
    await session.start(
        agent=VisionAgent(),
        room=ctx.room
    )
    
    logger.info("✅ Vision Agent session started")


async def request_handler(ctx: JobContext):
    """Request handler"""
    room_name = ctx.room.name if ctx.room else "unknown"
    logger.info(f"📨 Job request received for room: {room_name}")
    
    # Accept vision rooms
    if room_name.startswith("vision_room"):
        logger.info(f"✅ Accepting vision room: {room_name}")
        await ctx.accept()
    else:
        logger.info(f"❌ Rejecting non-vision room: {room_name}")
        await ctx.reject()


async def entrypoint_full(ctx: JobContext):
    """Full entrypoint with complete session management"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🏁 Starting Vision Agent Session: {session_id}")
    logger.info(f"📍 Room: {room_name}")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
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
        
        # 4. Create agent
        agent = VisionAgent()
        logger.info(f"✅ [{session_id}] Vision Agent created")
        
        # 5. Check for video tracks
        video_track_found = False
        for track_pub in participant.track_publications.values():
            if track_pub.kind == rtc.TrackKind.KIND_VIDEO and track_pub.track is not None:
                logger.info(f"📹 [{session_id}] Found existing video track")
                video_track_found = True
                break
        
        if not video_track_found:
            logger.info(f"⚠️ [{session_id}] No video track found initially")
        
        # 6. Create session
        session = AgentSession()
        
        # 7. Setup event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Response: {response_preview}...")
        
        # 8. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Wait for stabilization
        await asyncio.sleep(2.0)
        
        # 9. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        greeting_text = """Hallo! Ich bin Ihr Python Code-Assistent mit Bildschirm-Sichtfähigkeiten.
        
Ich kann Tippfehler in Python-Code für Sie finden. Bitte teilen Sie Ihren Bildschirm und zeigen Sie mir den Code.

Der Code den Sie mir zeigen hat einen Tippfehler - ich werde ihn finden!"""
        
        # Send greeting with retry
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


# Verwende den vollständigen entrypoint
entrypoint = entrypoint_full


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler
    ))
