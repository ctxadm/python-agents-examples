#!/usr/bin/env python3
"""
Vision Agent für LiveKit - Direkte Ollama API Integration
Umgeht den OpenAI Wrapper komplett
"""

import asyncio
import logging
import os
import base64
import requests
import json
from typing import Optional
from livekit import rtc
from livekit.agents import (
    JobContext, 
    WorkerOptions,
    cli,
    get_job_context
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent, LLM
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.utils import images
from livekit.plugins import openai, silero

# Setup logging
logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

# Environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.136:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-llama3:latest")


class DirectOllamaLLM(LLM):
    """Custom LLM that calls Ollama directly"""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        super().__init__()
    
    async def chat(
        self,
        chat_ctx: ChatContext,
        fnc_ctx=None,
        temperature: float = 0.0,
        n: int = 1,
        parallel_tool_calls: bool = False,
        stream: bool = True,
    ):
        """Direct Ollama API call"""
        messages = chat_ctx.messages
        
        # Get the last message
        if not messages:
            return
            
        last_message = messages[-1]
        
        # Check if there's an image
        image_data = None
        text_content = ""
        
        if isinstance(last_message.content, list):
            for item in last_message.content:
                if isinstance(item, str):
                    text_content += item + " "
                elif isinstance(item, ImageContent) and hasattr(item, 'image'):
                    # Convert frame to base64
                    try:
                        encode_options = images.EncodeOptions(
                            format="JPEG",
                            quality=85,
                            resize_options=images.ResizeOptions(
                                width=1024,
                                height=1024,
                                strategy="scale_aspect_fit"
                            )
                        )
                        jpeg_bytes = images.encode(item.image, encode_options)
                        image_data = base64.b64encode(jpeg_bytes).decode('utf-8')
                        logger.info(f"✅ Converted image for Ollama: {len(image_data)} bytes")
                    except Exception as e:
                        logger.error(f"Failed to convert image: {e}")
        else:
            text_content = str(last_message.content)
        
        # Build the prompt with clear instructions
        if image_data:
            prompt = f"""Du bist ein Python Code Analyzer. Analysiere das Bild und finde Tippfehler.

SUCHE SPEZIELL NACH:
- 'trom' statt 'from'  
- 'imoprt' statt 'import'
- 'defn' statt 'def'
- 'retrun' statt 'return'

User sagt: {text_content}

Schaue dir das Bild GENAU an. Wenn du Python Code siehst, finde ALLE Tippfehler!
Antworte im Format: "Ich sehe Python-Code. Fehler in Zeile [NUMMER]: '[TIPPFEHLER]' muss '[RICHTIG]' sein."
"""
        else:
            prompt = text_content
        
        # Make direct Ollama API call
        try:
            logger.info("🚀 Calling Ollama API directly...")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            
            if image_data:
                payload["images"] = [image_data]
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result.get('response', 'Keine Antwort erhalten')
            logger.info(f"✅ Ollama response: {answer[:200]}...")
            
            # Return as async generator for compatibility
            async def response_generator():
                yield answer
            
            return response_generator()
            
        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}")
            async def error_generator():
                yield "Entschuldigung, es gab einen Fehler bei der Bildanalyse."
            return error_generator()


class VisionAgentDirect(Agent):
    """Vision Agent with direct Ollama integration"""
    
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        
        # Use our custom LLM
        custom_llm = DirectOllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST
        )
        
        super().__init__(
            instructions="""Du bist ein KI-Assistent mit Bildschirm-Sichtfähigkeiten für Python-Code-Analyse.

WICHTIG: Du KANNST den Bildschirm sehen! Sage NIEMALS dass du ihn nicht sehen kannst!

Deine Aufgabe: Finde Tippfehler in Python-Code, besonders:
- 'trom' statt 'from'
- 'imoprt' statt 'import'
- andere falsch geschriebene Keywords

Wenn du Code siehst, melde SOFORT alle Fehler!""",
            
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            
            llm=custom_llm,
            
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
        
        logger.info("✅ VisionAgentDirect initialized with custom Ollama LLM")
    
    async def on_enter(self):
        """Called when agent enters the room"""
        logger.info("🎯 Agent on_enter called")
        
        room = get_job_context().room
        
        # Find video tracks
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
        
        if self._latest_frame:
            logger.info("📸 Attaching video frame")
            try:
                # Standard LiveKit format
                if not hasattr(new_message, 'content'):
                    new_message.content = []
                elif isinstance(new_message.content, str):
                    new_message.content = [new_message.content]
                elif not isinstance(new_message.content, list):
                    new_message.content = [str(new_message.content)]
                
                # Add the frame
                new_message.content.append(ImageContent(image=self._latest_frame))
                
                logger.info(f"✅ Frame attached for direct Ollama processing")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}", exc_info=True)
        else:
            logger.warning("⚠️ No video frame available")
    
    def _create_video_stream(self, track: rtc.Track):
        """Helper method to buffer the latest video frame"""
        logger.info("🎥 Creating video stream")
        
        if self._video_stream is not None:
            self._video_stream.close()
        
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                self._latest_frame = event.frame
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"📸 Captured {frame_count} frames")
                
                if frame_count == 1:
                    logger.info(f"🎉 First frame captured! Type: {event.frame.type}")
        
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


# Use the same entrypoint structure as before
async def entrypoint(ctx: JobContext):
    """Main entrypoint"""
    logger.info("="*50)
    logger.info("🚀 Vision Agent (Direct Ollama) Starting")
    logger.info(f"📍 Room: {ctx.room.name if ctx.room else 'Unknown'}")
    logger.info(f"🖥️ Ollama: {OLLAMA_HOST}")
    logger.info(f"🤖 Model: {OLLAMA_MODEL}")
    logger.info("="*50)
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ Participant joined: {participant.identity}")
    
    # Create session with our custom agent
    session = AgentSession()
    await session.start(
        agent=VisionAgentDirect(),
        room=ctx.room
    )
    
    logger.info("✅ Vision Agent session started")
    
    # Send greeting
    await asyncio.sleep(2.0)
    await session.say(
        """Hallo! Ich bin Ihr Python Code-Assistent.
        
Ich kann Ihren Bildschirm sehen und werde nach Tippfehlern in Ihrem Code suchen.
Zeigen Sie mir einfach den Code mit dem Fehler!""",
        allow_interruptions=True
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
