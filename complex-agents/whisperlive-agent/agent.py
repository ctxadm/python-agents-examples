import os
import logging
import asyncio
import websockets
import json
import numpy as np
from typing import Optional

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.stt import STT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData
from livekit.plugins import openai, silero
from livekit import rtc

# Verwende lokale Services-Paket für Piper TTS
from .services.local_services import RemotePiperTTS

logger = logging.getLogger("whisperlive-agent")

class WhisperLiveKitSTT(STT):
    """WebSocket-based STT using WhisperLiveKit"""
    
    def __init__(
        self, 
        ws_url: str = "ws://172.16.0.146:9090",
        model: str = "base",
        language: str = "de"
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True
            )
        )
        self.ws_url = ws_url
        self.model = model
        self.language = language
        self._websocket = None
        self._audio_buffer = []
        self._running = False
        
    async def _connect(self):
        """Establish WebSocket connection"""
        if not self._websocket:
            self._websocket = await websockets.connect(self.ws_url)
            # Send configuration
            await self._websocket.send(json.dumps({
                "type": "config",
                "config": {
                    "model": self.model,
                    "language": self.language,
                    "task": "transcribe",
                    "vad_enabled": True,
                    "realtime": True
                }
            }))
            logger.info(f"Connected to WhisperLiveKit at {self.ws_url}")
    
    async def _process_audio_stream(self):
        """Process audio and receive transcriptions"""
        try:
            async for message in self._websocket:
                data = json.loads(message)
                
                if data.get("type") == "transcript":
                    yield SpeechEvent(
                        type=SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[SpeechData(
                            text=data.get("text", ""),
                            confidence=data.get("confidence", 0.0)
                        )]
                    )
                elif data.get("type") == "final":
                    yield SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechData(
                            text=data.get("text", ""),
                            confidence=data.get("confidence", 1.0)
                        )]
                    )
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def recognize(
        self,
        buffer: rtc.AudioFrame,
        *,
        language: Optional[str] = None,
        final: bool = True,
    ) -> SpeechEvent:
        """Process single audio frame"""
        await self._connect()
        
        # Convert to bytes and send
        audio_bytes = buffer.data.tobytes()
        await self._websocket.send(audio_bytes)
        
        # Get response
        try:
            message = await asyncio.wait_for(
                self._websocket.recv(), 
                timeout=5.0
            )
            data = json.loads(message)
            
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT if final else SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(
                    text=data.get("text", ""),
                    confidence=data.get("confidence", 1.0)
                )]
            )
        except asyncio.TimeoutError:
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="")]
            )
    
    async def aclose(self):
        """Close WebSocket connection"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

async def entrypoint(ctx: JobContext):
    """Agent using WhisperLiveKit for STT"""
    await ctx.connect(auto_subscribe=True)
    
    logger.info("Starting WhisperLiveKit Agent")
    
    # Erstelle custom STT
    stt = WhisperLiveKitSTT(
        ws_url=f"ws://{os.getenv('VISION_AI_SERVER_IP', '172.16.0.146')}:9090",
        model="base",
        language="de"
    )
    
    # Verwende TTS aus local_services
    tts = RemotePiperTTS(
        voice="de_DE-thorsten-medium",
        base_url=f"http://{os.getenv('VISION_AI_SERVER_IP', '172.16.0.146')}:8001"
    )
    
    # LLM via OpenAI-Plugin
    llm = openai.LLM(
        model="llama3.2:latest",
        base_url=f"http://{os.getenv('RAG_AI_SERVER_IP', '172.16.0.136')}:11434/v1",
        api_key="ollama"
    )
    
    # Erzeuge Agent
    agent = Agent(
        instructions="Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load()
    )
    
    # Starte Session
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)
    
    # Begrüßung
    await session.say("Hallo! Wie kann ich dir helfen?")
    
    logger.info("WhisperLiveKit Agent running!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
