import os
import logging
import asyncio
import json
import wave
import io
import numpy as np
from typing import Optional
import websockets

from livekit import rtc, agents
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli,
    stt
)
from livekit.agents.stt import (
    STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, STT,
    StreamAdapter
)
from livekit.plugins import silero, openai

# Import local services
from local_services import RemotePiperTTS

logger = logging.getLogger("whisperlive-agent")


class WhisperLiveKitWebSocketSTT(STT):
    """WebSocket-based STT using WhisperLiveKit"""
    
    def __init__(
        self, 
        ws_url: str = "ws://172.16.0.146:9090/asr",
        model: str = "base",
        language: str = "de"
    ):
        # WICHTIG: streaming=False setzen
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,  # Wir nutzen recognize() statt stream()
                interim_results=False
            )
        )
        self.ws_url = ws_url
        self.model = model
        self.language = language
        self._websocket = None
        logger.info(f"Initialized WhisperLiveKit WebSocket STT with {ws_url}")
    
    async def _ensure_websocket(self):
        """Ensure WebSocket is connected"""
        if self._websocket is None or self._websocket.closed:
            try:
                self._websocket = await websockets.connect(self.ws_url)
                logger.info("Connected to WhisperLiveKit WebSocket")
            except Exception as e:
                logger.error(f"Failed to connect to WhisperLiveKit: {e}")
                self._websocket = None
    
    async def recognize(
        self,
        buffer: rtc.AudioFrame,
        *,
        language: Optional[str] = None,
        final: bool = True,
        conn_options: Optional[dict] = None
    ) -> SpeechEvent:
        """Process audio via WhisperLiveKit WebSocket"""
        
        try:
            # Get raw PCM audio data
            if hasattr(buffer, 'data'):
                if isinstance(buffer.data, np.ndarray):
                    audio_data = buffer.data.tobytes()
                else:
                    audio_data = bytes(buffer.data)
            else:
                audio_data = bytes(buffer)
            
            # WhisperLiveKit expects WebM/Opus from browser OR raw PCM
            # Let's send raw PCM data directly
            logger.debug(f"Sending {len(audio_data)} bytes of raw PCM")
            
            # Ensure WebSocket connection
            await self._ensure_websocket()
            
            if self._websocket:
                # Send raw PCM audio data
                await self._websocket.send(audio_data)
                logger.debug(f"Sent {len(audio_data)} bytes to WhisperLiveKit")
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(self._websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    
                    # Extract text from response
                    text = ""
                    if "text" in data:
                        text = data["text"]
                    elif "transcript" in data:
                        text = data["transcript"]
                    elif "lines" in data and len(data["lines"]) > 0:
                        # Join all lines
                        text = " ".join(line.get("text", "") for line in data["lines"])
                    
                    logger.info(f"Received transcript: {text}")
                    
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechData(
                            text=text,
                            language=language or self.language,
                            confidence=1.0
                        )]
                    )
                    
                except asyncio.TimeoutError:
                    logger.debug("No response from WhisperLiveKit within timeout")
                    
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(
                    text="",
                    language=language or self.language,
                    confidence=0.0
                )]
            )
                    
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(
                    text="",
                    language=language or self.language,
                    confidence=0.0
                )]
            )
    
    async def _recognize_impl(self, buffer, *, language=None):
        """Required by base class"""
        return await self.recognize(buffer, language=language)
    
    async def aclose(self):
        """Close WebSocket connection"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None


async def entrypoint(ctx: JobContext):
    """WhisperLiveKit Agent Entry Point"""
    await ctx.connect()
    
    logger.info("=== Starting WhisperLiveKit WebSocket Agent ===")
    logger.info(f"Room: {ctx.room.name}")
    
    try:
        # Initialize STT with WebSocket
        whisper_url = os.getenv('WHISPERLIVEKIT_WS_URL', 'ws://172.16.0.146:9090/asr')
        logger.info(f"Connecting to WhisperLiveKit WebSocket at {whisper_url}")
        
        # Create base STT
        base_stt = WhisperLiveKitWebSocketSTT(
            ws_url=whisper_url,
            model="base",
            language="de"
        )
        
        # Wrap with StreamAdapter for streaming support
        stt_service = StreamAdapter(
            stt=base_stt,
            vad=silero.VAD.load()
        )
        
        # Initialize TTS
        piper_url = os.getenv('PIPER_URL', 'http://172.16.0.146:8001')
        logger.info(f"Connecting to Piper TTS at {piper_url}")
        
        tts_service = RemotePiperTTS(
            voice="de_DE-thorsten-medium",
            base_url=piper_url
        )
        
        # Initialize LLM
        ollama_url = os.getenv('OLLAMA_URL', 'http://172.16.0.146:11434')
        logger.info(f"Connecting to Ollama at {ollama_url}")
        
        llm_service = openai.LLM(
            model="llama3.2:latest",
            base_url=f"{ollama_url}/v1",
            api_key="ollama"
        )
        
        # Create agent
        agent = Agent(
            instructions="""Du bist ein hilfreicher KI-Assistent.
            Antworte kurz und präzise auf Deutsch.
            Sei freundlich und professionell.""",
            stt=stt_service,  # StreamAdapter wrapped STT
            llm=llm_service,
            tts=tts_service,
            vad=silero.VAD.load()
        )
        
        # Start session
        session = AgentSession()
        await session.start(agent=agent, room=ctx.room)
        
        logger.info("✓ WhisperLiveKit WebSocket Agent successfully started!")
        logger.info(f"✓ STT: WhisperLiveKit WebSocket on {whisper_url}")
        logger.info(f"✓ TTS: Piper on {piper_url}")
        logger.info(f"✓ LLM: Ollama on {ollama_url}")
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
