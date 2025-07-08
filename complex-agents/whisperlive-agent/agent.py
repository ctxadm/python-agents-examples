import os
import logging
import asyncio
import json
import wave
import io
import numpy as np
from typing import Optional, AsyncIterator
import websockets
from collections import deque

from livekit import rtc, agents
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli,
    stt
)
from livekit.agents.stt import (
    STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, STT
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
        language: str = "de",
        chunk_duration_ms: int = 1000
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
        self.chunk_duration_ms = chunk_duration_ms
        self._websocket = None
        self._receive_task = None
        self._send_task = None
        self._audio_queue = asyncio.Queue()
        self._transcript_queue = asyncio.Queue()
        self._running = False
        logger.info(f"Initialized WhisperLiveKit WebSocket STT with {ws_url}")
    
    async def _connect(self):
        """Connect to WhisperLiveKit WebSocket"""
        if self._websocket is None or self._websocket.closed:
            try:
                self._websocket = await websockets.connect(self.ws_url)
                logger.info("Connected to WhisperLiveKit WebSocket")
                self._running = True
                
                # Start receive task
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._send_task = asyncio.create_task(self._send_loop())
                
                # Send initial configuration if needed
                # Some WhisperLiveKit versions might expect config
                config = {
                    "model": self.model,
                    "language": self.language,
                    "chunk_duration": self.chunk_duration_ms
                }
                # Try sending config, ignore if not supported
                try:
                    await self._websocket.send(json.dumps(config))
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to connect to WhisperLiveKit: {e}")
                raise
    
    async def _receive_loop(self):
        """Receive transcriptions from WhisperLiveKit"""
        try:
            while self._running and self._websocket and not self._websocket.closed:
                try:
                    message = await asyncio.wait_for(self._websocket.recv(), timeout=0.1)
                    data = json.loads(message)
                    
                    # WhisperLiveKit returns various message types
                    if "text" in data or "transcript" in data:
                        text = data.get("text") or data.get("transcript", "")
                        is_final = data.get("final", True)
                        
                        await self._transcript_queue.put({
                            "text": text,
                            "is_final": is_final
                        })
                    elif "lines" in data:
                        # Handle multi-line format
                        for line in data["lines"]:
                            if "text" in line:
                                await self._transcript_queue.put({
                                    "text": line["text"],
                                    "is_final": True
                                })
                                
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WhisperLiveKit WebSocket closed")
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")
                    
        except Exception as e:
            logger.error(f"Receive loop crashed: {e}")
        finally:
            self._running = False
    
    async def _send_loop(self):
        """Send audio to WhisperLiveKit"""
        try:
            while self._running and self._websocket and not self._websocket.closed:
                try:
                    audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                    
                    # Send as binary data (WhisperLiveKit expects webm/opus or raw audio)
                    await self._websocket.send(audio_data)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    
        except Exception as e:
            logger.error(f"Send loop crashed: {e}")
        finally:
            self._running = False
    
    async def recognize(
        self,
        buffer: rtc.AudioFrame,
        *,
        language: Optional[str] = None,
        final: bool = True,
        conn_options: Optional[dict] = None
    ) -> SpeechEvent:
        """Process audio via WhisperLiveKit WebSocket"""
        
        # Ensure connected
        if not self._websocket or self._websocket.closed:
            await self._connect()
        
        try:
            # Convert AudioFrame to WAV bytes
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                
                # Get audio data
                if hasattr(buffer, 'data'):
                    if isinstance(buffer.data, np.ndarray):
                        audio_data = buffer.data.tobytes()
                    else:
                        audio_data = bytes(buffer.data)
                else:
                    audio_data = bytes(buffer)
                    
                wav_file.writeframes(audio_data)
            
            # Queue audio for sending
            wav_buffer.seek(0)
            await self._audio_queue.put(wav_buffer.getvalue())
            
            # Try to get transcript with short timeout
            try:
                transcript_data = await asyncio.wait_for(
                    self._transcript_queue.get(), 
                    timeout=0.05  # 50ms timeout
                )
                
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT if transcript_data["is_final"] 
                          else SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[SpeechData(
                        text=transcript_data["text"],
                        language=language or self.language,
                        confidence=1.0
                    )]
                )
            except asyncio.TimeoutError:
                # No transcript available yet
                return SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
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
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
        if self._send_task:
            self._send_task.cancel()
            
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            
        logger.info("WhisperLiveKit WebSocket closed")


async def entrypoint(ctx: JobContext):
    """WhisperLiveKit Agent Entry Point"""
    await ctx.connect()
    
    logger.info("=== Starting WhisperLiveKit WebSocket Agent ===")
    logger.info(f"Room: {ctx.room.name}")
    
    try:
        # Initialize STT with WebSocket
        whisper_url = os.getenv('WHISPERLIVEKIT_WS_URL', 'ws://172.16.0.146:9090/asr')
        logger.info(f"Connecting to WhisperLiveKit WebSocket at {whisper_url}")
        
        stt_service = WhisperLiveKitWebSocketSTT(
            ws_url=whisper_url,
            model="base",
            language="de",
            chunk_duration_ms=1000
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
            stt=stt_service,
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
