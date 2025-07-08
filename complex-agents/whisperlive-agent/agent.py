import os
import logging
import asyncio
import json
import io
import wave
import numpy as np
from typing import Optional, AsyncIterator
import aiohttp

from livekit import rtc, agents
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli,
    stt, tts, llm
)
from livekit.agents.stt import (
    STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, STT
)
from livekit.plugins import silero, openai

# Import local services - stellen Sie sicher, dass local_services.py im gleichen Verzeichnis ist
from local_services import RemotePiperTTS

logger = logging.getLogger("whisperlive-agent")


class WhisperLiveKitSTT(STT):
    """HTTP-based STT using WhisperLiveKit REST API"""
    
    def __init__(
        self, 
        api_url: str = "http://172.16.0.146:9090",
        model: str = "base",
        language: str = "de"
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False
            )
        )
        self.api_url = api_url
        self.model = model
        self.language = language
        self._session = None
        logger.info(f"Initialized WhisperLiveKitSTT with {api_url}")
    
    async def _ensure_session(self):
        if not self._session:
            self._session = aiohttp.ClientSession()
    
    async def recognize(
        self,
        buffer: rtc.AudioFrame,
        *,
        language: Optional[str] = None,
        final: bool = True,
    ) -> SpeechEvent:
        """Process audio via WhisperLiveKit HTTP API"""
        await self._ensure_session()
        
        try:
            # Convert AudioFrame to WAV
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
            
            wav_buffer.seek(0)
            
            # Send to WhisperLiveKit API
            data = aiohttp.FormData()
            data.add_field('audio', wav_buffer, 
                          filename='audio.wav',
                          content_type='audio/wav')
            data.add_field('model', self.model)
            data.add_field('language', language or self.language)
            
            async with self._session.post(
                f"{self.api_url}/transcribe",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    text = result.get('text', '')
                    
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechData(
                            text=text,
                            confidence=result.get('confidence', 1.0)
                        )]
                    )
                else:
                    logger.error(f"WhisperLiveKit error: {response.status}")
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechData(text="")]
                    )
                    
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="")]
            )
    
    async def _recognize_impl(self, buffer, *, language=None):
        """Required by base class"""
        return await self.recognize(buffer, language=language)
    
    async def aclose(self):
        if self._session:
            await self._session.close()


async def entrypoint(ctx: JobContext):
    """WhisperLiveKit Agent Entry Point"""
    await ctx.connect()
    
    logger.info("=== Starting WhisperLiveKit Agent ===")
    logger.info(f"Room: {ctx.room.name}")
    # FIX: ctx.participant existiert nicht in dieser Version
    # logger.info(f"Participant: {ctx.participant}")
    
    try:
        # Initialize STT - verwende die korrekte IP für WhisperLiveKit
        whisper_url = os.getenv('WHISPERLIVEKIT_URL', 'http://172.16.0.146:9090')
        logger.info(f"Connecting to WhisperLiveKit at {whisper_url}")
        
        stt_service = WhisperLiveKitSTT(
            api_url=whisper_url,
            model="base",
            language="de"
        )
        
        # Initialize TTS - verwende die korrekte IP für Piper
        piper_url = os.getenv('PIPER_URL', 'http://172.16.0.146:8001')
        logger.info(f"Connecting to Piper TTS at {piper_url}")
        
        tts_service = RemotePiperTTS(
            voice="de_DE-thorsten-medium",
            base_url=piper_url
        )
        
        # Initialize LLM - anpassen Sie die URL für Ihren Ollama Server
        # Falls Ollama auf einem anderen Server läuft, ändern Sie die IP hier
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
        
        logger.info("✓ WhisperLiveKit Agent successfully started!")
        logger.info(f"✓ STT: WhisperLiveKit on {whisper_url}")
        logger.info(f"✓ TTS: Piper on {piper_url}")
        logger.info(f"✓ LLM: Ollama on {ollama_url}")
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
