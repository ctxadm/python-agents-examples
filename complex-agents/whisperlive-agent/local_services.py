import logging
import aiohttp
import io
from typing import Optional, AsyncIterator, Literal
from livekit.agents.tts import TTS, TTSCapabilities, ChunkedStream, SynthesizeStream

logger = logging.getLogger("local_services")


class RemotePiperTTS(TTS):
    """Remote Piper TTS implementation"""
    
    def __init__(
        self,
        voice: str = "de_DE-thorsten-medium",
        base_url: str = "http://localhost:8001",
        sample_rate: int = 16000,
        encoding: Literal["pcm_s16le", "pcm_f32le"] = "pcm_s16le"
    ):
        super().__init__(
            capabilities=TTSCapabilities(
                streaming=False
            ),
            sample_rate=sample_rate,
            num_channels=1
        )
        self.voice = voice
        self.base_url = base_url.rstrip('/')
        self.encoding = encoding
        self._session = None
        logger.info(f"Initialized RemotePiperTTS with {base_url}, voice: {voice}")
    
    async def _ensure_session(self):
        if not self._session:
            self._session = aiohttp.ClientSession()
    
    def synthesize(self, text: str) -> ChunkedStream:
        """Synthesize text to speech"""
        return RemotePiperChunkedStream(self, text)
    
    async def aclose(self):
        if self._session:
            await self._session.close()
            self._session = None


class RemotePiperChunkedStream(ChunkedStream):
    """Stream implementation for Piper TTS"""
    
    def __init__(self, tts: RemotePiperTTS, text: str):
        super().__init__(tts=tts, input_text=text)
        self._tts = tts
        self._text = text
    
    async def _run(self):
        """Generate audio from text"""
        try:
            await self._tts._ensure_session()
            
            # Prepare request data
            params = {
                'text': self._text,
                'voice': self._tts.voice,
                'format': 'wav'  # Request WAV format
            }
            
            logger.info(f"Synthesizing: {self._text[:50]}...")
            
            # Make request to Piper
            async with self._tts._session.get(
                f"{self._tts.base_url}/tts",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    # Read the entire response
                    audio_data = await response.read()
                    
                    # Parse WAV header to get audio format info
                    import wave
                    wav_buffer = io.BytesIO(audio_data)
                    
                    try:
                        with wave.open(wav_buffer, 'rb') as wav:
                            # Skip WAV header and get raw PCM data
                            wav_buffer.seek(44)  # Standard WAV header size
                            pcm_data = wav_buffer.read()
                            
                        # Send as a single chunk
                        self._event_ch.send_nowait(
                            SynthesizeStream.Event(
                                type=SynthesizeStream.EventType.AUDIO,
                                audio=rtc.AudioFrame(
                                    data=pcm_data,
                                    sample_rate=self._tts.sample_rate,
                                    num_channels=1,
                                    samples_per_channel=len(pcm_data) // 2  # 16-bit samples
                                )
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error processing WAV: {e}")
                        # Fallback: send raw data
                        self._event_ch.send_nowait(
                            SynthesizeStream.Event(
                                type=SynthesizeStream.EventType.AUDIO,
                                audio=rtc.AudioFrame(
                                    data=audio_data[44:],  # Skip header
                                    sample_rate=self._tts.sample_rate,
                                    num_channels=1,
                                    samples_per_channel=(len(audio_data) - 44) // 2
                                )
                            )
                        )
                else:
                    logger.error(f"Piper TTS error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
        finally:
            # Always mark as finished
            self._event_ch.send_nowait(
                SynthesizeStream.Event(
                    type=SynthesizeStream.EventType.FINISHED
                )
            )


# Import rtc for AudioFrame
from livekit import rtc
from livekit.agents.tts import SynthesizeStream
