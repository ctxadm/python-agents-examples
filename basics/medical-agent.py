import os
import logging
from livekit import agents
from livekit.plugins import openai, deepgram, silero
from livekit.agents.voice import VoicePipelineAgent

logger = logging.getLogger("medical-ollama")

# Medical-spezifische Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.146:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    participant = await ctx.wait_for_participant()
    logger.info(f"Medical assistant starting for participant {participant.identity}")
    
    # Medical-spezifisches LLM
    medical_llm = openai.LLM(
        model=OLLAMA_MODEL,
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama",
        timeout=120.0,
        temperature=0.7,
    )
    
    # System Prompt für Medical Assistant
    medical_llm = medical_llm.with_instructions(
        """You are a helpful medical information assistant. 
        You provide general health information and always remind users 
        to consult with healthcare professionals for medical advice.
        Keep responses concise and friendly."""
    )
    
    # Voice Pipeline Setup
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=medical_llm,
        tts=openai.TTS(voice="sage"),
    )
    
    agent.start(ctx.room, participant)
    
    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: str):
        logger.info(f"Medical agent said: {msg}")
    
    await agent.join()
