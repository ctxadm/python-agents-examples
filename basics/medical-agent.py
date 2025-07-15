import os
import logging
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, deepgram, silero

logger = logging.getLogger("medical-ollama")

# Medical-spezifische Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.16.0.146:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

class MedicalAgent(Agent):
    def __init__(self) -> None:
        # Medical-spezifisches LLM
        medical_llm = openai.LLM(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.7,
        )
        
        super().__init__(
            instructions="""You are a helpful medical information assistant.
            Important: Always remind users that you provide general information only 
            and they should consult healthcare professionals for medical advice.
            Be accurate, empathetic, and clear in your responses.""",
            stt=deepgram.STT(),
            llm=medical_llm,
            tts=deepgram.TTS(),
            vad=silero.VAD.load()
        )
    
    async def on_enter(self):
        logger.info("Medical assistant ready")
        # Optional: Begrüßung
        # self.session.generate_reply()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    logger.info("Medical assistant starting")
    
    session = AgentSession()
    await session.start(
        agent=MedicalAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
