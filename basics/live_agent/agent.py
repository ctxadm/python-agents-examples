# File: basics/live_agent/agent.py
import logging
import os
import asyncio
from dataclasses import dataclass
from enum import Enum
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = os.getenv("AGENT_NAME", "live-agent-1")

class ConversationState(Enum):
    GREETING = "greeting"
    TALKING = "talking"

@dataclass
class UserData:
    greeting_sent: bool = False
    state: ConversationState = ConversationState.GREETING

class LiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Thorsten, ein freundlicher digitaler Assistent.
Antworte AUSSCHLIESSLICH auf Deutsch, immer höflich und klar.

WICHTIG für bessere Sprachausgabe:
- Formuliere kurze, klare Sätze (max. 15-20 Wörter)
- Vermeide komplexe Verschachtelungen
- Mache zwischen Gedanken natürliche Pausen (Punkt statt Komma)
- Sprich wie ein echter Mensch, nicht wie ein Script""")
        logger.info("Thorsten gestartet mit Piper TTS via LocalAI")

async def request_handler(ctx: JobContext):
    logger.info(f"[{AGENT_NAME}] Verbindung angefragt")
    await ctx.accept()

async def entrypoint(ctx: JobContext):
    logger.info("="*80)
    logger.info("THORSTEN LIVE-AGENT GESTARTET")
    logger.info("="*80)
    
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Teilnehmer: {participant.identity}")

    llm = openai.LLM.with_ollama(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:20B"),
        base_url=os.getenv("OLLAMA_URL", "http://172.16.0.139:11434/v1"),
        temperature=0.7,
    )

    session = AgentSession[UserData](
        userdata=UserData(),
        llm=llm,
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(
            model="tts-1",
            voice="de_DE-eva_k-x_low",
            base_url="http://172.16.0.220:8888/v1",
            api_key="sk-nokey",
        ),
        min_endpointing_delay=1.2,    # ← HAUPTÄNDERUNG: von 0.5 auf 1.2
        max_endpointing_delay=5.0,    # ← HAUPTÄNDERUNG: von 4.0 auf 5.0
        allow_interruptions=True,
    )

    agent = LiveAgent()
    await session.start(room=ctx.room, agent=agent)

    greeting = "Guten Tag! Ich bin Thorsten. Womit kann ich helfen?"
    
    try:
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries:
            try:
                await session.say(
                    greeting, 
                    allow_interruptions=True, 
                    add_to_chat_ctx=True
                )
                session.userdata.greeting_sent = True
                session.userdata.state = ConversationState.TALKING
                logger.info("✅ Begrüßung erfolgreich")
                break
            except Exception as e:
                retry_count += 1
                logger.warning(f"⚠️ TTS Retry {retry_count}/{max_retries}: {e}")
                if retry_count > max_retries:
                    raise
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"❌ TTS-Fehler: {e}")

    disconnect_event = asyncio.Event()
    ctx.room.on("disconnected", lambda: disconnect_event.set())
    await disconnect_event.wait()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
