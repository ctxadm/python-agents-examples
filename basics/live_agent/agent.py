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
        super().__init__(instructions="""Du bist Thorsten, ein freundlicher und kompetenter digitaler Assistent.
Antworte AUSSCHLIESSLICH auf Deutsch, immer höflich, klar und natürlich.
Du bist geduldig, hilfsbereit und sprichst wie ein echter Mensch.
Führe einfach ein normales, lockeres Gespräch – kein Skript.""")
        logger.info("Thorsten (Live-Agent) gestartet – 100% lokal mit Piper TTS")

async def request_handler(ctx: JobContext):
    logger.info(f"[{AGENT_NAME}] Verbindung angefragt")
    await ctx.accept()

async def entrypoint(ctx: JobContext):
    logger.info("="*80)
    logger.info("LIVE-AGENT (THORSTEN) GESTARTET – bereit für live_room_*")
    logger.info("="*80)

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Teilnehmer beigetreten: {participant.identity}")

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
            voice="alloy",                              # = thorsten-high
            base_url="http://172.16.0.220:8888/v1",     # deine Piper-Instanz
            api_key="sk-nokey",
        ),
        min_endpointing_delay=0.5,
        max_endpointing_delay=4.0
    )

    agent = LiveAgent()
    await session.start(room=ctx.room, agent=agent)

    greeting = "Guten Tag! Ich bin Thorsten. Schön, dass Sie da sind. Womit kann ich Ihnen heute helfen?"
    await session.say(greeting, allow_interruptions=True, add_to_chat_ctx=True)

    session.userdata.greeting_sent = True
    session.userdata.state = ConversationState.TALKING
    logger.info("Thorsten ist live und spricht mit deiner eigenen deutschen Stimme")

    disconnect_event = asyncio.Event()
    ctx.room.on("disconnected", lambda: disconnect_event.set())
    await disconnect_event.wait()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
