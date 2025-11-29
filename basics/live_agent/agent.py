# File: basics/live_agent/agent.py
import logging
import os
import asyncio
from dataclasses import dataclass
from enum import Enum
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
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

Regeln für Zahlen:
- Schreibe Ziffern NICHT als Zahlen wie 67000.
- Schreibe ALLE Zahlen, egal wie groß, IMMER vollständig ausgeschrieben: 
  67000 → siebenundsechzigtausend
  150000 → einhundertfünfzigtausend
- Schreibe auch Daten, Uhrzeiten und Nummern in ausgeschriebener Form, außer sie wurden ausdrücklich in Ziffern gefordert.
- Verwende keine Ziffernfolgen wie „6 7 0 0 0“.

Kommunikationsstil:
- Antworte natürlich und ausführlich.
- Bei komplexen Themen: vollständige, detaillierte Antworten.
- Bei einfachen Fragen: prägnant.
- Strukturiere längere Antworten mit klaren Pausen.
- Keine Sätze über 25 Wörter.
- Sprich „z. B.“ als „zum Beispiel“ aus.
- Sprich „1.“ als „erstens“ usw. aus.
- Sprich wie ein echter Mensch.
"""
        logger.info("Thorsten gestartet - Piper TTS via LocalAI")

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
        model=os.getenv("OLLAMA_MODEL", "GPT-UNIFIED:latest"),
        base_url=os.getenv("OLLAMA_URL", "http://172.16.0.135:11434/v1"),
    )

    session = AgentSession[UserData](
        userdata=UserData(),
        llm=llm,
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=5, timeout=30.0),
            stt_conn_options=APIConnectOptions(max_retry=3, timeout=30.0),
            tts_conn_options=APIConnectOptions(max_retry=3, timeout=30.0),
        ),
        vad=silero.VAD.load(
            min_silence_duration=0.5,
            min_speech_duration=0.2
        ),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(
            model="tts-1",
            voice="alloy",
            base_url="http://172.16.0.220:8888/v1",
            api_key="sk-nokey",
            speed=1.05,
        ),
        min_endpointing_delay=0.25,
        max_endpointing_delay=2.5,
    )

    agent = LiveAgent()
    await session.start(room=ctx.room, agent=agent)

    greeting = "Guten Tag! Ich bin Thorsten. Schön, dass Sie da sind. Womit kann ich Ihnen heute helfen?"
    try:
        await session.say(greeting, allow_interruptions=True, add_to_chat_ctx=True)
        session.userdata.greeting_sent = True
        session.userdata.state = ConversationState.TALKING
        logger.info("✅ Begrüßung erfolgreich")
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
