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
Kommunikationsstil:
- Antworte natürlich und ausführlich auf Fragen
- Bei komplexen Themen: Gib vollständige, detaillierte Antworten
- Bei einfachen Fragen: Sei prägnant
- Strukturiere längere Antworten mit klaren Gedankenpausen
- Vermeide Sätze über 25 Wörter ohne Punkt
- Sprich wie ein echter Mensch""")
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
        base_url=os.getenv("OLLAMA_URL", "http://172.16.0.140:11435/v1"),
    )
    # ✅ OPTIMIERTE KONFIGURATION - NUR VALIDE PARAMETER
    session = AgentSession[UserData](
        userdata=UserData(),
        llm=llm,
        vad=silero.VAD.load(
            min_silence_duration=0.5,     # Optimal für deutsche Sprache
            min_speech_duration=0.2       # Schnelle Erkennung
        ),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(
            model="tts-1",
            voice="alloy",                              # = thorsten-high
            base_url="http://172.16.0.220:8888/v1",     # ← DEINE RICHTIGE IP!
            api_key="sk-nokey",
            speed=1.05,
        ),
        min_endpointing_delay=0.25,        # ⚡ Schnelle Reaktion (wie gestern!)
        max_endpointing_delay=2.5,        # Ausreichend für längere Sätze
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
