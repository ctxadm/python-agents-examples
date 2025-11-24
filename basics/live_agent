# conversation_agent.py
import logging
import os
import asyncio
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("conversation-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = os.getenv("AGENT_NAME", "conversation-agent-1")

# =============================================================================
# Zustände (minimal)
# =============================================================================
class ConversationState(Enum):
    GREETING = "greeting"
    TALKING = "talking"

@dataclass
class ConversationUserData:
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING

# =============================================================================
# Der reine, freie Conversation Agent – THORSTEN
# =============================================================================
class ConversationAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Thorsten, ein freundlicher und kompetenter digitaler Assistent.
Antworte AUSSCHLIESSLICH auf Deutsch, immer höflich, klar und natürlich.
Du bist geduldig, hilfsbereit und sprichst wie ein echter Mensch.
Du hast kein festes Skript – du führst einfach ein normales, lockeres Gespräch.""")
        
        logger.info("Thorsten (ConversationAgent) erfolgreich gestartet – 100 % lokal mit Piper TTS")

# =============================================================================
# Job-Request Handler
# =============================================================================
async def request_handler(ctx: JobContext):
    logger.info(f"[{AGENT_NAME}] Verbindung angefragt")
    await ctx.accept()

# =============================================================================
# Haupt-Entrypoint
# =============================================================================
async def entrypoint(ctx: JobContext):
    session_id482 = f"thorsten_{int(asyncio.get_event_loop().time())}"
    logger.info("="*70)
    logger.info(f"THORSTEN GESTARTET – Session: {session_id482}")
    logger.info("="*70)

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Teilnehmer verbunden: {participant.identity}")

    # Ollama LLM (oder was du sonst nutzt)
    llm = openai.LLM.with_ollama(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:20B"),
        base_url=os.getenv("OLLAMA_URL", "http://172.16.0.139:11434/v1"),
        temperature=0.7,
    )

    # Session – 100 % lokal mit deiner Piper TTS (Thorsten High = alloy)
    session = AgentSession[ConversationUserData](
        userdata=ConversationUserData(),
        llm=llm,
        vad=silero.VAD.load(min_silence_duration=0.5, min_speech_duration=0.2),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(
            model="tts-1",
            voice="alloy",                                # = thorsten-high → beste deutsche Männerstimme
            base_url="http://172.16.0.205:8000/v1",       # ← DEINE PIPER SERVER IP
            api_key="sk-nokey",
            temperature=0.7,
            response_format="pcm_16000"                   # minimale Latenz
        ),
        min_endpointing_delay=0.3,
        max_endpointing_delay=3.0
    )

    agent = ConversationAgent()
    await session.start(room=ctx.room, agent=agent)

    # Begrüßung von Thorsten
    greeting = "Guten Tag! Ich bin Thorsten. Schön, dass Sie da sind. Womit kann ich Ihnen heute helfen?"
    await session.say(greeting, allow_interruptions=True, add_to_chat_ctx=True)
    
    session.userdata.greeting_sent = True
    session.userdata.conversation_state = ConversationState.TALKING

    logger.info("Thorsten ist bereit und spricht jetzt mit deiner eigenen deutschen Stimme")

    # Warten bis der Anrufer auflegt
    disconnect_event = asyncio.Event()
    ctx.room.on("disconnected", lambda: disconnect_event.set())
    await disconnect_event.wait()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
