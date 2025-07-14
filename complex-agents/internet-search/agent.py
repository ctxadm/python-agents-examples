import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero

from web_tools import WebSearchTools

logger = logging.getLogger("internet-search-agent")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

# Ollama Konfiguration (basierend auf Ihrer aktuellen Installation)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

class InternetSearchAgent(Agent):
    def __init__(self) -> None:
        # Web-Tools initialisieren
        self.web_tools = WebSearchTools()
        
        # Ollama LLM konfigurieren (wie Ihr aktueller vision-ollama Agent)
        ollama_llm = openai.LLM(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.3,
            max_tokens=300  # Etwas mehr Tokens für Web-Antworten
        )
        
        super().__init__(
            instructions="""
Du bist ein hilfreicher Assistent mit Internetzugriff.

WICHTIGE REGELN:
- Verwende die verfügbaren Web-Suchfunktionen für aktuelle Informationen
- Antworte präzise und unter 150 Wörtern
- Gib IMMER Quellen an, wenn du Webinformationen verwendest
- Verwende search_web() für allgemeine Informationen
- Verwende search_news() für aktuelle Nachrichten
- Verwende fetch_webpage() um spezifische Webseiten vollständig zu lesen
- Verwende get_weather() für Wetterinformationen
- Suche nur dann im Internet, wenn die Frage aktuelle oder spezifische Informationen erfordert
- Bei unklaren Fragen, frage kurz nach welche Art von Information benötigt wird

BEISPIELE wann zu suchen:
- "Was sind die neuesten Nachrichten über..."
- "Wie ist das Wetter in..."
- "Was kostet aktuell..."
- "Aktuelle Informationen über..."
- "Neueste Entwicklungen bei..."
            """,
            stt=deepgram.STT(),
            llm=ollama_llm,
            tts=openai.TTS(),
            # Web-Tools beim Agent registrieren
            functions=[
                self.web_tools.search_web,
                self.web_tools.fetch_webpage,
                self.web_tools.search_news,
                self.web_tools.get_weather
            ]
        )
    
    async def on_exit(self):
        """Cleanup beim Beenden"""
        await self.web_tools.close()
        await super().on_exit()

async def entrypoint(ctx: JobContext):
    """Agent-Startpunkt für LiveKit"""
    # Verbindung zu LiveKit Room herstellen
    await ctx.connect(auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY)
    
    logger.info("Starte Internet Search Agent...")
    
    # Agent initialisieren und starten
    agent = InternetSearchAgent()
    
    # Agent mit dem Room verbinden
    await agent.start(ctx.room)
    
    logger.info("🌐 Internet Search Agent gestartet und bereit!")
    logger.info("Verfügbare Funktionen:")
    logger.info("  - Web-Suche (search_web)")
    logger.info("  - Nachrichten-Suche (search_news)")
    logger.info("  - Webseiten laden (fetch_webpage)")
    logger.info("  - Wetter-Informationen (get_weather)")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
