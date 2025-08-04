# LiveKit Agents - Internet Search Agent (Angepasst nach Garage Agent Muster)
import logging
import os
import httpx
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

# Importiere die WebSearchTools aus deinem web_tools Modul
# from web_tools import WebSearchTools

load_dotenv()

# Logging
logger = logging.getLogger("internet-search-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-internet-search-1")

class SearchState(Enum):
    """State Machine für Such-Konversationen"""
    GREETING = "greeting"
    AWAITING_QUERY = "awaiting_query"
    SEARCHING = "searching"
    PROVIDING_RESULTS = "providing_results"
    FOLLOW_UP = "follow_up"

@dataclass
class SearchContext:
    """Kontext für Internet-Suchen"""
    last_query: Optional[str] = None
    last_results: Optional[List[Dict[str, Any]]] = None
    search_type: Optional[str] = None  # web, news, weather, webpage
    follow_up_count: int = 0
    
    def reset(self):
        """Reset des Suchkontexts"""
        self.last_query = None
        self.last_results = None
        self.search_type = None
        self.follow_up_count = 0

@dataclass
class InternetSearchUserData:
    """User data context für den Internet Search Agent"""
    authenticated_user: Optional[str] = None
    greeting_sent: bool = False
    conversation_state: SearchState = SearchState.GREETING
    search_context: SearchContext = field(default_factory=SearchContext)
    current_location: Optional[str] = None  # Für Wetter-Suchen
    language: str = "de"  # Standard: Deutsch


class QueryAnalyzer:
    """Analysiert User-Queries und extrahiert Intent"""
    
    @classmethod
    def extract_intent_from_query(cls, query: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Query"""
        query_lower = query.lower().strip()
        
        # Wetter-Abfragen
        weather_keywords = ["wetter", "temperatur", "regen", "sonne", "schnee", "wind", "vorhersage", "forecast"]
        if any(word in query_lower for word in weather_keywords):
            # Versuche Ort zu extrahieren
            location = cls._extract_location(query)
            return {"intent": "weather", "data": location or query}
        
        # Nachrichten-Abfragen
        news_keywords = ["nachrichten", "news", "aktuell", "neuigkeiten", "schlagzeilen", "heute", "breaking"]
        if any(word in query_lower for word in news_keywords):
            return {"intent": "news", "data": query}
        
        # Webseiten-Abruf (URLs)
        if "http://" in query_lower or "https://" in query_lower or "www." in query_lower:
            return {"intent": "webpage", "data": query}
        
        # Spezifische Webseiten-Anfragen
        webpage_keywords = ["webseite", "website", "seite von", "homepage"]
        if any(word in query_lower for word in webpage_keywords):
            return {"intent": "webpage_search", "data": query}
        
        # Grüße
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        if any(g in query_lower for g in greetings) and len(query_lower) < 20:
            return {"intent": "greeting", "data": None}
        
        # Standard: Web-Suche
        return {"intent": "web_search", "data": query}
    
    @classmethod
    def _extract_location(cls, query: str) -> Optional[str]:
        """Extrahiert Ortsnamen aus Wetter-Abfragen"""
        # Einfache Extraktion nach "in" oder "für"
        patterns = [
            r"(?:in|für|von)\s+([A-Za-zäöüÄÖÜß\s]+?)(?:\s|$|\.|\?)",
            r"wetter\s+([A-Za-zäöüÄÖÜß\s]+?)(?:\s|$|\.|\?)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Filter common words
                if location.lower() not in ["heute", "morgen", "übermorgen"]:
                    return location
        
        return None


class InternetSearchAssistant(Agent):
    """Internet Search Assistant mit Web-Zugriff"""
    
    def __init__(self) -> None:
        super().__init__(instructions="""Du bist ein hilfreicher Internet-Such-Assistent. ANTWORTE NUR AUF DEUTSCH.

DEINE HAUPTAUFGABEN:
1. Aktuelle Informationen aus dem Internet suchen
2. Nachrichten abrufen
3. Wetter-Informationen bereitstellen
4. Webseiten-Inhalte zusammenfassen

WICHTIGE REGELN:
- Verwende die Such-Funktionen NUR wenn nötig
- Gib IMMER Quellen an bei Web-Informationen
- Halte Antworten präzise und unter 150 Wörtern
- Bei Wetter: Frage nach dem Ort wenn nicht angegeben
- Bei News: Fokussiere auf aktuelle Schlagzeilen
- NIEMALS URLs oder Links erfinden

ANTWORT-STRUKTUR bei Suchergebnissen:
"Hier sind die Ergebnisse zu Ihrer Anfrage:

📌 HAUPTERGEBNISSE:
[Die wichtigsten Informationen]

🔗 QUELLEN:
[Liste der verwendeten Quellen]

Kann ich Ihnen noch weitere Informationen zu diesem Thema liefern?"

VERBOTENE WÖRTER: Verwende NIEMALS "Entschuldigung", "Es tut mir leid", "Sorry" - nutze stattdessen "Leider".""")
        
        self.query_analyzer = QueryAnalyzer()
        logger.info("✅ InternetSearchAssistant initialized")
    
    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Internet Search Agent on_enter called")
    
    @function_tool
    async def search_web(self,
                        context: RunContext[InternetSearchUserData],
                        query: str) -> str:
        """
        Führt eine allgemeine Web-Suche durch.
        
        Args:
            query: Suchbegriff
            
        Returns:
            Suchergebnisse oder Fehlermeldung
        """
        logger.info(f"🔍 Web search for: {query}")
        
        # Store query in context
        context.userdata.search_context.last_query = query
        context.userdata.search_context.search_type = "web"
        context.userdata.conversation_state = SearchState.SEARCHING
        
        try:
            # Simuliere Web-Suche (ersetze mit echter Implementierung)
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Beispiel mit DuckDuckGo API (ersetze mit deiner Implementierung)
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    results = []
                    
                    # Abstract
                    if data.get("Abstract"):
                        results.append(f"📄 {data['Abstract']}")
                        if data.get("AbstractURL"):
                            results.append(f"   Quelle: {data['AbstractURL']}")
                    
                    # Related Topics
                    if data.get("RelatedTopics"):
                        results.append("\n🔍 Verwandte Themen:")
                        for i, topic in enumerate(data["RelatedTopics"][:3]):
                            if isinstance(topic, dict) and topic.get("Text"):
                                results.append(f"   {i+1}. {topic['Text']}")
                    
                    if results:
                        context.userdata.conversation_state = SearchState.PROVIDING_RESULTS
                        return "\n".join(results)
                    else:
                        return f"Leider konnte ich keine relevanten Ergebnisse für '{query}' finden."
                else:
                    return "Die Suche ist momentan nicht verfügbar. Bitte versuchen Sie es später noch einmal."
                    
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return "Bei der Suche ist ein Fehler aufgetreten."
    
    @function_tool
    async def search_news(self,
                         context: RunContext[InternetSearchUserData],
                         query: str) -> str:
        """
        Sucht nach aktuellen Nachrichten.
        
        Args:
            query: Nachrichtenthema
            
        Returns:
            Nachrichtenergebnisse oder Fehlermeldung
        """
        logger.info(f"📰 News search for: {query}")
        
        context.userdata.search_context.last_query = query
        context.userdata.search_context.search_type = "news"
        context.userdata.conversation_state = SearchState.SEARCHING
        
        try:
            # Simuliere Nachrichten-Suche
            # In der echten Implementierung würdest du eine News API verwenden
            results = [
                "📰 AKTUELLE NACHRICHTEN:",
                f"Zu '{query}' wurden folgende Schlagzeilen gefunden:",
                "",
                "1. [Beispiel-Schlagzeile 1]",
                "   Quelle: Nachrichtenagentur",
                "   Zeit: vor 2 Stunden",
                "",
                "2. [Beispiel-Schlagzeile 2]",
                "   Quelle: Online-Zeitung",
                "   Zeit: vor 4 Stunden"
            ]
            
            context.userdata.conversation_state = SearchState.PROVIDING_RESULTS
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"News search error: {e}", exc_info=True)
            return "Die Nachrichtensuche ist momentan nicht verfügbar."
    
    @function_tool
    async def get_weather(self,
                         context: RunContext[InternetSearchUserData],
                         location: str) -> str:
        """
        Ruft Wetterinformationen für einen Ort ab.
        
        Args:
            location: Ortsname
            
        Returns:
            Wetterinformationen oder Fehlermeldung
        """
        logger.info(f"🌤️ Weather search for: {location}")
        
        context.userdata.search_context.last_query = f"Wetter in {location}"
        context.userdata.search_context.search_type = "weather"
        context.userdata.conversation_state = SearchState.SEARCHING
        
        try:
            # Simuliere Wetter-API Aufruf
            # In der echten Implementierung würdest du eine Wetter API verwenden
            results = [
                f"🌤️ WETTER IN {location.upper()}:",
                "",
                "Aktuell:",
                "🌡️ Temperatur: 18°C",
                "☁️ Bewölkt",
                "💨 Wind: 12 km/h",
                "💧 Luftfeuchtigkeit: 65%",
                "",
                "Vorhersage:",
                "Heute: 20°C / 14°C - Teilweise bewölkt",
                "Morgen: 22°C / 15°C - Sonnig"
            ]
            
            context.userdata.conversation_state = SearchState.PROVIDING_RESULTS
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Weather search error: {e}", exc_info=True)
            return f"Die Wetterinformationen für {location} sind momentan nicht verfügbar."
    
    @function_tool
    async def fetch_webpage(self,
                           context: RunContext[InternetSearchUserData],
                           url: str) -> str:
        """
        Lädt und fasst den Inhalt einer Webseite zusammen.
        
        Args:
            url: Die URL der Webseite
            
        Returns:
            Zusammenfassung des Webseiteninhalts
        """
        logger.info(f"🌐 Fetching webpage: {url}")
        
        context.userdata.search_context.last_query = url
        context.userdata.search_context.search_type = "webpage"
        context.userdata.conversation_state = SearchState.SEARCHING
        
        try:
            # Bereinige URL
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    # In der echten Implementierung würdest du HTML parsen
                    # und den relevanten Text extrahieren
                    results = [
                        f"📄 WEBSEITEN-ZUSAMMENFASSUNG:",
                        f"URL: {url}",
                        "",
                        "Hauptinhalt:",
                        "[Hier würde der extrahierte und zusammengefasste Inhalt stehen]",
                        "",
                        "Die Webseite wurde erfolgreich abgerufen."
                    ]
                    
                    context.userdata.conversation_state = SearchState.PROVIDING_RESULTS
                    return "\n".join(results)
                else:
                    return f"Die Webseite {url} konnte nicht abgerufen werden (Status: {response.status_code})."
                    
        except Exception as e:
            logger.error(f"Webpage fetch error: {e}", exc_info=True)
            return f"Beim Abrufen der Webseite {url} ist ein Fehler aufgetreten."


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Internet Search Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🏁 Starting Internet Search Agent Session: {session_id}")
    logger.info("="*50)
    
    session = None
    session_closed = False
    
    # Register disconnect handler FIRST
    def on_disconnect():
        nonlocal session_closed
        logger.info(f"[{session_id}] Room disconnected event received")
        session_closed = True
    
    if ctx.room:
        ctx.room.on("disconnected", on_disconnect)
    
    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10
        
        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                    logger.info(f"✅ [{session_id}] Audio track found and subscribed")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
            
            await asyncio.sleep(1)
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
        
        if not audio_track_received:
            logger.warning(f"⚠️ [{session_id}] No audio track found after {max_wait_time}s, continuing anyway...")
        
        # 4. Configure LLM with Ollama
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.3,
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 for Internet Search")
        
        # 5. Create session
        session = AgentSession[InternetSearchUserData](
            userdata=InternetSearchUserData(
                authenticated_user=None,
                greeting_sent=False,
                conversation_state=SearchState.GREETING,
                search_context=SearchContext(),
                current_location=None,
                language="de"
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,
                min_speech_duration=0.2
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )
        
        # 6. Create agent
        agent = InternetSearchAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Warte auf Audio-Stabilisierung
        await asyncio.sleep(2.0)
        
        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            intent_result = QueryAnalyzer.extract_intent_from_query(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("function_call")
        def on_function_call(event):
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
        
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Generated response preview: {response_preview}...")
        
        # 8. Initial greeting
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = """Guten Tag! Ich bin Ihr persönlicher Internet-Such-Assistent.

Ich kann Ihnen bei folgenden Aufgaben helfen:
🔍 Allgemeine Web-Suchen
📰 Aktuelle Nachrichten abrufen
🌤️ Wetterinformationen liefern
🌐 Webseiten-Inhalte zusammenfassen

Womit kann ich Ihnen heute helfen?"""
            
            session.userdata.greeting_sent = True
            session.userdata.conversation_state = SearchState.AWAITING_QUERY
            
            # Retry-Mechanismus für Greeting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await session.say(
                        greeting_text,
                        allow_interruptions=True,
                        add_to_chat_ctx=True
                    )
                    logger.info(f"✅ [{session_id}] Initial greeting sent successfully")
                    break
                except Exception as e:
                    logger.warning(f"[{session_id}] Greeting attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error after all retries: {e}", exc_info=True)
        
        logger.info(f"✅ [{session_id}] Internet Search Agent ready!")
        logger.info(f"Available functions: search_web, search_news, get_weather, fetch_webpage")
        
        # Wait for disconnect
        disconnect_event = asyncio.Event()
        
        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()
        
        ctx.room.on("disconnected", handle_disconnect)
        
        await disconnect_event.wait()
        logger.info(f"[{session_id}] Session ending...")
    
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass
        
        logger.info(f"✅ [{session_id}] Cleanup complete")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
