# LiveKit Agents - Garage Management Agent (FULLY FIXED FOR 1.0.23)
# Für LiveKit Agents Version 1.0.23 mit Llama 3.2
import logging
import os
import httpx
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    JobContext, 
    WorkerOptions,
    cli,
    llm,
    RunContext,
    function_tool
)
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("garage-agent")

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    current_customer_id: Optional[str] = None
    active_repair_id: Optional[str] = None
    user_language: str = "de"
    greeting_sent: bool = False
    customer_name: Optional[str] = None

# Klare Instructions für Llama 3.2
LLAMA32_INSTRUCTIONS = """Du bist Pia von der Autowerkstatt Zürich.

WICHTIGSTE REGEL: Bei der ersten Nachricht oder wenn du gebeten wirst zu grüßen, sage IMMER GENAU:
"Guten Tag und herzlich willkommen bei der Autowerkstatt Zürich! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?"

Weitere Regeln:
- Antworte IMMER auf Deutsch
- Verwende NIE: Entschuldigung, Sorry, Leider
- Erfinde KEINE Daten
- Nutze Suchfunktionen nur bei konkreten Fragen"""

class GarageAssistant(Agent):
    """Garage Assistant für die Autowerkstatt Zürich"""
    
    def __init__(self) -> None:
        super().__init__(instructions=LLAMA32_INSTRUCTIONS)
        logger.info("✅ GarageAssistant initialized für Autowerkstatt Zürich")

@function_tool
async def search_customer_data(
    context: RunContext[GarageUserData],
    query: str
) -> str:
    """
    Sucht nach Kundendaten in der Garage-Datenbank.
    
    Args:
        query: Suchbegriff (Name, Telefonnummer oder Autokennzeichen)
    """
    logger.info(f"🔍 Searching customer data for: {query}")
    
    # GUARD gegen Begrüßungen
    greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
    if query.lower().strip() in greetings or len(query) < 3:
        return "Für die Suche benötige ich Ihren Namen oder Ihr Autokennzeichen."
    
    # Extrahiere Kennzeichen
    kennzeichen_pattern = r'[A-Z]{2}\s*\d{3,6}'
    kennzeichen_match = re.search(kennzeichen_pattern, query.upper())
    
    try:
        async with httpx.AsyncClient() as client:
            # Baue Request
            search_request = {
                "query": query,
                "agent_type": "garage",
                "top_k": 5,
                "collection": "garage_management"
            }
            
            response = await client.post(
                f"{context.userdata.rag_url}/search",
                json=search_request
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                if results:
                    logger.info(f"✅ Found {len(results)} results")
                    
                    relevant_results = []
                    for result in results[:3]:
                        content = result.get("content", "").strip()
                        if content and not any(word in content.lower() for word in ["mhz", "bakom", "funk"]):
                            # Formatiere für bessere Lesbarkeit
                            content = content.replace('_', ' ')
                            content = re.sub(r'CHF\s*(\d+)', r'\1 Franken', content)
                            content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3.\2.\1', content)
                            relevant_results.append(content)
                    
                    if relevant_results:
                        return "\n\n".join(relevant_results[:2])
                    else:
                        if kennzeichen_match:
                            return f"Ich habe keine Daten zum Kennzeichen {kennzeichen_match.group()} gefunden."
                        else:
                            return "Ich habe keine Daten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                
                return "Ich habe keine Kundendaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen (z.B. ZH 123456)."
                
            else:
                logger.error(f"Search failed: {response.status_code}")
                return "Die Datenbank ist momentan nicht erreichbar."
                
    except Exception as e:
        logger.error(f"Search error: {e}")
        return "Es gab einen technischen Fehler."

@function_tool
async def search_repair_status(
    context: RunContext[GarageUserData],
    query: str
) -> str:
    """
    Sucht nach Reparaturstatus und Aufträgen.
    
    Args:
        query: Kundenname, Autokennzeichen oder Auftragsnummer
    """
    logger.info(f"🔧 Searching repair status for: {query}")
    
    if len(query) < 3:
        return "Für die Reparatursuche benötige ich einen Namen oder ein Autokennzeichen."
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{context.userdata.rag_url}/search",
                json={
                    "query": f"Reparatur Service Status {query}",
                    "agent_type": "garage",
                    "top_k": 5,
                    "collection": "garage_management"
                }
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                if results:
                    relevant_results = []
                    for result in results[:3]:
                        content = result.get("content", "").strip()
                        if content and "service" in content.lower():
                            content = content.replace('_', ' ')
                            content = re.sub(r'CHF\s*(\d+)', r'\1 Franken', content)
                            relevant_results.append(content)
                    
                    if relevant_results:
                        return "\n\n".join(relevant_results[:2])
                    
                return "Ich habe keine Reparaturdaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                
            else:
                return "Die Reparaturdatenbank ist momentan nicht verfügbar."
                
    except Exception as e:
        logger.error(f"Repair search error: {e}")
        return "Technischer Fehler bei der Reparatursuche."

@function_tool
async def search_invoice_data(
    context: RunContext[GarageUserData],
    query: str
) -> str:
    """
    Sucht nach Rechnungsinformationen und Kosten.
    
    Args:
        query: Kundenname, Rechnungsnummer oder Kennzeichen
    """
    logger.info(f"💰 Searching invoice/cost data for: {query}")
    
    if len(query) < 3:
        return "Für die Rechnungssuche benötige ich einen Namen oder ein Autokennzeichen."
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{context.userdata.rag_url}/search",
                json={
                    "query": f"Rechnung Kosten Service {query}",
                    "agent_type": "garage",
                    "top_k": 5,
                    "collection": "garage_management"
                }
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                if results:
                    cost_results = []
                    for result in results[:3]:
                        content = result.get("content", "").strip()
                        if content and any(word in content.lower() for word in ["kosten", "franken", "rechnung"]):
                            content = content.replace('_', ' ')
                            content = re.sub(r'CHF\s*(\d+)', r'\1 Franken', content)
                            cost_results.append(content)
                    
                    if cost_results:
                        return "\n\n".join(cost_results[:2])
                
                return "Ich habe keine Kosteninformationen gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                
            else:
                return "Die Rechnungsdatenbank ist momentan nicht verfügbar."
                
    except Exception as e:
        logger.error(f"Invoice search error: {e}")
        return "Technischer Fehler bei der Rechnungssuche."

async def request_handler(ctx: JobContext):
    """Request handler"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()

async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚗 Starting Garage Agent Session: {session_id}")
    logger.info(f"🏢 Autowerkstatt Zürich")
    logger.info("="*50)
    
    session = None
    session_closed = False
    
    try:
        # 1. Connect to room mit auto_subscribe
        await ctx.connect(auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY)
        logger.info(f"✅ [{session_id}] Connected to room with auto_subscribe")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. WICHTIG: Warte auf Track Subscription
        max_attempts = 30  # 15 Sekunden warten
        audio_track = None
        
        for attempt in range(max_attempts):
            for pub in participant.track_publications.values():
                if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track is not None:
                    audio_track = pub.track
                    logger.info(f"✅ [{session_id}] Audio track subscribed after {attempt*0.5}s")
                    break
            
            if audio_track:
                break
                
            await asyncio.sleep(0.5)
        
        if not audio_track:
            logger.warning(f"⚠️ [{session_id}] No audio track after 15s, continuing anyway")
        
        # 4. Configure LLM
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,  # Absolut 0 für konsistente Antworten
            max_tokens=150   # Limitiere die Antwortlänge
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama (temp=0.0)")
        
        # 5. Create agent FIRST
        agent = GarageAssistant()
        
        # 6. Create session with proper configuration
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                current_customer_id=None,
                active_repair_id=None,
                user_language="de",
                greeting_sent=False,
                customer_name=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,
                min_speech_duration=0.2,
                padding_duration=0.3
            ),
            stt=openai.STT(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="tts-1",
                voice="nova"
            ),
            # Audio-spezifische Einstellungen
            min_endpointing_delay=0.5,
            max_endpointing_delay=3.0,
            min_interruption_duration=0.3,
            allow_interruptions=True
        )
        
        # 7. Event handlers VOR dem Start
        @session.on("user_speech_committed")
        def on_user_speech(event):
            logger.info(f"[{session_id}] 🎤 User: {event}")
        
        @session.on("agent_speech_committed")
        def on_agent_speech(event):
            logger.info(f"[{session_id}] 🤖 Agent: {event}")
        
        # 8. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(room=ctx.room, agent=agent)
        
        # 9. WICHTIG: Warte bis Session vollständig initialisiert ist
        await asyncio.sleep(2.0)
        
        # 10. Tools hinzufügen
        session.llm.function_tools = [
            llm.FunctionCallInfo(
                name="search_customer_data",
                description="Sucht nach Kundendaten. NUR bei konkreten Anfragen nutzen, NICHT bei Begrüßungen.",
                callable=search_customer_data
            ),
            llm.FunctionCallInfo(
                name="search_repair_status",
                description="Sucht nach Reparaturstatus.",
                callable=search_repair_status
            ),
            llm.FunctionCallInfo(
                name="search_invoice_data",
                description="Sucht nach Rechnungen und Kosten.",
                callable=search_invoice_data
            )
        ]
        
        # 11. AUTOMATISCHE BEGRÜSSUNG - VERBESSERTE VERSION
        await asyncio.sleep(1.0)
        
        if not session.userdata.greeting_sent:
            logger.info(f"📢 [{session_id}] Sending automatic greeting...")
            try:
                # Option 1: Direkt mit say() für garantierte Ausgabe
                greeting_text = "Guten Tag und herzlich willkommen bei der Autowerkstatt Zürich! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?"
                
                await session.say(
                    text=greeting_text,
                    allow_interruptions=True,
                    add_to_chat_ctx=True
                )
                
                session.userdata.greeting_sent = True
                logger.info(f"✅ [{session_id}] Greeting sent successfully via say()!")
                
            except Exception as e:
                logger.error(f"❌ [{session_id}] Greeting failed: {e}", exc_info=True)
                # Fallback: Versuche generate_reply
                try:
                    await session.generate_reply(
                        user_message="",
                        instructions="Sage GENAU diesen Text: 'Guten Tag und herzlich willkommen bei der Autowerkstatt Zürich! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?'"
                    )
                    session.userdata.greeting_sent = True
                except:
                    pass
        
        logger.info(f"✅ [{session_id}] Garage Agent ready!")
        
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
