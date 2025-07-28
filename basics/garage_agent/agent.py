# LiveKit Agents - Garage Management Agent (FINAL CORRECTED VERSION)
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
    function_tool  # KORREKTER Import
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

# Einfachere Instructions für bessere Kompatibilität
LLAMA32_INSTRUCTIONS = """Du bist Pia von der Autowerkstatt Zürich.

Wenn jemand dich begrüßt oder das Gespräch beginnt, sage:
"Guten Tag und herzlich willkommen bei der Autowerkstatt Zürich! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?"

Antworte IMMER auf Deutsch.
Verwende NIE die Wörter: Entschuldigung, Sorry, Leider.
Erfinde KEINE Daten - wenn du nichts findest, sage das ehrlich.

Nutze Suchfunktionen nur bei konkreten Fragen nach Fahrzeugen, Reparaturen oder Kosten."""

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
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. Wait for audio track
        audio_track_received = False
        for i in range(10):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found")
                    audio_track_received = True
                    break
            if audio_track_received:
                break
            await asyncio.sleep(1)
        
        # 4. Configure LLM - KRITISCH: Korrekte Ollama Integration!
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # WICHTIG: Keine zusätzlichen Parameter bei with_ollama!
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0  # Absolut 0 für Konsistenz
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama (temp=0.0)")
        
        # 5. Create session
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
                min_silence_duration=0.4,
                min_speech_duration=0.15
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            )
        )
        
        # 6. Create agent
        agent = GarageAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(room=ctx.room, agent=agent)
        
        # 8. WICHTIG: Tools nach dem Start hinzufügen
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
        
        # 9. Event handlers
        @session.on("user_speech_committed")
        def on_user_speech(event):
            logger.info(f"[{session_id}] 🎤 User: {event}")
        
        @session.on("agent_speech_committed")
        def on_agent_speech(event):
            logger.info(f"[{session_id}] 🤖 Agent: {event}")
        
        # 10. AUTOMATISCHE BEGRÜSSUNG - Vereinfacht
        await asyncio.sleep(3.0)  # Wichtig: 3 Sekunden warten!
        
        logger.info(f"📢 [{session_id}] Sending automatic greeting...")
        
        try:
            # Einfachste Methode ohne spezielle Parameter
            await session.generate_reply()
            
            session.userdata.greeting_sent = True
            logger.info(f"✅ [{session_id}] Default greeting triggered!")
            
        except Exception as e:
            logger.error(f"❌ [{session_id}] Default greeting failed: {e}")
            
            # Alternative: Mit expliziter Instruction
            try:
                await session.generate_reply(
                    instructions="Begrüße den Kunden als Pia von der Autowerkstatt Zürich."
                )
                logger.info(f"✅ [{session_id}] Instruction-based greeting sent")
            except Exception as e2:
                logger.error(f"❌ [{session_id}] All greeting methods failed: {e2}")
        
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
