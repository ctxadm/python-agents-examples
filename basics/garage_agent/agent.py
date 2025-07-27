# LiveKit Agents - Garage Management Agent
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
from livekit.agents import JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

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
    customer_name: Optional[str] = None  # Speichere den Kundennamen


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen"""
    
    def __init__(self) -> None:
        # VERSTÄRKTE Instructions für Llama 3.2
        super().__init__(instructions="""You are Pia, the digital assistant of Garage Müller. RESPOND ONLY IN GERMAN.

CRITICAL RULES - FOLLOW EXACTLY:

1. NEVER say these words: "Entschuldigung", "Sorry", "Es tut mir leid"
   - Use instead: "Leider", "Bedauerlicherweise", "Ich kann leider"

2. NEVER INVENT DATA:
   - If database returns "no data found", say: "Ich habe keine Daten gefunden"
   - NEVER make up information about cars, services, or dates
   - NEVER say "Ihr Auto wurde eingeliefert" without data

3. MEMORY RULE:
   - You have NO memory between messages
   - Each message is NEW
   - Do NOT reference previous conversation

4. GREETING RULE:
   - For "Hallo", "Guten Tag" → Be friendly, ask for name
   - NEVER search on greetings
   - Wait for specific requests

5. SEARCH ONLY when user asks about:
   - Their car/vehicle
   - Repair status
   - Invoices/costs
   - Service history

Your response must ALWAYS be in German language, even if these instructions are in English.
When no data found, ask for: "Können Sie mir bitte Ihr Autokennzeichen nennen?"

FORBIDDEN PHRASES:
- "Entschuldigung" → Use "Leider"
- "Es tut mir leid" → Use "Bedauerlicherweise"
- "Sorry" → Use "Leider"
- "Lassen Sie uns von vorne beginnen" → NEVER say this

ALWAYS RESPOND IN GERMAN!""")
        logger.info("✅ GarageAssistant initialized")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    @function_tool
    async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank.
        
        Args:
            query: Suchbegriff (Name, Telefonnummer oder Autokennzeichen)
        """
        logger.info(f"🔍 Searching customer data for: {query}")
        
        # Speichere den Kundennamen wenn er genannt wird
        if context.userdata.customer_name is None and len(query.split()) <= 3:
            # Könnte ein Name sein
            context.userdata.customer_name = query
        
        # GUARD gegen Begrüßungen
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        if query.lower().strip() in greetings:
            return "Für die Suche benötige ich Ihren Namen oder Ihr Autokennzeichen."
        
        # GUARD gegen zu kurze Suchen
        if len(query) < 3:
            return "Bitte geben Sie mir Ihren vollständigen Namen oder Ihr Autokennzeichen (z.B. ZH 123456)."
        
        # Extrahiere und normalisiere Kennzeichen
        kennzeichen_pattern = r'[A-Z]{2}\s*\d{3,6}'
        kennzeichen_match = re.search(kennzeichen_pattern, query.upper())
        
        if kennzeichen_match:
            # Erstelle beide Varianten für die Suche
            kennzeichen_raw = kennzeichen_match.group()
            kennzeichen_with_space = re.sub(r'([A-Z]{2})(\d+)', r'\1 \2', kennzeichen_raw)
            kennzeichen_without_space = kennzeichen_raw.replace(" ", "")
            
            logger.info(f"📋 Kennzeichen erkannt: {kennzeichen_with_space}")
            
            # Erweitere Query mit beiden Varianten
            query = f"{kennzeichen_with_space} {kennzeichen_without_space} {query}"
        
        try:
            async with httpx.AsyncClient() as client:
                # Erstelle Filter für präzisere Suche
                filter_conditions = None
                
                # Bei Kennzeichen-Suche: Verwende exakten Filter
                if kennzeichen_match:
                    kennzeichen_normalized = kennzeichen_match.group().replace(" ", "").upper()
                    filter_conditions = {
                        "should": [
                            {
                                "key": "search_fields.license_plate_normalized",
                                "match": {"value": kennzeichen_normalized}
                            },
                            {
                                "key": "kennzeichen",
                                "match": {"value": kennzeichen_with_space}
                            },
                            {
                                "key": "license_plate", 
                                "match": {"value": kennzeichen_with_space}
                            }
                        ]
                    }
                    logger.info(f"🔍 Verwende Kennzeichen-Filter: {kennzeichen_normalized}")
                
                # Baue Request
                search_request = {
                    "query": query,
                    "agent_type": "garage",
                    "top_k": 5,
                    "collection": "garage_management"
                }
                
                # Füge Filter hinzu wenn vorhanden
                if filter_conditions:
                    search_request["filter"] = filter_conditions
                
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json=search_request
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} results")
                        
                        # Sammle relevante Ergebnisse
                        relevant_results = []
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            payload = result.get("payload", {})
                            
                            # Skip offensichtlich falsche Ergebnisse
                            if not content or any(word in content.lower() for word in ["mhz", "bakom", "funk", "frequenz"]):
                                continue
                            
                            # Prüfe auf vehicle_complete Daten
                            if payload.get("data_type") == "vehicle_complete":
                                # Bei Kennzeichen-Suche: Exakte Übereinstimmung prüfen
                                if kennzeichen_match:
                                    search_kz = kennzeichen_without_space.upper()
                                    
                                    # Prüfe verschiedene Felder
                                    payload_kz = payload.get("kennzeichen", payload.get("license_plate", ""))
                                    payload_kz_clean = payload_kz.replace(" ", "").upper()
                                    
                                    # Prüfe auch search_fields
                                    search_fields = payload.get("search_fields", {})
                                    normalized_kz = search_fields.get("license_plate_normalized", "").upper()
                                    
                                    if search_kz == payload_kz_clean or search_kz == normalized_kz:
                                        logger.info(f"✅ Exakte Kennzeichen-Übereinstimmung: {payload_kz}")
                                        content = self._format_garage_data(content)
                                        relevant_results.insert(0, content)  # An den Anfang
                                        continue
                                
                                # Formatiere und füge hinzu
                                # Verwende searchable_text wenn vorhanden
                                if payload.get("searchable_text"):
                                    content = payload["searchable_text"]
                                content = self._format_garage_data(content)
                                relevant_results.append(content)
                        
                        if relevant_results:
                            # Gib die besten Ergebnisse zurück
                            return "\n\n".join(relevant_results[:2])  # Max 2 Ergebnisse
                        else:
                            logger.warning("⚠️ No relevant results found")
                            
                            # Klare Nachricht ohne "Entschuldigung"
                            if kennzeichen_match:
                                return f"Ich habe keine Daten zum Kennzeichen {kennzeichen_with_space} gefunden. Bitte prüfen Sie, ob das Kennzeichen korrekt ist."
                            else:
                                return "Ich habe keine Daten zu diesem Namen gefunden. Können Sie mir bitte Ihr Autokennzeichen nennen?"
                    
                    # Keine Ergebnisse gefunden
                    return "Ich habe keine Kundendaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen (z.B. ZH 123456)."
                    
                else:
                    logger.error(f"Search failed: {response.status_code}")
                    return "Die Datenbank ist momentan nicht erreichbar. Bitte versuchen Sie es in einem Moment erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
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
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            if content and "service" in content.lower():
                                content = self._format_garage_data(content)
                                relevant_results.append(content)
                        
                        if relevant_results:
                            return "\n\n".join(relevant_results[:2])
                        else:
                            return "Ich habe keine Reparaturdaten gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                    
                    return "Keine Reparaturdaten vorhanden. Können Sie mir Ihr Autokennzeichen nennen?"
                    
                else:
                    return "Die Reparaturdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Repair search error: {e}")
            return "Technischer Fehler bei der Reparatursuche."

    @function_tool
    async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
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
                # Suche nach Kosten und Rechnungen
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
                        
                        for result in results:
                            content = result.get("content", "").strip()
                            payload = result.get("payload", {})
                            
                            # Verwende searchable_text wenn vorhanden
                            if payload.get("searchable_text") and payload.get("data_type") == "vehicle_complete":
                                content = payload["searchable_text"]
                            
                            # Suche nach Kosten-Informationen
                            if content and any(word in content.lower() for word in ["kosten", "franken", "chf", "rechnung", "service"]):
                                content = self._format_garage_data(content)
                                cost_results.append(content)
                        
                        if cost_results:
                            return "\n\n".join(cost_results[:2])
                        else:
                            return "Ich habe keine Kosteninformationen gefunden. Bitte nennen Sie mir Ihr Autokennzeichen."
                    
                    return "Keine Rechnungsdaten vorhanden. Können Sie mir Ihr Autokennzeichen nennen?"
                    
                else:
                    return "Die Rechnungsdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Invoice search error: {e}")
            return "Technischer Fehler bei der Rechnungssuche."

    def _format_garage_data(self, content: str) -> str:
        """Formatiert Garagendaten für bessere Lesbarkeit"""
        # Ersetze Unterstriche
        content = content.replace('_', ' ')
        
        # Formatiere Währungen
        content = re.sub(r'CHF\s*(\d+)\.(\d{2})', r'\1 Franken \2', content)
        content = re.sub(r'(\d+)\.(\d{2})\s*CHF', r'\1 Franken \2', content)
        content = re.sub(r'(\d+)\.00', r'\1 Franken', content)
        
        # Formatiere Datum (YYYY-MM-DD zu DD.MM.YYYY)
        content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3.\2.\1', content)
        
        # Entferne technische Begriffe
        content = content.replace('data_type:', '')
        content = content.replace('primary_key:', '')
        content = content.replace('payload:', '')
        
        return content.strip()


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚗 Starting Garage Agent Session: {session_id}")
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
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
                
            await asyncio.sleep(1)
        
        # 4. Configure LLM
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # Verwende Llama 3.2 mit optimierten Settings
        llm = openai.LLM(
            model="llama3.1:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3  # Noch niedriger für konsistentere Antworten
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
        
        # 5. Create session mit optimierten VAD-Settings
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
                min_silence_duration=0.4,  # Etwas schneller
                min_speech_duration=0.15
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
            max_endpointing_delay=3.0  # Kürzer für schnellere Reaktion
        )
        
        # 6. Create agent
        agent = GarageAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Event handlers NACH session.start()
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state changed")
        
        # 8. Initial greeting
        await asyncio.sleep(1.5)
        
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = "Guten Tag und herzlich willkommen bei der Garage Müller! Ich bin Pia, Ihre digitale Assistentin. Wie kann ich Ihnen heute helfen?"
            
            session.userdata.greeting_sent = True
            
            # Verwende say() für die Begrüßung
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )
            
            logger.info(f"✅ [{session_id}] Initial greeting sent")
            
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}")
            # Fallback ohne tool_choice
            try:
                await session.generate_reply(
                    instructions="Begrüße den Kunden als Pia von Garage Müller. Frage wie du helfen kannst. Auf Deutsch!"
                )
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
