# LiveKit Agents - Garage Management Agent (Llama 3.2 Optimized)
# Für LiveKit Agents Version 1.0.23
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
    llm
)
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai as lkopenai
from livekit.plugins import silero

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("garage-agent")

# OpenAI Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Data models
@dataclass
class GarageUserData:
    name: str
    phone: Optional[str] = None
    car_number: Optional[str] = None
    repair_history: List[Dict] = None
    current_repair: Optional[Dict] = None
    invoices: List[Dict] = None
    greeting_sent: bool = False
    
    def __post_init__(self):
        if self.repair_history is None:
            self.repair_history = []
        if self.invoices is None:
            self.invoices = []

# Mock database with Swiss/German data
MOCK_DATABASE = {
    "customers": {
        "thomas müller": GarageUserData(
            name="Thomas Müller",
            phone="+41 79 123 45 67",
            car_number="ZH 123456",
            repair_history=[
                {
                    "date": "2024-12-15",
                    "description": "Winterreifen-Wechsel",
                    "cost": 180,
                    "status": "abgeschlossen"
                },
                {
                    "date": "2024-11-20",
                    "description": "Ölwechsel und Service",
                    "cost": 450,
                    "status": "abgeschlossen"
                }
            ],
            current_repair={
                "order_number": "REP-2025-0142",
                "date": "2025-01-28",
                "description": "Bremsen-Service vorne und hinten",
                "estimated_cost": 850,
                "status": "in Bearbeitung",
                "completion": "2025-01-29"
            }
        ),
        "anna schmidt": GarageUserData(
            name="Anna Schmidt",
            phone="+41 78 987 65 43",
            car_number="BE 789012",
            repair_history=[
                {
                    "date": "2024-10-10",
                    "description": "MFK-Vorbereitung",
                    "cost": 320,
                    "status": "abgeschlossen"
                }
            ],
            current_repair={
                "order_number": "REP-2025-0143",
                "date": "2025-01-27",
                "description": "Auspuff-Reparatur",
                "estimated_cost": 620,
                "status": "Warten auf Ersatzteil",
                "completion": "2025-01-31"
            }
        )
    }
}

# Optimized instructions for Llama 3.2 - NOCH KLARER
LLAMA32_OPTIMIZED_INSTRUCTIONS = """Du bist Pia von der Autowerkstatt Zürich. Antworte IMMER auf Deutsch.

ERSTE REGEL - BEGRÜSSUNG:
Wenn jemand "Hallo" sagt, antworte GENAU SO:
"Hallo! Ich bin Pia von der Autowerkstatt Zürich. Wie kann ich Ihnen heute helfen?"

VERBOTENE WÖRTER (NIE verwenden):
- Entschuldigung
- Sorry  
- Leider
- Es tut mir leid

FUNKTIONEN:
Nutze Suchfunktionen NUR bei konkreten Fragen nach:
- Kundendaten
- Reparaturstatus
- Rechnungen

NICHT bei Begrüßungen!

WENN KEINE DATEN GEFUNDEN:
"Ich habe keine Daten gefunden. Bitte nennen Sie mir Ihren Namen oder Ihr Kennzeichen."

ERFINDE NIEMALS Daten!"""

class GarageAgent(Agent):
    """Garage Assistant für die Autowerkstatt Zürich"""
    
    def __init__(self) -> None:
        super().__init__(instructions=LLAMA32_OPTIMIZED_INSTRUCTIONS)
        logger.info("✅ GarageAgent initialized for Llama 3.2")

async def search_customer_data(query: str) -> str:
    """Search for customer data in the garage database."""
    logger.info(f"🔍 Searching customer data for: {query}")
    
    # GUARD: Prevent searches for greetings
    greeting_words = ["hallo", "hi", "guten tag", "grüezi", "hey", "servus", "morgen"]
    if any(word in query.lower() for word in greeting_words) or query.strip() == "":
        logger.warning(f"⚠️ Ignoring greeting search: {query}")
        return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder Ihre Autonummer, damit ich Ihre Kundendaten finden kann."
    
    query_lower = query.lower().strip()
    
    # Search in mock database
    for key, customer in MOCK_DATABASE["customers"].items():
        if (query_lower in key or 
            query_lower in customer.name.lower() or
            (customer.phone and query_lower in customer.phone) or
            (customer.car_number and query_lower in customer.car_number.lower())):
            
            result = f"Kundendaten gefunden:\n"
            result += f"Name: {customer.name}\n"
            result += f"Telefon: {customer.phone}\n"
            result += f"Autonummer: {customer.car_number}\n"
            
            if customer.current_repair:
                result += f"\nAktueller Auftrag:\n"
                result += f"- Auftragsnummer: {customer.current_repair['order_number']}\n"
                result += f"- Arbeiten: {customer.current_repair['description']}\n"
                result += f"- Status: {customer.current_repair['status']}\n"
                result += f"- Geschätzte Kosten: {customer.current_repair['estimated_cost']} Franken\n"
                result += f"- Voraussichtliche Fertigstellung: {customer.current_repair['completion']}"
            
            return result
    
    return "Keine Kundendaten gefunden. Bitte überprüfen Sie die Eingabe oder geben Sie weitere Informationen an."

async def search_repair_status(query: str) -> str:
    """Search for repair status and orders."""
    logger.info(f"🔧 Searching repair status for: {query}")
    
    query_lower = query.lower().strip()
    
    # Search for customer or order number
    if query_lower.startswith("rep-"):
        # Direct order number search
        for customer in MOCK_DATABASE["customers"].values():
            if customer.current_repair and customer.current_repair["order_number"].lower() == query_lower:
                return f"Auftrag {customer.current_repair['order_number']}:\n" \
                       f"- Kunde: {customer.name}\n" \
                       f"- Arbeiten: {customer.current_repair['description']}\n" \
                       f"- Status: {customer.current_repair['status']}\n" \
                       f"- Fertigstellung: {customer.current_repair['completion']}"
    
    # Search by customer name
    for key, customer in MOCK_DATABASE["customers"].items():
        if query_lower in key or query_lower in customer.name.lower():
            if customer.current_repair:
                return f"Reparaturstatus für {customer.name}:\n" \
                       f"- Auftragsnummer: {customer.current_repair['order_number']}\n" \
                       f"- Arbeiten: {customer.current_repair['description']}\n" \
                       f"- Status: {customer.current_repair['status']}\n" \
                       f"- Fertigstellung: {customer.current_repair['completion']}"
            else:
                return f"Keine aktuelle Reparatur für {customer.name} gefunden."
    
    return "Keine Reparaturaufträge gefunden. Bitte geben Sie einen Kundennamen oder eine Auftragsnummer an."

async def search_invoice_data(query: str) -> str:
    """Search for invoice information."""
    logger.info(f"📄 Searching invoice data for: {query}")
    
    query_lower = query.lower().strip()
    
    for key, customer in MOCK_DATABASE["customers"].items():
        if query_lower in key or query_lower in customer.name.lower():
            if customer.repair_history:
                result = f"Rechnungshistorie für {customer.name}:\n"
                for repair in customer.repair_history:
                    result += f"\n- Datum: {repair['date']}\n"
                    result += f"  Arbeiten: {repair['description']}\n"
                    result += f"  Betrag: {repair['cost']} Franken\n"
                    result += f"  Status: {repair['status']}\n"
                return result
            else:
                return f"Keine Rechnungen für {customer.name} gefunden."
    
    return "Keine Rechnungsdaten gefunden. Bitte geben Sie einen Kundennamen an."

async def request_handler(ctx: JobContext) -> None:
    """Handle incoming job requests"""
    logger.info(f"🎯 Accepting job: {ctx}")
    await ctx.accept()

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info(f"🚀 Starting Garage Agent for room: {ctx.room.name}")
    
    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ Connected to room: {ctx.room.name}")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"👤 Participant joined: {participant.identity}")
        
        # 3. Configure Ollama with Llama 3.2 - ULTRA OPTIMIZED SETTINGS
        llm_config = lkopenai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,  # CRITICAL: Absolute zero for consistency
            top_p=0.1,        # Extremely focused
            top_k=1,          # Only the most probable token
            repetition_penalty=1.15,  # Strong penalty against repetition
            seed=42           # Fixed seed for reproducibility
        )
        
        logger.info(f"🤖 Using Llama 3.2 with optimized settings (temp=0.0)")
        
        # 4. Configure tools
        tools = [
            llm.FunctionCallInfo(
                name="search_customer_data",
                description="Sucht nach Kundendaten. Nutze diese Funktion NUR bei konkreten Anfragen, NICHT bei Begrüßungen.",
                callable=search_customer_data
            ),
            llm.FunctionCallInfo(
                name="search_invoice_data",
                description="Sucht nach Rechnungsinformationen.",
                callable=search_invoice_data
            ),
            llm.FunctionCallInfo(
                name="search_repair_status",
                description="Sucht nach Reparaturstatus.",
                callable=search_repair_status
            )
        ]
        
        # 5. Create Agent
        agent = GarageAgent()
        
        # 6. Create Session with all components
        session = AgentSession(
            userdata=GarageUserData(
                name="",
                greeting_sent=False
            ),
            llm=llm_config,
            vad=silero.VAD.load(),
            stt=lkopenai.STT(
                api_key=API_KEY,
                model="whisper-1",
                language="de"
            ),
            tts=lkopenai.TTS(
                api_key=API_KEY,
                model="tts-1",
                voice="nova"
            )
        )
        
        # 7. Start session
        await session.start(agent=agent, room=ctx.room)
        session.llm.function_tools = tools  # Add tools after session start
        
        logger.info(f"✅ Session started successfully")
        
        # 8. CRITICAL: Send automatic greeting
        # WICHTIG: Längerer Delay für Llama 3.2 und Session-Initialisierung
        await asyncio.sleep(3.0)
        
        logger.info(f"📢 Sending automatic greeting...")
        
        # Verwende generate_reply als primäre Methode für v1.0.23
        try:
            # Direkte, klare Anweisungen für Llama 3.2
            await session.generate_reply(
                instructions=(
                    "Sage EXAKT diesen Text: "
                    "'Guten Tag und herzlich willkommen bei der Autowerkstatt Zürich! "
                    "Ich bin Pia, Ihre digitale Assistentin. "
                    "Wie kann ich Ihnen heute bei Ihrem Fahrzeug helfen?' "
                    "NICHTS ANDERES. Keine Variation."
                )
            )
            
            session.userdata.greeting_sent = True
            logger.info(f"✅ Greeting via generate_reply sent successfully")
            
        except Exception as e:
            logger.error(f"❌ generate_reply failed: {e}")
            
            # Alternative: Verwende user_input um eine Antwort zu triggern
            try:
                await session.generate_reply(
                    user_input="Hallo",
                    instructions="Antworte mit der Standard-Begrüßung als Pia von der Autowerkstatt Zürich."
                )
                logger.info(f"✅ Alternative greeting triggered")
            except Exception as e2:
                logger.error(f"❌ Both greeting methods failed: {e2}")
        
        # 9. Event handlers
        @session.on("user_speech_committed")
        def on_user_speech(event):
            logger.info(f"🎤 User: {event}")
        
        @session.on("agent_speech_committed") 
        def on_agent_speech(event):
            logger.info(f"🤖 Agent: {event}")
            # Check for hallucinations
            if isinstance(event, str) and ("LU 234567" in event or "Entschuldigung" in event):
                logger.warning(f"⚠️ HALLUCINATION DETECTED: {event}")
        
        @session.on("function_calls_finished")
        def on_function_calls(event):
            logger.info(f"🛠️ Function calls finished: {event}")
        
        logger.info(f"✅ Garage Agent ready with Llama 3.2 optimizations!")
        
        # Keep the session running
        # The session will automatically handle the conversation
        
    except Exception as e:
        logger.error(f"❌ Fatal error in agent: {e}", exc_info=True)
        raise

def run_agent():
    """Run the agent with proper worker options"""
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,
        port=8080,
        host="0.0.0.0"
    )
    
    logger.info("🏭 Starting Garage Agent Worker (Llama 3.2 Optimized for v1.0.23)...")
    cli.run_app(worker_options)

if __name__ == "__main__":
    run_agent()
