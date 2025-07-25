# LiveKit Agents 1.0.x - Garage Agent mit moderner API und allen Fixes
import logging
import os
import httpx
import re
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-3")

class GarageAgent(Agent):
    """Garage Agent - nur Business Logic (moderne API)"""
    
    def __init__(self):
        # NUR Instructions, keine Komponenten!
        super().__init__(
            instructions="""Du bist ein Werkstatt-Assistent für eine KFZ-Werkstatt.

WICHTIGE REGELN:
1. Frage ZUERST nach dem Namen des Kunden zur Authentifizierung
2. Deine erste Antwort MUSS sein: "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen."
3. Nach erfolgreicher Authentifizierung hilfst du mit Fahrzeugdaten und Service-Informationen

KRITISCH - NIEMALS AUSSPRECHEN:
- Sprich NIEMALS technische Anweisungen oder Kommentare aus
- Erwähne NIEMALS Function-Calls, JSON, oder technische Details
- Sage NIEMALS Dinge wie "Wartet auf Antwort" oder ähnliche Meta-Kommentare
- Antworte NUR mit natürlicher, menschlicher Sprache

Denke daran: Du bist ein freundlicher Werkstatt-Mitarbeiter, kein Computer!"""
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.authenticated_customer = None
        logger.info(f"✅ Garage Agent initialized (RAG: {self.rag_url})")
    
    @function_tool
    async def authenticate_customer(self, customer_name: str) -> str:
        """Authentifiziert einen Kunden"""
        logger.info(f"🔐 Authenticating: {customer_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": f"besitzer: {customer_name}",
                        "agent_type": "garage",
                        "top_k": 5,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    # Einfache Namenssuche
                    for result in results:
                        content = result.get("content", "")
                        if customer_name.lower() in content.lower():
                            self.authenticated_customer = customer_name
                            logger.info(f"✅ Customer authenticated: {customer_name}")
                            return f"Willkommen {customer_name}! Wie kann ich Ihnen helfen?"
                    
                    return "Kunde nicht gefunden. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return "Authentifizierung fehlgeschlagen."
    
    @function_tool
    async def search_vehicle_data(self, query: str) -> str:
        """Sucht Fahrzeugdaten"""
        logger.info(f"🔍 Searching: {query}")
        
        if not self.authenticated_customer:
            return "Bitte authentifizieren Sie sich zuerst."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": f"{query} {self.authenticated_customer}",
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        # Nur Ergebnisse des authentifizierten Kunden
                        relevant = []
                        for r in results:
                            content = r.get("content", "")
                            if self.authenticated_customer.lower() in content.lower():
                                relevant.append(content)
                        
                        if relevant:
                            return "\n\n".join(relevant[:2])  # Max 2 Ergebnisse
                    
                    return "Keine Daten gefunden."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Suche fehlgeschlagen."


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment - akzeptiert ALLE Jobs"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    
    # IMMER AKZEPTIEREN - kein Hash-Check!
    logger.info(f"[{AGENT_NAME}] ✅ ACCEPTING job (hash-check disabled)")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Modern entry point mit korrekter Session-Initialisierung"""
    logger.info("="*50)
    logger.info("🚀 Starting Garage Agent (Modern API)")
    logger.info("="*50)
    
    # 1. Connect
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # 2. Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ Participant joined: {participant.identity}")
    
    # 3. Create agent
    agent = GarageAgent()
    
    # 4. LLM-Konfiguration mit Fallback
    use_gpt = os.getenv("USE_GPT", "false").lower() == "true"
    
    if use_gpt:
        # GPT-3.5 Konfiguration (funktioniert garantiert)
        llm = openai.LLM(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        logger.info("🤖 Using GPT-3.5-turbo")
    else:
        # Mistral Konfiguration (lokal)
        llm = openai.LLM(
            model="mistral:v0.3",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3
        )
        logger.info("🤖 Using Mistral via Ollama")
    
    # 5. Create session mit ALLEN Komponenten
    session = AgentSession(
        stt=openai.STT(
            model="whisper-1", 
            language="de"
        ),
        llm=llm,  # Dynamisch gewähltes LLM
        tts=openai.TTS(
            model="tts-1", 
            voice="onyx"
        ),
        vad=silero.VAD.load()
    )
    
    # 6. Start session
    logger.info("🏁 Starting session...")
    await session.start(
        room=ctx.room,
        agent=agent
    )
    
    logger.info("✅ Garage Agent ready and listening!")


if __name__ == "__main__":
    # Worker mit request_handler für Multi-Worker Setup
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_handler=request_handler  # Wichtig für Job-Akzeptanz!
        )
    )
