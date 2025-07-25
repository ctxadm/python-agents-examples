# LiveKit Agents 1.0.x - Minimal Garage Agent with Mistral
import logging
import os
import httpx
import re
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import ChatContext, ChatRole, ChatMessage
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

class GarageAgent:
    """Minimaler Garage Agent - nur Business Logic"""
    
    def __init__(self):
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.authenticated_customer = None
        logger.info(f"✅ Garage Agent initialized (RAG: {self.rag_url})")

async def entrypoint(ctx: JobContext):
    """Modern entrypoint following LiveKit 1.0.x pattern"""
    logger.info("="*50)
    logger.info("🚀 Starting Minimal Garage Agent")
    logger.info("="*50)
    
    # 1. Connect first
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # 2. Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ Participant joined: {participant.identity}")
    
    # 3. Create agent instance (just logic)
    agent = GarageAgent()
    
    # 4. Create session with all components
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="""Du bist ein Werkstatt-Assistent. 
WICHTIG: Frage ZUERST nach dem Namen des Kunden zur Authentifizierung.
Erste Antwort: "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen."
Nach Authentifizierung: Hilf mit Fahrzeugdaten und Service-Informationen."""
            )
        ]
    )
    
    session = AgentSession(
        chat_ctx=initial_ctx,
        llm=openai.LLM(
            model="mistral:v0.3",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            temperature=0.3
        ),
        stt=openai.STT(
            model="whisper-1", 
            language="de"
        ),
        tts=openai.TTS(
            model="tts-1", 
            voice="onyx"
        ),
        vad=silero.VAD.load()
    )
    
    # Function tools für Session registrieren
    @session.function_tool()
    async def authenticate_customer(customer_name: str) -> str:
        """Authentifiziert einen Kunden"""
        logger.info(f"🔐 Authenticating: {customer_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.rag_url}/search",
                    json={
                        "query": f"besitzer: {customer_name}",
                        "agent_type": "garage",
                        "top_k": 5
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    # Einfache Namenssuche
                    for result in results:
                        content = result.get("content", "")
                        if customer_name.lower() in content.lower():
                            agent.authenticated_customer = customer_name
                            logger.info(f"✅ Customer authenticated: {customer_name}")
                            return f"Willkommen {customer_name}! Wie kann ich Ihnen helfen?"
                    
                    return "Kunde nicht gefunden. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return "Authentifizierung fehlgeschlagen."
    
    @session.function_tool()
    async def search_vehicle_data(query: str) -> str:
        """Sucht Fahrzeugdaten"""
        logger.info(f"🔍 Searching: {query}")
        
        if not agent.authenticated_customer:
            return "Bitte authentifizieren Sie sich zuerst."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.rag_url}/search",
                    json={
                        "query": f"{query} {agent.authenticated_customer}",
                        "agent_type": "garage",
                        "top_k": 3
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        # Nur Ergebnisse des authentifizierten Kunden
                        relevant = []
                        for r in results:
                            content = r.get("content", "")
                            if agent.authenticated_customer.lower() in content.lower():
                                relevant.append(content)
                        
                        if relevant:
                            return "\n\n".join(relevant[:2])  # Max 2 Ergebnisse
                    
                    return "Keine Daten gefunden."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Suche fehlgeschlagen."
    
    # 5. Session starten
    logger.info("🏁 Starting session...")
    await session.start(ctx.room)
    
    # 6. Initiale Begrüßung
    await session.say("Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen zur Identifikation.")
    
    logger.info("✅ Garage Agent ready!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
