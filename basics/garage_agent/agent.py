# LiveKit Agents - Garage Agent (Moderne API wie Nutrition-Agent)
import logging
import os
import httpx
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-3")

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_customer: Optional[str] = None
    rag_url: str = "http://localhost:8000"


class GarageAssistant(Agent):
    """Garage Assistant mit korrekter API-Nutzung"""
    
    def __init__(self) -> None:
        # Instructions optimiert für Llama 3.2
        super().__init__(instructions="""You are a helpful garage assistant at an auto repair shop.

Your first response must be: "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen."

After the customer provides their name, use the authenticate_customer function silently.
After authentication, help with vehicle data and service information.
When asked about vehicles, use the search_vehicle_data function.

Important: Always respond in German. Never mention functions or technical details to the customer.""")
        logger.info("✅ GarageAssistant initialized")

    @function_tool
    async def authenticate_customer(self, 
                                  context: RunContext[GarageUserData],
                                  customer_name: str) -> str:
        """
        Authentifiziert einen Kunden in der Werkstattdatenbank.
        
        Args:
            customer_name: Der Name des Kunden
        """
        logger.info(f"🔐 Authenticating: {customer_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": f"besitzer: {customer_name}",
                        "agent_type": "garage",
                        "top_k": 5,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    for result in results:
                        content = result.get("content", "")
                        if customer_name.lower() in content.lower():
                            context.userdata.authenticated_customer = customer_name
                            logger.info(f"✅ Customer authenticated: {customer_name}")
                            return f"Guten Tag {customer_name}! Schön Sie wieder bei uns zu sehen. Wie kann ich Ihnen heute helfen?"
                    
                    return "Entschuldigung, ich konnte Sie in unserem System nicht finden. Könnten Sie Ihren Namen bitte noch einmal nennen?"
                    
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return "Es tut mir leid, es gab ein technisches Problem. Bitte versuchen Sie es noch einmal."

    @function_tool
    async def search_vehicle_data(self,
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Fahrzeugdaten in der Werkstattdatenbank.
        
        Args:
            query: Die Suchanfrage (z.B. "Tesla", "Service", "Kilometerstand")
        """
        logger.info(f"🔍 Searching: {query}")
        
        if not context.userdata.authenticated_customer:
            return "Bitte nennen Sie zuerst Ihren Namen zur Authentifizierung."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": f"{query} {context.userdata.authenticated_customer}",
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        relevant = []
                        for r in results:
                            content = r.get("content", "")
                            if context.userdata.authenticated_customer.lower() in content.lower():
                                relevant.append(content)
                        
                        if relevant:
                            return "\n\n".join(relevant[:2])
                    
                    return "Zu Ihrer Anfrage konnte ich leider keine Daten finden."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Die Suche ist momentan nicht verfügbar. Bitte versuchen Sie es später noch einmal."


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point mit moderner API wie im Nutrition-Agent"""
    logger.info("="*50)
    logger.info("🚀 Starting Garage Agent (Modern API)")
    logger.info("="*50)
    
    # 1. Connect FIRST (wie im Nutrition-Agent!)
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # 2. Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ Participant joined: {participant.identity}")
    
    # 3. LLM-Konfiguration
    use_gpt = os.getenv("USE_GPT", "false").lower() == "true"
    rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
    
    if use_gpt:
        llm = openai.LLM(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        logger.info("🤖 Using GPT-3.5-turbo")
    else:
        # Llama 3.2 Konfiguration
        llm = openai.LLM(
            model="llama3.2:latest",  # Geändert zu Llama 3.2
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3
        )
        logger.info("🤖 Using Llama 3.2 via Ollama")
    
    # 4. Create session with userdata (wie im Nutrition-Agent!)
    session = AgentSession[GarageUserData](
        userdata=GarageUserData(
            authenticated_customer=None,
            rag_url=rag_url
        ),
        llm=llm,
        vad=silero.VAD.load(),
        stt=openai.STT(
            model="whisper-1",
            language="de"
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="onyx"
        )
    )
    
    # 5. Create agent instance
    agent = GarageAssistant()
    
    # 6. Start session
    logger.info("🏁 Starting session...")
    await session.start(
        room=ctx.room,
        agent=agent
    )
    
    logger.info("✅ Garage Agent ready and listening!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
