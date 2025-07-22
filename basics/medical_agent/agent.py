import asyncio
import logging
import os
import httpx
import re
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero
from typing import Optional
import time

load_dotenv()

# Logging
logger = logging.getLogger("medical-assistant")
logger.setLevel(logging.INFO)

# Medical Assistant with RAG
class MedicalAssistant:
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info("Medical Assistant initialized with RAG service")
        self.last_processed_message = None
        self.message_count = 0
        
    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "agent_type": "medical",
                        "top_k": 3,
                        "collection": "medical_nutrition"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"RAG search successful: {len(results)} results")
                        formatted_results = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted_results.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted_results)
                    else:
                        logger.info(f"No RAG results found")
                        return None
                else:
                    logger.error(f"RAG search failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return None

    def process_patient_id(self, query_text: str) -> str:
        """Process patient ID from speech to text"""
        pattern = r'p\s*null\s*null\s*(\w+)'
        match = re.search(pattern, query_text.lower())
        if match:
            number = match.group(1)
            number_map = {
                'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                'neun': '9', 'null': '0'
            }
            if number in number_map:
                number = number_map[number]
            
            corrected_id = f"P00{number}"
            query_text = f"Patienten-ID {corrected_id}"
            logger.info(f"Corrected patient ID to '{corrected_id}'")
        
        return query_text

# Global assistant instance
medical_assistant = None

# WORKAROUND: Monitor chat context periodically
async def monitor_chat_context(session: AgentSession, agent: Agent):
    """Workaround: Monitor chat context for new messages"""
    last_message_count = 0
    
    while True:
        try:
            # Check chat context every 0.5 seconds
            await asyncio.sleep(0.5)
            
            if hasattr(agent, 'chat_ctx') and agent.chat_ctx:
                messages = agent.chat_ctx.messages
                current_count = len(messages)
                
                # New message detected
                if current_count > last_message_count:
                    last_message = messages[-1]
                    
                    # Check if it's a user message
                    if last_message.role == "user" and last_message.content:
                        user_text = last_message.content
                        logger.info(f"=== WORKAROUND: Detected new user message: {user_text}")
                        
                        # Process with RAG
                        processed_text = medical_assistant.process_patient_id(user_text)
                        rag_results = await medical_assistant.search_knowledge(processed_text)
                        
                        if rag_results:
                            # Create enhanced response
                            enhanced_prompt = f"""Basierend auf der Anfrage: {processed_text}

Relevante Informationen aus der Datenbank:
{rag_results}

Bitte antworte auf die Anfrage unter Berücksichtigung der Datenbankinformationen."""
                            
                            # Use session.say to respond with RAG context
                            await session.say(enhanced_prompt, allow_interruptions=True)
                            logger.info("WORKAROUND: Enhanced response sent")
                        
                last_message_count = current_count
                
        except Exception as e:
            logger.error(f"Error in monitor_chat_context: {e}")
            await asyncio.sleep(1)

async def entrypoint(ctx: agents.JobContext):
    global medical_assistant
    logger.info("=== Medical Agent Starting (1.2.1 Workaround) ===")
    
    # Initialize assistant
    medical_assistant = MedicalAssistant()
    
    # Check RAG service health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{medical_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to check RAG service health: {e}")
    
    # Create agent with enhanced instructions
    instructions = """Du bist ein Agent mit Zugriff auf die Patientendatenbank.
            
ERSTE ANTWORT (IMMER):
"Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?"

EXTREM WICHTIG - So liest du die Datenbank:
1. Die Daten stehen IMMER unter "Relevante Informationen aus der Datenbank"
2. Wenn dort "Patient: Emma Fischer" steht, dann EXISTIERT Emma Fischer zu 100%!
3. Wenn dort "patient_id": "P005" steht, dann ist P005 eine GÜLTIGE ID!
4. Du MUSST die Daten die du bekommst als WAHR akzeptieren!

Patienten-IDs verstehen:
- "p null null fünf" bedeutet wahrscheinlich "P005" 
- "p null null drei" bedeutet wahrscheinlich "P003"

REGEL: Wenn ein Patient in den "Relevanten Informationen" steht, dann:
- Sage NIEMALS "nicht gefunden" oder "keine Daten"
- Der Patient IST in der Datenbank
- Gib die Informationen aus den relevanten Daten wieder

Währungsangaben:
- Schreibe Beträge IMMER als "X Franken" aus
- NIEMALS "15.50" sondern "15 Franken 50"
- Dosierungen klar aussprechen: "10mg" → "zehn Milligramm" """
    
    # Create agent
    agent = Agent(
        instructions=instructions,
        llm=openai.LLM(
            model="llama3.1:8b",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            temperature=0.7,
        ),
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(model="tts-1", voice="shimmer"),
        vad=silero.VAD.load(
            min_silence_duration=0.8,
            min_speech_duration=0.3,
            activation_threshold=0.5
        ),
    )
    
    # Create session
    session = AgentSession(agent=agent)
    
    # Connect to room
    await ctx.connect()
    
    # Start session
    await session.start(room=ctx.room)
    
    # Start the chat context monitor (WORKAROUND)
    monitor_task = asyncio.create_task(monitor_chat_context(session, agent))
    logger.info("Started chat context monitor workaround")
    
    # Initial greeting
    await session.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", allow_interruptions=True)
    logger.info("Medical agent started successfully with workaround")
    
    # Keep the agent running
    try:
        await monitor_task
    except asyncio.CancelledError:
        logger.info("Monitor task cancelled")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
