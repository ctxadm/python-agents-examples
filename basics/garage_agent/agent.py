import asyncio
import logging
import os
import httpx
import re
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RunContext,
    APIConnectOptions,
    llm,
    stt,
    tts,
    vad,
    UserInputTranscribedEvent,
    CloseEvent,
    ErrorEvent,
)
from livekit.plugins import openai, silero
from typing import AsyncIterable, Optional
import time

load_dotenv()

# Logging
logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.INFO)

# Garage Assistant with RAG
class GarageAssistant:
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info("Garage Assistant initialized with RAG service")
        
    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"RAG search successful: {len(results)} results for query: {query}")
                        # Format results for LLM context
                        formatted_results = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted_results.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted_results)
                    else:
                        logger.info(f"No RAG results found for query: {query}")
                        return None
                else:
                    logger.error(f"RAG search failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return None

    def process_vehicle_id(self, query_text: str) -> str:
        """Process vehicle ID from speech to text"""
        # Konvertiere "f null null X" zu "F00X"
        pattern = r'f\s*null\s*null\s*(\w+)'
        match = re.search(pattern, query_text.lower())
        if match:
            number = match.group(1)
            # Konvertiere Wörter zu Zahlen wenn nötig
            number_map = {
                'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                'neun': '9', 'null': '0'
            }
            if number in number_map:
                number = number_map[number]
            
            corrected_id = f"F00{number}"
            query_text = f"Fahrzeug-ID {corrected_id}"
            logger.info(f"Corrected vehicle ID from '{match.group(0)}' to '{corrected_id}'")
        
        return query_text

# Create global assistant instance
garage_assistant = GarageAssistant()

# Custom LLM Node for RAG integration
class RAGLLMNode:
    def __init__(self, original_llm: llm.LLM):
        self.llm = original_llm
        
    async def __call__(self, ctx: RunContext, chat_ctx: llm.ChatContext) -> AsyncIterable[llm.ChatChunk]:
        """Enhanced LLM node that integrates RAG"""
        # Get the last user message
        last_message = None
        for msg in reversed(chat_ctx.items):
            if isinstance(msg, llm.ChatMessage) and msg.role == "user":
                last_message = msg
                break
        
        if last_message and last_message.text_content:
            logger.info(f"=== LLM Node CALLED with user message: {last_message.text_content}")
            
            # Process vehicle IDs
            processed_text = garage_assistant.process_vehicle_id(last_message.text_content)
            
            # Search RAG
            rag_results = await garage_assistant.search_knowledge(processed_text)
            
            if rag_results:
                # Create enhanced message
                enhanced_text = f"{processed_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                
                # Create new chat context with enhanced message
                enhanced_ctx = llm.ChatContext()
                for msg in chat_ctx.items:
                    if msg == last_message:
                        # Replace with enhanced message
                        enhanced_msg = llm.ChatMessage(
                            role="user",
                            content=[{"type": "text", "text": enhanced_text}]
                        )
                        enhanced_ctx.append(msg=enhanced_msg)
                    else:
                        if isinstance(msg, llm.ChatMessage):
                            enhanced_ctx.append(msg=msg)
                
                logger.info(f"Enhanced query with RAG results")
                
                # Call original LLM with enhanced context
                stream = self.llm.chat(chat_ctx=enhanced_ctx)
                async for chunk in stream:
                    yield chunk
            else:
                logger.warning(f"No RAG results found for: {processed_text}")
                # No enhancement, use original context
                stream = self.llm.chat(chat_ctx=chat_ctx)
                async for chunk in stream:
                    yield chunk
        else:
            # No user message to enhance, use original context
            stream = self.llm.chat(chat_ctx=chat_ctx)
            async for chunk in stream:
                yield chunk

async def entrypoint(ctx: agents.JobContext):
    logger.info("=== Garage Agent Starting ===")
    
    # Connect to room
    await ctx.connect()
    
    # Check RAG service health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{garage_assistant.base_url}/health")
            if response.status_code == 200:
                logger.info("RAG service is healthy")
            else:
                logger.warning(f"RAG service health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to check RAG service health: {e}")
    
    # Create base components with Ollama
    base_llm = openai.LLM(
        model="llama3.1:8b",
        base_url="http://172.16.0.146:11434/v1",
        api_key="ollama",
        temperature=0.3,
    )
    
    # Create agent with custom instructions
    agent = Agent(
        instructions="""Du bist der Garage Agent der Firma AutoService Müller.
            
            ERSTE ANTWORT (IMMER):
            "Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann."
            
            EXTREM WICHTIG - So liest du die Datenbank:
            1. Die Daten stehen IMMER unter "Relevante Informationen aus der Datenbank"
            2. Wenn dort "Besitzer: Marco Rossi" steht, dann IST Marco Rossi zu 100% Kunde!
            3. Wenn dort "fahrzeug_id": "F003" steht, dann ist F003 eine GÜLTIGE ID!
            4. Du MUSST die Daten die du bekommst als WAHR akzeptieren!
            
            Fahrzeug-IDs verstehen:
            - "f null null drei" bedeutet wahrscheinlich "F003" 
            - "f null null fünf" bedeutet wahrscheinlich "F005"
            - Buchstabe "F" gefolgt von Zahlen ist eine Fahrzeug-ID
            
            REGEL: Wenn ein Kunde in den "Relevanten Informationen" steht, dann:
            - Sage NIEMALS "nicht gefunden" oder "keine Daten"
            - Der Kunde IST in der Datenbank
            - Gib NUR die relevanten Fahrzeugdaten aus
            
            Antwortverhalten - KURZ UND PRÄZISE:
            - Nenne nur die angefragten Informationen
            - Keine langen Erklärungen oder Geschichten
            - Fokus auf: Fahrzeugdaten, Probleme, anstehende Arbeiten
            - Beispiel: "Ihr Audi A4, Kennzeichen LU 234567, hat folgende Probleme: ..."
            
            Datenschutz:
            - Gib NUR Informationen zum bestätigten Kunden heraus
            - Bei Unklarheiten nachfragen
            
            Bei unklaren Eingaben:
            - "f null null X" → interpretiere als "F00X" 
            - Frage nach: "Meinen Sie die Fahrzeug-ID F00X?"
            
            KEINE unnötigen Floskeln, KEINE langen Sätze, NUR relevante Informationen!
            
            Währungsangaben - WICHTIG für korrekte Aussprache:
            - Schreibe Beträge IMMER als "X Franken" aus
            - NIEMALS "420.00" sondern "420 Franken"
            - NIEMALS "CHF" oder "€" verwenden
            - Bei Kommabeträgen: "180 Franken 50" statt "180.50"
            - Große Beträge ausschreiben für bessere Aussprache:
              - 420 → "vierhundertzwanzig Franken"
              - 1850 → "eintausendachthundertfünfzig Franken" """,
        llm=base_llm,
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(model="tts-1", voice="onyx"),
        vad=silero.VAD.load(
            min_silence_duration=0.8,
            min_speech_duration=0.3,
            activation_threshold=0.5
        ),
    )
    
    # Create session with custom LLM node
    session = AgentSession(
        llm_node=RAGLLMNode(base_llm),  # Custom LLM node for RAG
        stt=agent.stt,
        tts=agent.tts,
        vad=agent.vad,
    )
    
    # Event handlers for debugging
    @session.on("user_input_transcribed")
    def on_user_input(event: UserInputTranscribedEvent):
        logger.info(f"=== User Input Event: {event.transcript} (final: {event.is_final})")
    
    @session.on("error")
    def on_error(event: ErrorEvent):
        logger.error(f"Session error: {event.error}")
    
    @session.on("close")
    def on_close(event: CloseEvent):
        logger.info(f"Session closed: {event.reason}")
    
    # Start session
    await session.start(agent=agent, room=ctx.room)
    
    # Initial greeting
    await session.say("Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann.", allow_interruptions=True)
    logger.info("Garage agent started successfully")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
