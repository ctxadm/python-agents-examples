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
    SessionConnectOptions,
)
from livekit.plugins import openai, silero
from typing import AsyncIterable, Optional
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

    def process_patient_id(self, query_text: str) -> str:
        """Process patient ID from speech to text"""
        # Konvertiere "p null null X" zu "P00X"
        pattern = r'p\s*null\s*null\s*(\w+)'
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
            
            corrected_id = f"P00{number}"
            query_text = f"Patienten-ID {corrected_id}"
            logger.info(f"Corrected patient ID from '{match.group(0)}' to '{corrected_id}'")
        
        return query_text

# Create global assistant instance
medical_assistant = MedicalAssistant()

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
            
            # Process patient IDs
            processed_text = medical_assistant.process_patient_id(last_message.text_content)
            
            # Search RAG
            rag_results = await medical_assistant.search_knowledge(processed_text)
            
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
    logger.info("=== Medical Agent Starting ===")
    
    # Connect to room
    await ctx.connect()
    
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
    
    # Create base components with Ollama
    base_llm = openai.LLM(
        model="llama3.1:8b",
        base_url="http://172.16.0.146:11434/v1",
        api_key="ollama",
        temperature=0.7,
    )
    
    # Create agent with custom instructions
    agent = Agent(
        instructions="""Du bist ein Agent mit Zugriff auf die Patientendatenbank.
            
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
            - Buchstabe "P" gefolgt von Zahlen ist eine Patienten-ID
            
            REGEL: Wenn ein Patient in den "Relevanten Informationen" steht, dann:
            - Sage NIEMALS "nicht gefunden" oder "keine Daten"
            - Der Patient IST in der Datenbank
            - Gib die Informationen aus den relevanten Daten wieder
            
            Datenschutz:
            - Frage nach dem Namen des Patienten oder der Patienten-ID
            - Gib NUR Informationen zum bestätigten Patienten heraus
            - Antworte professionell und präzise
            
            Bei unklaren Eingaben:
            - "p null null X" → interpretiere als "P00X" 
            - Frage nach: "Meinen Sie die Patienten-ID P00X, Herr Doktor?"
            
            Nenne dich selbst nur "Agent" und duze niemals.
            
            Währungsangaben - WICHTIG für korrekte Aussprache:
            - Schreibe Beträge IMMER als "X Franken" aus
            - NIEMALS "15.50" sondern "15 Franken 50"
            - NIEMALS "CHF" oder "€" verwenden
            - Dosierungen klar aussprechen:
              - "10mg" → "zehn Milligramm"
              - "200µg" → "zweihundert Mikrogramm"
              - "5ml" → "fünf Milliliter"
            - Telefonnummern mit Pausen:
              - "+41 79 123 4567" → "plus 41... 79... 123... 45... 67" """,
        llm=base_llm,
        stt=openai.STT(model="whisper-1", language="de"),
        tts=openai.TTS(model="tts-1", voice="shimmer"),
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
        input_audio_options=SessionConnectOptions(
            room_input_options=rtc.RoomInputOptions(
                auto_subscribe=rtc.AutoSubscribe.ALL,
                subscribe_all=True,
            ),
        ),
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
    await session.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", allow_interruptions=True)
    logger.info("Medical agent started successfully")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
