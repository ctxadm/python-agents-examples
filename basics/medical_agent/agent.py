# ========================================
# MEDICAL AGENT mit VoicePipelineAgent - FINAL
# ========================================
import os
import logging
import httpx
import json
import asyncio
import re
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

logger = logging.getLogger("medical-assistant")

class MedicalAssistant:
    """Medical Assistant with RAG integration"""
    
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting medical agent entrypoint with VoicePipelineAgent")
    
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
    
    # Create initial chat context
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(
        llm.ChatMessage(
            role="system",
            content="""Du bist ein Agent mit Zugriff auf die Patientendatenbank.
            
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
              - "+41 79 123 4567" → "plus 41... 79... 123... 45... 67"
            """
        )
    )
    
    # Custom callback für RAG-Enhancement - WIRD BEI JEDER USER NACHRICHT AUFGERUFEN!
    async def before_llm_cb(assistant, chat_ctx):
        """Called before LLM processing to enhance with RAG"""
        logger.info("=== before_llm_cb CALLED ===")
        
        # Get last user message
        user_messages = [msg for msg in chat_ctx.messages if msg.role == "user"]
        if user_messages:
            last_message = user_messages[-1]
            query_text = last_message.content
            logger.info(f"Processing user query: {query_text}")
            
            # Process patient IDs
            query_text = medical_assistant.process_patient_id(query_text)
            
            # Search RAG
            rag_results = await medical_assistant.search_knowledge(query_text)
            
            if rag_results:
                # Enhance the last message with RAG results
                enhanced_content = f"{query_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                last_message.content = enhanced_content
                logger.info(f"Enhanced query with RAG results")
            else:
                logger.warning(f"No RAG results found for: {query_text}")
        else:
            logger.warning("No user messages found in chat context")
        
        return None  # Let default LLM processing continue
    
    # Create the VoicePipelineAgent
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(
            min_silence_duration=0.8,
            min_speech_duration=0.3,
            activation_threshold=0.5
        ),
        stt=openai.STT(
            model="whisper-1",
            language="de"
        ),
        llm=openai.LLM(
            model="llama3.1:8b",
            base_url="http://172.16.0.146:11434/v1",
            api_key="ollama",
            timeout=120.0,
            temperature=0.7
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="shimmer"
        ),
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb  # DIESER CALLBACK WIRD BEI JEDER USER-NACHRICHT AUFGERUFEN!
    )
    
    # Start the assistant
    assistant.start(ctx.room)
    
    # Initial greeting
    await assistant.say("Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?", allow_interruptions=True)
    
    logger.info("Medical agent with VoicePipelineAgent started successfully")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
