# ========================================
# GARAGE AGENT mit VoicePipelineAgent - FINAL
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

logger = logging.getLogger("garage-assistant")

class GarageAssistant:
    """Garage Assistant with RAG integration"""
    
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting garage agent entrypoint with VoicePipelineAgent")
    
    # Initialize assistant
    garage_assistant = GarageAssistant()
    
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
    
    # Create initial chat context
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(
        llm.ChatMessage(
            role="system",
            content="""Du bist der Garage Agent der Firma AutoService Müller.
            
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
              - 1850 → "eintausendachthundertfünfzig Franken"
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
            
            # Process vehicle IDs
            query_text = garage_assistant.process_vehicle_id(query_text)
            
            # Search RAG
            rag_results = await garage_assistant.search_knowledge(query_text)
            
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
            temperature=0.3  # Niedriger für präzisere Antworten
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="onyx"
        ),
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb  # DIESER CALLBACK WIRD BEI JEDER USER-NACHRICHT AUFGERUFEN!
    )
    
    # Start the assistant
    assistant.start(ctx.room)
    
    # Initial greeting
    await assistant.say("Guten Tag, hier ist der Garage Agent der AutoService Müller. Bitte nennen Sie mir Ihren Namen, damit ich Ihnen weiterhelfen kann.", allow_interruptions=True)
    
    logger.info("Garage agent with VoicePipelineAgent started successfully")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
