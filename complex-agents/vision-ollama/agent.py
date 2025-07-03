## This is a basic example of how to use function calling.
## To test the function, you can ask the agent to print to the console!
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import FunctionContext, FunctionArgInfo
from livekit.plugins import deepgram, openai, silero
from livekit.agents.voice_assistant import VoiceAssistant
import aiohttp
import json
from typing import Optional, List, Dict

logger = logging.getLogger("vision-agent-ollama")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")

# RAG Service Konfiguration
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

# RAG Search Function
async def search_knowledge(
    ctx: FunctionContext,
    query: str,
    collection: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    Search the knowledge base for information.
    This function searches through company documents, PDFs, and other indexed content.
    
    Args:
        query: The search query (e.g., "bakom", "funkkonzession", "radio license")
        collection: Optional specific collection to search in. If not specified, will use default mapping.
        top_k: Number of results to return (default: 5)
    
    Returns:
        Formatted search results from the knowledge base
    """
    logger.info(f"RAG Search triggered: query='{query}', collection='{collection}', top_k={top_k}")
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": query,
                "top_k": top_k
            }
            
            # Wenn eine spezifische Collection angegeben wurde
            if collection:
                payload["collection"] = collection
            else:
                # Automatische Collection-Auswahl basierend auf Keywords
                query_lower = query.lower()
                if any(word in query_lower for word in ["bakom", "funk", "konzession", "radio", "frequenz", "schweiz"]):
                    payload["collection"] = "scrape-data"
                elif any(word in query_lower for word in ["qdrant", "vector", "database", "search"]):
                    payload["collection"] = "Qdrant-Documentation"
                else:
                    # Default: Erste verfügbare Collection
                    payload["agent_type"] = "general"
            
            logger.info(f"Sending RAG request: {payload}")
            
            async with session.post(
                f"{RAG_SERVICE_URL}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    logger.info(f"RAG Response: Found {len(data.get('results', []))} results from collection '{data.get('collection_used')}'")
                    
                    if data.get('results'):
                        # Formatiere die Ergebnisse
                        formatted_results = []
                        for i, result in enumerate(data['results'], 1):
                            content = result.get('content', '')
                            score = result.get('score', 0)
                            
                            # Kürze sehr lange Inhalte
                            if len(content) > 500:
                                content = content[:500] + "..."
                            
                            formatted_results.append(
                                f"Result {i} (Relevance: {score:.2f}):\n{content}"
                            )
                        
                        response = f"I found {len(data['results'])} relevant results in the knowledge base:\n\n" + \
                                 "\n\n---\n\n".join(formatted_results)
                        
                        # Collection Info hinzufügen
                        response += f"\n\n(Searched in collection: {data.get('collection_used', 'unknown')})"
                        
                        return response
                    else:
                        return f"No results found for '{query}' in the knowledge base. Try different keywords or ask me to search for something else."
                else:
                    error_msg = f"Error searching knowledge base: HTTP {resp.status}"
                    logger.error(error_msg)
                    return error_msg
                    
    except asyncio.TimeoutError:
        error_msg = "Search timeout - the knowledge base took too long to respond"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error accessing knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    logger.info(f"Room: {ctx.room.name}")
    logger.info(f"Using Ollama at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
    logger.info(f"Using RAG Service at {RAG_SERVICE_URL}")
    
    # Test RAG Service connection
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{RAG_SERVICE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    health = await resp.json()
                    logger.info(f"✅ RAG Service connected: {health}")
                    
                    # Collections anzeigen
                    async with session.get(f"{RAG_SERVICE_URL}/collections") as resp2:
                        if resp2.status == 200:
                            collections = await resp2.json()
                            logger.info(f"Available collections: {collections}")
                else:
                    logger.warning(f"⚠️ RAG Service health check failed: {resp.status}")
    except Exception as e:
        logger.error(f"❌ Could not connect to RAG Service: {e}")
        logger.error("The agent will work but won't be able to search the knowledge base!")
    
    # Ollama über OpenAI-kompatible API
    ollama_llm = openai.LLM(
        model=OLLAMA_MODEL,
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama",
        timeout=120.0  # 120 Sekunden Timeout für lokale Modelle
    )
    
    # Instructions mit RAG-Kontext
    instructions = """You are an AI assistant with vision capabilities and access to a comprehensive knowledge base.

    IMPORTANT: You have access to company documents including:
    - Swiss telecommunications regulations (BAKOM documents)
    - Radio licenses and frequency information
    - Technical documentation about Qdrant vector database
    - Company policies and procedures
    
    When users ask about these topics, YOU MUST use the search_knowledge function to find accurate information.
    
    Key topics that require knowledge base search:
    - BAKOM (Bundesamt für Kommunikation)
    - Funkkonzessionen / Radio licenses
    - Frequencies / Frequenzen
    - Swiss regulations
    - Qdrant documentation
    - Any specific technical or regulatory questions
    
    For vision tasks:
    - Be accurate when describing what you see
    - Common applications include YouTube, web browsers, IDEs, and other software
    - If corrected, reconsider and look again
    
    Always search the knowledge base when asked about specific information, regulations, or documentation.
    Answer based on the search results, not on general knowledge when specific documents are available."""
    
    # Assistant mit Function Calling
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=ollama_llm,
        tts=openai.TTS(),
        fnc_ctx=FunctionContext(),
        chat_ctx=openai.ChatContext(
            messages=[
                {
                    "role": "system",
                    "content": instructions
                }
            ]
        )
    )
    
    # RAG Function registrieren
    assistant.fnc_ctx.ai_callable(
        name="search_knowledge",
        description="Search the knowledge base for information about regulations, documentation, or company data",
        auto_retry=True
    )(search_knowledge)
    
    # Session starten
    session = assistant.session(
        auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY,
    )
    
    # Participants handhaben
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")
        session.set_participant(participant)
    
    # Track subscriptions
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"Video track subscribed from {participant.identity}")
            # Video handling könnte hier hinzugefügt werden
    
    # Initial participant setup
    if ctx.room.remote_participants:
        session.set_participant(list(ctx.room.remote_participants.values())[0])
    
    logger.info("Agent ready and listening...")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
