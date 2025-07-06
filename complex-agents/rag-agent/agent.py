# complex-agents/rag-agent/agent.py

import logging
import os
from typing import Annotated
import httpx

from livekit.agents import JobContext, function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("rag-agent")

# Global RAG Service URL
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")


@function_tool(description="Durchsucht die Qdrant Wissensdatenbank nach relevanten Informationen")
async def search_knowledge_base(query: str) -> str:
    """
    Durchsucht die Wissensdatenbank nach relevanten Informationen.
    
    Args:
        query: Die Suchanfrage für die Wissensdatenbank
    """
    logger.info(f"RAG Suche: {query}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/search",
                json={
                    "query": query,
                    "agent_type": "general",
                    "top_k": 3
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                collection = data.get("collection_used", "unknown")
                
                logger.info(f"Gefunden: {len(results)} Ergebnisse aus '{collection}'")
                
                if results:
                    # Formatiere die Ergebnisse
                    formatted = "\n\n".join([
                        f"[Relevanz: {r['score']:.2f}]\n{r['content']}"
                        for r in results
                    ])
                    return f"Ergebnisse aus {collection}:\n\n{formatted}"
                else:
                    return "Keine relevanten Informationen gefunden."
            else:
                logger.error(f"RAG Service Error: {response.status_code}")
                return f"Fehler beim Zugriff auf die Wissensdatenbank (Status: {response.status_code})"
                
    except Exception as e:
        logger.error(f"RAG Service Exception: {e}")
        return f"Verbindungsfehler zur Wissensdatenbank: {str(e)}"


@function_tool(description="Zeigt alle verfügbaren Sammlungen in der Wissensdatenbank")
async def list_available_collections() -> str:
    """Zeigt alle verfügbaren Sammlungen in der Wissensdatenbank"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{RAG_SERVICE_URL}/collections")
            
            if response.status_code == 200:
                data = response.json()
                collections = data.get("collections", [])
                
                if collections:
                    formatted = "\n".join([
                        f"- {c['name']} ({c['documents']} Dokumente)"
                        for c in collections
                    ])
                    return f"Verfügbare Wissenssammlungen:\n{formatted}"
                else:
                    return "Keine Sammlungen gefunden."
            else:
                return "Fehler beim Abrufen der Sammlungen."
                
    except Exception as e:
        logger.error(f"Collections Error: {e}")
        return "Verbindungsfehler zur Wissensdatenbank."


async def entrypoint(ctx: JobContext):
    """RAG Agent Entrypoint"""
    await ctx.connect()
    
    logger.info(f"RAG Agent startet mit Service URL: {RAG_SERVICE_URL}")
    
    # Health Check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_response = await client.get(f"{RAG_SERVICE_URL}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.info(f"RAG Service Status: {health_data}")
            else:
                logger.warning(f"RAG Service Health Check fehlgeschlagen: {health_response.status_code}")
    except Exception as e:
        logger.error(f"Konnte RAG Service nicht erreichen: {e}")
    
    # System Instructions
    instructions = """Du bist ein intelligenter KI-Assistent mit Zugriff auf eine umfangreiche Wissensdatenbank über Schweizer Funkregulierungen (BAKOM).

WICHTIG: Du hast Zugriff auf folgende Funktionen:
- search_knowledge_base: Um in der Wissensdatenbank zu suchen
- list_available_collections: Um verfügbare Datensammlungen anzuzeigen

Nutze IMMER die Funktion 'search_knowledge_base' wenn Nutzer nach Informationen fragen über:
- BAKOM (Bundesamt für Kommunikation)
- Funkkonzessionen und Regulierungen in der Schweiz
- CB-Funk Frequenzen und andere Funkdienste
- Konzessionsfreie und konzessionspflichtige Dienste

Beantworte Fragen präzise basierend auf den gefundenen Informationen aus der Datenbank.
Wenn du dir unsicher bist oder keine Informationen findest, sage das ehrlich.
Antworte auf Deutsch, wenn du auf Deutsch angesprochen wirst."""
    
    # LLM Configuration - Use Ollama if available, otherwise OpenAI
    ollama_host = os.getenv("OLLAMA_HOST")
    if ollama_host:
        logger.info(f"Verwende Ollama LLM: {ollama_host}")
        agent_llm = openai.LLM(
            model="llama3.2:latest",
            base_url=f"{ollama_host}/v1",
            api_key="ollama",
            timeout=60.0,
            temperature=0.1  # Niedrig für präzise Antworten
        )
    else:
        logger.info("Verwende OpenAI LLM")
        agent_llm = openai.LLM(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
    
    # Create Agent with new API
    agent = Agent(
        instructions=instructions,
        stt=deepgram.STT(
            language="de-DE",
            model="nova-2"
        ),
        llm=agent_llm,
        tts=openai.TTS(
            voice="alloy",
            model="tts-1"            
        ),
        tools=[search_knowledge_base, list_available_collections]  # Functions direkt als Liste
    )
    
    # Start Session
    session = AgentSession(
        vad=silero.VAD.load()
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("RAG Agent läuft und wartet auf Anfragen!")


if __name__ == "__main__":
    from livekit.agents import cli, WorkerOptions
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )
