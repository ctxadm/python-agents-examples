# complex-agents/rag-agent/agent.py

import logging
import os
from typing import Annotated
import httpx

from livekit.agents import JobContext, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("rag-agent")

class RAGFunctions(llm.FunctionContext):
    """Functions für RAG Agent"""
    
    def __init__(self, rag_service_url: str):
        super().__init__()
        self.rag_service_url = rag_service_url
    
    @llm.ai_callable()
    async def search_knowledge_base(
        self,
        query: Annotated[str, llm.TypeInfo(description="Die Suchanfrage für die Wissensdatenbank")]
    ) -> str:
        """Durchsucht die Qdrant Wissensdatenbank nach relevanten Informationen"""
        logger.info(f"RAG Suche: {query}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.rag_service_url}/search",
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
    
    @llm.ai_callable()
    async def list_available_collections(self) -> str:
        """Zeigt alle verfügbaren Sammlungen in der Wissensdatenbank"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.rag_service_url}/collections")
                
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
    
    # RAG Service URL
    rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
    logger.info(f"RAG Agent startet mit Service URL: {rag_service_url}")
    
    # Health Check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_response = await client.get(f"{rag_service_url}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.info(f"RAG Service Status: {health_data}")
            else:
                logger.warning(f"RAG Service Health Check fehlgeschlagen: {health_response.status_code}")
    except Exception as e:
        logger.error(f"Konnte RAG Service nicht erreichen: {e}")
    
    # System Instructions
    instructions = """Du bist ein intelligenter KI-Assistent mit Zugriff auf eine umfangreiche Wissensdatenbank über Schweizer Funkregulierungen (BAKOM).

WICHTIG: Nutze IMMER die Funktion 'search_knowledge_base' um nach Informationen zu suchen, wenn der Nutzer Fragen stellt!

Die Wissensdatenbank enthält Informationen über:
- BAKOM (Bundesamt für Kommunikation)
- Funkkonzessionen und Regulierungen in der Schweiz
- CB-Funk Frequenzen und andere Funkdienste
- Konzessionsfreie und konzessionspflichtige Dienste

Beantworte Fragen präzise basierend auf den gefundenen Informationen.
Wenn du dir unsicher bist oder keine Informationen findest, sage das ehrlich.
Du kannst auch 'list_available_collections' nutzen um zu sehen, welche Wissensgebiete verfügbar sind.
Antworte auf Deutsch, wenn du auf Deutsch angesprochen wirst."""
    
    # Function Context mit RAG Service
    fnc_ctx = RAGFunctions(rag_service_url)
    
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
            model="tts-1",
            language="de"
        ),
        tools=[fnc_ctx]  # Functions als tools übergeben
    )
    
    # Start Session
    session = AgentSession(
        vad=silero.VAD.load()
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("RAG Agent läuft und wartet auf Anfragen!")
    
    # Keep alive - neue API hat kein wait_for_close
    # Der Agent läuft automatisch bis der Room geschlossen wird


if __name__ == "__main__":
    from livekit.agents import cli, WorkerOptions
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )
