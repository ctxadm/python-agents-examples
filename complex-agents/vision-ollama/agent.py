## This is a basic example of how to use function calling.
## To test the function, you can ask the agent to print to the console!
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import function_tool, ImageContent, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero
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

class VisionAgentWithRAG(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []

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

        super().__init__(
            instructions=instructions,
            stt=deepgram.STT(),
            llm=ollama_llm,
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )

    @function_tool(description="Search the knowledge base for information about regulations, documentation, or company data")
    async def search_knowledge(
        self,
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

    async def on_enter(self):
        room = get_job_context().room

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

        # Find the first video track (if any) from the remote participant
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])

        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            logger.info(f"Adding frame to message: {self._latest_frame.width}x{self._latest_frame.height}")
            new_message.content.append(ImageContent(image=self._latest_frame))
            # Nicht löschen, damit wir kontinuierlich Frames haben
            # self._latest_frame = None
        else:
            logger.warning("No frame available when user asked question")

    # Helper method to buffer the latest video frame from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # VideoStream hat keine close() Methode - wir setzen es einfach auf None
        if self._video_stream is not None:
            self._video_stream = None

        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)

        async def read_stream():
            frame_count = 0
            async for event in self._video_stream:
                frame_count += 1
                # Store the latest frame for use later
                self._latest_frame = event.frame

                # Log every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Received {frame_count} frames, latest: {event.frame.width}x{event.frame.height}")

        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    logger.info(f"Using Ollama at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
    logger.info(f"Using RAG Service at {RAG_SERVICE_URL}")

    session = AgentSession()
    await session.start(
        agent=VisionAgentWithRAG(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
