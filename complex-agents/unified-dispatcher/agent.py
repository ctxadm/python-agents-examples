import os
import sys
import logging
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("unified-dispatcher")
logger.setLevel(logging.INFO)

# Ollama Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

async def entrypoint(ctx: JobContext):
    """Unified dispatcher for all agent types"""
    
    room_name = ctx.room.name.lower()
    logger.info(f"New job for room: {room_name}")
    
    # Route to appropriate agent
    if room_name.startswith("voice_assistant_room_"):
        logger.info("→ Starting Medical Agent for Voice Assistant Frontend")
        await ctx.connect()
        
        # Medical agent mit lokalem LLM
        medical_llm = openai.LLM(
            model="llama3",  # oder "mistral", "qwen2.5"
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=60.0,
            temperature=0.7
        )
        
        agent = Agent(
            instructions="You are a medical triage assistant. Be helpful and professional.",
            stt=deepgram.STT(),
            llm=medical_llm,
            tts=openai.TTS(),  # Noch OpenAI TTS
        )
        session = AgentSession(vad=silero.VAD.load())
        await session.start(agent=agent, room=ctx.room)
        
    elif "vision" in room_name:
        logger.info("→ Starting Vision Agent")
        try:
            sys.path.insert(0, '/app')
            from complex_agents.vision_ollama.agent import entrypoint as vision_entrypoint
            await vision_entrypoint(ctx)
        except ImportError as e:
            logger.error(f"Could not import vision agent: {e}")
            await ctx.connect()
            
    elif "rag" in room_name:
        logger.info("→ Starting RAG Agent")
        await ctx.connect()
        
        # RAG agent mit lokalem LLM (mit Function Calling Support)
        rag_llm = openai.LLM(
            model="llama3.2:latest",  # ✅ Korrekt: llama3.2:latest für RAG
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=60.0,
            temperature=0.1  # Niedrig für präzise RAG-Antworten
        )
        
        agent = Agent(
            instructions="You are a helpful RAG assistant with access to knowledge base.",
            stt=deepgram.STT(),
            llm=rag_llm,
            tts=openai.TTS(),
        )
        session = AgentSession(vad=silero.VAD.load())
        await session.start(agent=agent, room=ctx.room)
        
    else:
        logger.info("→ Starting Default Agent")
        await ctx.connect()
        
        # Default agent mit lokalem LLM
        default_llm = openai.LLM(
            model="llama3",
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=60.0,
            temperature=0.7
        )
        
        agent = Agent(
            instructions="You are a helpful assistant.",
            stt=deepgram.STT(),
            llm=default_llm,
            tts=openai.TTS(),
        )
        session = AgentSession(vad=silero.VAD.load())
        await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint
    ))
