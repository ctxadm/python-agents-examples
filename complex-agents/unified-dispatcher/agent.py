import os
import sys
import logging
import importlib.util
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
        
        # Medical agent mit lokalem LLM - HARDCODED
        medical_llm = openai.LLM(
            model="llama3.2:latest",  # HARDCODED: llama3.2:latest für Medical Agent
            base_url=f"{OLLAMA_HOST}/v1",
            api_key="ollama",
            timeout=60.0,
            temperature=0.7
        )
        
        agent = Agent(
            instructions="You are a medical triage assistant. Be helpful and professional.",
            stt=deepgram.STT(),
            llm=medical_llm,
            tts=openai.TTS(),
        )
        session = AgentSession(vad=silero.VAD.load())
        await session.start(agent=agent, room=ctx.room)
        
    elif "vision" in room_name:
        logger.info("→ Starting Vision Agent")
        try:
            # Direkter Import mit vollem Pfad wegen Bindestrich im Ordnernamen
            spec = importlib.util.spec_from_file_location(
                "vision_agent", 
                "/app/complex-agents/vision-ollama/agent.py"
            )
            vision_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vision_module)
            await vision_module.entrypoint(ctx)
        except Exception as e:
            logger.error(f"Could not import vision agent: {e}")
            await ctx.connect()
            
    elif "rag" in room_name:
        logger.info("→ Starting RAG Agent")
        try:
            # Import des originalen RAG Agents mit allen Functions
            spec = importlib.util.spec_from_file_location(
                "rag_agent", 
                "/app/complex-agents/rag-agent/agent.py"
            )
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            
            # Nutze den originalen RAG Agent mit VoiceAssistant und Function Context
            await rag_module.entrypoint(ctx)
            
        except Exception as e:
            logger.error(f"Could not import RAG agent: {e}")
            # Fallback zu einem einfachen Agent ohne RAG-Funktionalität
            await ctx.connect()
            
            rag_llm = openai.LLM(
                model="llama3.2:latest",
                base_url=f"{OLLAMA_HOST}/v1",
                api_key="ollama",
                timeout=60.0,
                temperature=0.1
            )
            
            agent = Agent(
                instructions="You are a helpful assistant. (Note: RAG functions are not available in fallback mode)",
                stt=deepgram.STT(),
                llm=rag_llm,
                tts=openai.TTS(),
            )
            session = AgentSession(vad=silero.VAD.load())
            await session.start(agent=agent, room=ctx.room)
        
    else:
        logger.info("→ Starting Default Agent")
        await ctx.connect()
        
        # Default agent mit lokalem LLM - HARDCODED
        default_llm = openai.LLM(
            model="llama3.2:latest",  # HARDCODED: llama3.2:latest für Default Agent
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
