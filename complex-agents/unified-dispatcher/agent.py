import os
import sys
import logging
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("unified-dispatcher")
logger.setLevel(logging.INFO)

async def entrypoint(ctx: JobContext):
   """Unified dispatcher for all agent types"""
   
   room_name = ctx.room.name.lower()
   logger.info(f"New job for room: {room_name}")
   
   # Route to appropriate agent
   if room_name.startswith("voice_assistant_room_"):
       logger.info("→ Starting Medical Agent for Voice Assistant Frontend")
       # Connect first
       await ctx.connect()
       
       # Simple medical agent for now
       from livekit.agents.voice import Agent
       agent = Agent(
           instructions="You are a medical triage assistant. Be helpful and professional.",
           stt=deepgram.STT(),
           llm=openai.LLM(model="gpt-4o-mini"),
           tts=openai.TTS(),
       )
       session = AgentSession(vad=silero.VAD.load())
       await session.start(agent=agent, room=ctx.room)
       
   elif "vision" in room_name:
       logger.info("→ Starting Vision Agent")
       # Dynamically import at runtime (after container is running)
       try:
           # Add path to make imports work
           sys.path.insert(0, '/app')
           from complex_agents.vision_ollama.agent import entrypoint as vision_entrypoint
           await vision_entrypoint(ctx)
       except ImportError as e:
           logger.error(f"Could not import vision agent: {e}")
           # Fallback
           await ctx.connect()
           
   elif "rag" in room_name:
       logger.info("→ Starting RAG Agent")
       await ctx.connect()
       
       from livekit.agents.voice import Agent
       agent = Agent(
           instructions="You are a helpful RAG assistant with access to knowledge base.",
           stt=deepgram.STT(),
           llm=openai.LLM(model="gpt-4o-mini"),
           tts=openai.TTS(),
       )
       session = AgentSession(vad=silero.VAD.load())
       await session.start(agent=agent, room=ctx.room)
       
   else:
       logger.info("→ Starting Default Agent")
       await ctx.connect()
       
       from livekit.agents.voice import Agent
       agent = Agent(
           instructions="You are a helpful assistant.",
           stt=deepgram.STT(),
           llm=openai.LLM(model="gpt-4o-mini"),
           tts=openai.TTS(),
       )
       session = AgentSession(vad=silero.VAD.load())
       await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
   cli.run_app(WorkerOptions(
       entrypoint_fnc=entrypoint,
       max_concurrent_jobs=5
   ))
