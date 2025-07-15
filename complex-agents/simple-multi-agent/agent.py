import os
import sys
import json
import logging
from livekit.agents import JobContext, WorkerOptions, cli

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-agent")

AGENT_TYPE = os.getenv("AGENT_TYPE", "general")

async def entrypoint(ctx: JobContext):
    # Metadata Check
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
            requested_type = metadata.get("agent_type", "general")
            
            if requested_type != AGENT_TYPE:
                logger.info(f"Skipping room - requested: {requested_type}, I am: {AGENT_TYPE}")
                return
                
        except json.JSONDecodeError:
            logger.warning("Could not parse room metadata")
            return
    
    logger.info(f"Processing room as {AGENT_TYPE} agent")
    await ctx.connect()
    
    # Agent-spezifische Imports
    try:
        if AGENT_TYPE == "garage":
            from complex_agents.vision_ollama.agent import entrypoint as agent_entrypoint
        elif AGENT_TYPE == "medical":
            from complex_agents.nutrition_assistant.agent import entrypoint as agent_entrypoint
        elif AGENT_TYPE == "search":
            from rag.main import entrypoint as agent_entrypoint
        else:
            logger.error(f"Unknown agent type: {AGENT_TYPE}")
            return
            
        # Original Agent Entrypoint aufrufen
        await agent_entrypoint(ctx)
        
    except ImportError as e:
        logger.error(f"Failed to import agent module: {e}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        worker_type=f"agent-{AGENT_TYPE}"
    ))
