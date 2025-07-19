#!/usr/bin/env python3
"""
Modern Multi-Agent mit expliziter Agent-Name-Registrierung
Ersetzt simple_multi_agent.py
"""
import os
import sys
import json
import logging
from livekit.agents import cli, WorkerOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modern-multi-agent")

# Agent Configuration aus Environment
AGENT_TYPE = os.getenv("AGENT_TYPE", "general")
AGENT_NAME = os.getenv("AGENT_NAME", "general-assistant")

# Pfade zu den GitHub Agents
sys.path.insert(0, '/app')

def main():
    logger.info(f"Starting Modern Multi-Agent: type={AGENT_TYPE}, name={AGENT_NAME}")
    
    # Import des richtigen Agents basierend auf Type
    try:
        if AGENT_TYPE == "vision":
            from complex_agents.vision_ollama.agent import entrypoint
        elif AGENT_TYPE == "medical":
            from complex_agents.nutrition_assistant.agent import entrypoint
        elif AGENT_TYPE == "search":
            from rag.main import entrypoint
        elif AGENT_TYPE == "garage":
            # Garage nutzt den simple-multi-agent
            from complex_agents.simple_multi_agent import entrypoint
        else:
            # Default: general agent
            from complex_agents.simple_multi_agent import entrypoint
            
        logger.info(f"Successfully imported entrypoint for {AGENT_TYPE}")
    except ImportError as e:
        logger.error(f"Failed to import agent for type {AGENT_TYPE}: {e}")
        # Fallback to simple agent
        from complex_agents.simple_multi_agent import entrypoint
    
    # Worker Options mit explizitem Agent Namen
    options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,  # KRITISCH: Muss mit Token Server übereinstimmen!
        max_idle_time=60,
        shutdown_process_timeout=60
    )
    
    logger.info(f"Registering worker with agent_name='{AGENT_NAME}'")
    logger.info("Worker will ONLY receive jobs explicitly dispatched to this agent name")
    
    # Start worker
    cli.run_app(options)

if __name__ == "__main__":
    main()
