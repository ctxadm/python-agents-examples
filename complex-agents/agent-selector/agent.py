# complex-agents/agent-selector/agent.py

import asyncio
import logging
import os
import sys
import importlib.util
from typing import Optional

from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe

logger = logging.getLogger("agent-selector")

class AgentSelector:
    def __init__(self):
        self.active_agents = {}
        
    async def entrypoint(self, ctx: JobContext):
        """Haupteinstiegspunkt - wählt Agent basierend auf Room-Namen"""
        
        room_name = ctx.room.name.lower()
        room_id = await ctx.room.sid  # FIX: await hinzugefügt
        
        logger.info(f"=== Neuer Job empfangen ===")
        logger.info(f"Room Name: '{ctx.room.name}'")
        logger.info(f"Room ID: {room_id}")
        logger.info(f"Participant Count: {len(ctx.room.remote_participants)}")
        
        # Agent-Auswahl basierend auf Room-Namen
        if any(keyword in room_name for keyword in ['vision', 'ollama', 'bild', 'image', 'visual', 'camera']):
            logger.info("→ Vision-Keywords erkannt, lade Vision-Ollama Agent")
            await self.run_vision_agent(ctx)
            
        elif any(keyword in room_name for keyword in ['rag', 'knowledge', 'wissen', 'datenbank', 'qdrant', 'search']):
            logger.info("→ RAG-Keywords erkannt, lade RAG Agent")
            await self.run_rag_agent(ctx)
            
        else:
            # Standard: Vision Agent
            logger.info("→ Keine Keywords erkannt, lade Vision-Ollama Agent als Standard")
            await self.run_vision_agent(ctx)
    
    async def run_vision_agent(self, ctx: JobContext):
        """Lädt und startet den Vision-Ollama Agent"""
        try:
            logger.info("Lade Vision-Ollama Agent Module...")
            
            # Dynamisch das vision-ollama Modul laden
            spec = importlib.util.spec_from_file_location(
                "vision_agent", 
                "/app/complex-agents/vision-ollama/agent.py"
            )
            vision_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vision_module)
            
            # Die entrypoint Funktion aufrufen
            logger.info("Starte Vision-Ollama Agent...")
            await vision_module.entrypoint(ctx)
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Vision Agents: {e}")
            raise
    
    async def run_rag_agent(self, ctx: JobContext):
        """Lädt und startet den RAG Agent"""
        try:
            logger.info("Lade RAG Agent Module...")
            
            # Dynamisch das RAG Modul laden
            spec = importlib.util.spec_from_file_location(
                "rag_agent", 
                "/app/rag/main.py"
            )
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            
            # Prüfen ob es eine entrypoint Funktion gibt
            if hasattr(rag_module, 'entrypoint'):
                logger.info("Starte RAG Agent...")
                await rag_module.entrypoint(ctx)
            else:
                # Falls der RAG Agent anders strukturiert ist
                logger.error("RAG Agent hat keine entrypoint Funktion")
                logger.info("Verfügbare Funktionen im RAG Modul:")
                for attr in dir(rag_module):
                    if not attr.startswith('_'):
                        logger.info(f"  - {attr}")
                
                # Fallback auf Vision Agent
                logger.info("Fallback: Starte Vision Agent stattdessen")
                await self.run_vision_agent(ctx)
                
        except Exception as e:
            logger.error(f"Fehler beim Laden des RAG Agents: {e}")
            logger.info("Fallback: Starte Vision Agent")
            await self.run_vision_agent(ctx)


# Hauptprogramm
if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("=== LiveKit Agent Selector gestartet ===")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Python Version: {sys.version}")
    
    # Verfügbare Agents anzeigen
    logger.info("Prüfe verfügbare Agents:")
    
    vision_path = "/app/complex-agents/vision-ollama/agent.py"
    rag_path = "/app/rag/main.py"
    
    if os.path.exists(vision_path):
        logger.info(f"✓ Vision-Ollama Agent gefunden: {vision_path}")
    else:
        logger.error(f"✗ Vision-Ollama Agent nicht gefunden: {vision_path}")
        
    if os.path.exists(rag_path):
        logger.info(f"✓ RAG Agent gefunden: {rag_path}")
    else:
        logger.error(f"✗ RAG Agent nicht gefunden: {rag_path}")
    
    # Umgebungsvariablen anzeigen
    logger.info("Umgebungsvariablen:")
    logger.info(f"  LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'nicht gesetzt')}")
    logger.info(f"  OLLAMA_HOST: {os.getenv('OLLAMA_HOST', 'nicht gesetzt')}")
    logger.info(f"  OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'nicht gesetzt')}")
    logger.info(f"  RAG_SERVICE_URL: {os.getenv('RAG_SERVICE_URL', 'nicht gesetzt')}")
    
    # Agent Selector starten
    selector = AgentSelector()
    
    # Einfache WorkerOptions ohne Type
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=selector.entrypoint
        )
    )
