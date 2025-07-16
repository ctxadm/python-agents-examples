import os
import sys
import logging
from typing import Optional

# Pfad-Fix für Imports
sys.path.insert(0, '/app')

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, openai

# Import des RAG Clients - relativer Import für GitHub
try:
    from .rag_client import RAGServiceClient
except ImportError:
    from rag_client import RAGServiceClient

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-agent")

# Konfiguration aus Environment Variables
AGENT_TYPE = os.getenv("AGENT_TYPE", "search")
AGENT_NAME = os.getenv("AGENT_NAME", f"RAG {AGENT_TYPE.capitalize()} Assistant")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

# System Prompts für verschiedene Agent-Typen
SYSTEM_PROMPTS = {
    "search": """Du bist ein hilfreicher Such-Assistent mit Zugriff auf eine umfangreiche Wissensdatenbank.
    
Deine Aufgaben:
- Durchsuche die Wissensdatenbank nach relevanten Informationen
- Beantworte Fragen präzise und fundiert
- Erwähne immer, wenn Informationen aus der Wissensdatenbank stammen
- Gib zu, wenn du keine passenden Informationen findest

Wichtige Regeln:
- Antworte immer auf Deutsch
- Sei freundlich und professionell
- Fasse dich kurz und prägnant
- Nutze die search_knowledge_base Funktion für alle Anfragen""",
    
    "garage": """Du bist ein erfahrener Kfz-Meister und Experte für Autoreparaturen und Wartung.
    
Deine Expertise:
- Diagnose von Fahrzeugproblemen
- Wartungsempfehlungen und -intervalle
- Reparaturanleitungen und Tipps
- Ersatzteilberatung

Wichtige Regeln:
- Antworte immer auf Deutsch
- Gib praktische, umsetzbare Ratschläge
- Erwähne Sicherheitshinweise bei gefährlichen Arbeiten
- Nutze die search_knowledge_base Funktion für technische Details
- Empfehle bei komplexen Problemen den Besuch einer Fachwerkstatt""",
    
    "medical": """Du bist ein medizinischer Ernährungsberater mit Zugriff auf eine umfangreiche Gesundheitsdatenbank.
    
Deine Expertise:
- Ernährungsberatung und Diätpläne
- Nährwertinformationen und Lebensmittelkunde
- Gesundheitliche Auswirkungen von Ernährung
- Allergien und Unverträglichkeiten

Wichtiger Hinweis:
- Du bist KEIN Arzt und gibst keine medizinischen Diagnosen
- Bei ernsthaften gesundheitlichen Problemen empfiehlst du immer einen Arztbesuch
- Deine Beratung ersetzt keine professionelle medizinische Behandlung

Regeln:
- Antworte immer auf Deutsch
- Sei einfühlsam und verständnisvoll
- Nutze die search_knowledge_base Funktion für fundierte Informationen
- Betone bei Bedarf, dass deine Ratschläge allgemeiner Natur sind""",
    
    "general": """Du bist ein hilfreicher KI-Assistent mit Zugriff auf eine Wissensdatenbank.
Beantworte Fragen präzise und nutze die search_knowledge_base Funktion wenn nötig.
Antworte immer auf Deutsch."""
}

class RAGAgent:
    """RAG Agent mit LiveKit Integration"""
    
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.rag_client: Optional[RAGServiceClient] = None
        self.assistant: Optional[VoiceAssistant] = None
    
    async def initialize(self):
        """Initialisiere RAG Client und Voice Assistant"""
        # RAG Client erstellen
        self.rag_client = RAGServiceClient(RAG_SERVICE_URL)
        
        # Health Check
        health = await self.rag_client.health_check()
        logger.info(f"RAG Service Status: {health}")
        
        if health.get("status") != "healthy":
            logger.warning(f"RAG Service nicht healthy: {health}")
        
        # Collections laden
        collections = await self.rag_client.get_collections()
        if collections:
            logger.info(f"Verfügbare Collections: {[c['name'] for c in collections.get('collections', [])]}")
        
        # System Prompt auswählen
        system_prompt = SYSTEM_PROMPTS.get(AGENT_TYPE, SYSTEM_PROMPTS["general"])
        
        # Chat Context erstellen
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=system_prompt
        )
        
        # Voice Assistant konfigurieren
        self.assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(
                language="de-DE"  # Deutsch für STT
            ),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.7
            ),
            tts=openai.TTS(
                voice="alloy",
                model="tts-1",
                speed=1.0
            ),
            chat_ctx=initial_ctx,
            interrupt_speech_duration=0.5,
            interrupt_min_words=3,
        )
        
        # RAG Such-Funktion registrieren
        await self._register_functions()
        
        logger.info(f"RAG Agent ({AGENT_TYPE}) initialisiert")
    
    async def _register_functions(self):
        """Registriere LLM Funktionen"""
        
        @self.assistant.llm.register_function()
        async def search_knowledge_base(
            query: str,
            collection: Optional[str] = None
        ) -> str:
            """
            Durchsuche die Wissensdatenbank nach Informationen.
            
            Diese Funktion sollte für ALLE Benutzeranfragen verwendet werden,
            um relevante Informationen aus der Wissensdatenbank zu finden.
            
            Args:
                query: Die Suchanfrage - formuliere sie klar und präzise
                collection: Optional - spezifische Collection (normalerweise automatisch)
            
            Returns:
                Relevante Informationen aus der Wissensdatenbank
            """
            logger.info(f"[{AGENT_TYPE}] Suche: '{query}'")
            
            # Suche durchführen
            results = await self.rag_client.search(
                query=query,
                agent_type=AGENT_TYPE,
                collection=collection
            )
            
            # Ergebnisse formatieren
            formatted = self.rag_client.format_results(results)
            logger.info(f"[{AGENT_TYPE}] Suchergebnis: {len(results.get('results', []))} Treffer")
            
            return formatted
        
        # Agent-spezifische Funktionen
        if AGENT_TYPE == "garage":
            @self.assistant.llm.register_function()
            async def get_maintenance_schedule(
                vehicle_type: str = "PKW",
                component: Optional[str] = None
            ) -> str:
                """
                Hole Wartungsintervalle für Fahrzeugkomponenten.
                
                Args:
                    vehicle_type: Fahrzeugtyp (PKW, LKW, Motorrad)
                    component: Spezifische Komponente (z.B. "Ölwechsel", "Bremsen")
                
                Returns:
                    Wartungsempfehlungen und Intervalle
                """
                query = f"Wartungsintervalle {vehicle_type}"
                if component:
                    query += f" {component}"
                
                results = await self.rag_client.search(
                    query=query,
                    agent_type=AGENT_TYPE
                )
                
                return self.rag_client.format_results(results)
        
        elif AGENT_TYPE == "medical":
            @self.assistant.llm.register_function()
            async def get_nutrition_info(
                food_item: str,
                portion_size: Optional[str] = "100g"
            ) -> str:
                """
                Hole Nährwertinformationen für Lebensmittel.
                
                Args:
                    food_item: Das Lebensmittel
                    portion_size: Portionsgröße (Standard: 100g)
                
                Returns:
                    Nährwertinformationen und gesundheitliche Hinweise
                """
                query = f"Nährwerte {food_item} {portion_size}"
                
                results = await self.rag_client.search(
                    query=query,
                    agent_type=AGENT_TYPE
                )
                
                return self.rag_client.format_results(results)
    
    async def run(self):
        """Starte den Agent"""
        # Assistant starten
        self.assistant.start(self.ctx.room)
        
        # Begrüßung senden
        greeting = self._get_greeting()
        await self.assistant.say(greeting, allow_interruptions=True)
        
        logger.info(f"RAG Agent ({AGENT_TYPE}) läuft in Raum {self.ctx.room.name}")
    
    def _get_greeting(self) -> str:
        """Hole agent-spezifische Begrüßung"""
        greetings = {
            "search": "Hallo! Ich bin Ihr Such-Assistent. Ich kann Ihnen helfen, Informationen in unserer Wissensdatenbank zu finden. Was möchten Sie wissen?",
            "garage": "Guten Tag! Ich bin Ihr Kfz-Experte. Ich helfe Ihnen gerne bei Fragen zu Autoreparaturen, Wartung und technischen Problemen. Wie kann ich Ihnen helfen?",
            "medical": "Hallo! Ich bin Ihr Ernährungsberater. Ich kann Ihnen bei Fragen zu gesunder Ernährung, Nährwerten und Diätplänen helfen. Womit kann ich Ihnen dienen?",
            "general": "Hallo! Ich bin Ihr KI-Assistent mit Zugriff auf eine umfangreiche Wissensdatenbank. Womit kann ich Ihnen helfen?"
        }
        return greetings.get(AGENT_TYPE, greetings["general"])
    
    async def cleanup(self):
        """Räume auf"""
        if self.rag_client:
            await self.rag_client.close()
        logger.info(f"RAG Agent ({AGENT_TYPE}) beendet")

async def entrypoint(ctx: JobContext):
    """
    Haupteinstiegspunkt für den RAG Agent.
    Wird von LiveKit aufgerufen wenn ein neuer Job zugewiesen wird.
    """
    logger.info(f"RAG Agent Job gestartet - Typ: {AGENT_TYPE}, Raum: {ctx.room.name}")
    
    # Mit Auto-Subscribe für Medien-Tracks
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # RAG Agent erstellen und initialisieren
    agent = RAGAgent(ctx)
    
    try:
        await agent.initialize()
        await agent.run()
        
        # Warte auf Raum-Schließung
        @ctx.room.on("disconnected")
        async def on_disconnected():
            logger.info("Raum getrennt, beende Agent")
            await agent.cleanup()
        
    except Exception as e:
        logger.error(f"Fehler im RAG Agent: {e}", exc_info=True)
        await agent.cleanup()
        raise

# Worker starten
if __name__ == "__main__":
    # CLI Argumente parsen und Worker starten
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_TYPE,  # WICHTIG: Für Multi-Agent Dispatch!
            port=0,  # Automatische Port-Zuweisung
            host="0.0.0.0",
            log_level="info"
        ),
        # Job Process Konfiguration
        job_process_opts=JobProcess(
            num_idle_processes=1,  # Immer 1 Prozess bereit
            max_concurrency=10     # Max 10 gleichzeitige Jobs
        )
    )
