# LiveKit Agents 1.0.x Version
import logging
import os
import httpx
import re
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from typing import Optional

load_dotenv()

logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.INFO)

class GarageAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Du bist ein Werkstatt-Assistent mit Zugriff auf die Fahrzeugdatenbank.
            
            ERSTE ANTWORT: "Willkommen in der Werkstatt! Wie kann ich Ihnen helfen?"
            
            Du kannst:
            - Fahrzeugdaten abrufen (Kennzeichen, Marke, Modell)
            - Wartungshistorie einsehen
            - Reparaturkosten kalkulieren
            - Termine vereinbaren
            
            WICHTIG:
            - Nutze IMMER die search_vehicle_data Funktion für Fahrzeuganfragen
            - Korrigiere Kennzeichen: "ZH 12345" oder "zh eins zwei drei vier fünf"
            - Währungen: "850 Franken" statt "850.00"
            """,
            llm=openai.LLM(
                model="llama3.1:8b",  # Gleiche Version wie Medical für Konsistenz
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                temperature=0.7,
            ),
            stt=openai.STT(model="whisper-1", language="de"),
            tts=openai.TTS(model="tts-1", voice="onyx"),
            vad=silero.VAD.load(
                min_silence_duration=0.8,
                min_speech_duration=0.3,
                activation_threshold=0.5
            ),
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info(f"Garage Agent initialized with RAG service at {self.rag_url}")
        
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("Garage Agent entering session")
        await self.session.say(
            "Willkommen in der Werkstatt! Wie kann ich Ihnen helfen?",
            allow_interruptions=True
        )
    
    @function_tool
    async def search_vehicle_data(self, query: str) -> str:
        """Sucht in der Fahrzeugdatenbank nach Informationen.
        
        Args:
            query: Die Suchanfrage (z.B. Kennzeichen, Fahrzeugtyp, Problem)
            
        Returns:
            Die gefundenen Fahrzeugdaten
        """
        try:
            # Korrigiere Kennzeichen
            processed_query = self._process_license_plate(query)
            logger.info(f"Searching vehicles for: {processed_query}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"  # ✅ KORRIGIERT!
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"Found {len(results)} vehicle results")
                        formatted = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted)
                    else:
                        return "Keine Fahrzeugdaten zu dieser Anfrage gefunden."
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return "Fehler beim Zugriff auf die Fahrzeugdatenbank."
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return "Die Fahrzeugdatenbank ist momentan nicht erreichbar."
    
    def _process_license_plate(self, text: str) -> str:
        """Korrigiert Sprache-zu-Text Fehler bei Kennzeichen"""
        # Beispiel: "zh eins zwei drei vier fünf" -> "ZH 12345"
        
        # Deutsche Zahlwörter zu Ziffern
        number_map = {
            'null': '0', 'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
            'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 'neun': '9'
        }
        
        # Ersetze Zahlwörter durch Ziffern
        for word, digit in number_map.items():
            text = text.replace(f" {word} ", f" {digit} ")
            text = text.replace(f" {word}", f" {digit}")
        
        # Normalisiere Kennzeichen Format (z.B. "zh 1 2 3 4 5" -> "ZH 12345")
        pattern = r'([a-zA-Z]{2})\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            canton = match.group(1).upper()
            numbers = ''.join([match.group(i) for i in range(2, 7)])
            corrected = f"{canton} {numbers}"
            text = re.sub(pattern, corrected, text, flags=re.IGNORECASE)
            logger.info(f"Corrected license plate to '{corrected}'")
        
        return text

async def entrypoint(ctx: JobContext):
    """Entry point for garage agent"""
    logger.info("=== Garage Agent Starting (1.0.x) ===")
    
    # Connect to room
    await ctx.connect()
    logger.info("Connected to room")
    
    # Create session with new 1.0.x API
    session = AgentSession()
    
    # Start session with agent instance
    await session.start(
        room=ctx.room,
        agent=GarageAgent()
    )
    
    logger.info("Garage Agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
