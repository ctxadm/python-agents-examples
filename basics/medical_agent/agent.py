import logging
import os
import httpx
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from typing import Optional

load_dotenv()

logger = logging.getLogger("medical-assistant")
logger.setLevel(logging.INFO)

class MedicalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Du bist ein medizinischer Assistent mit Zugriff auf die Patientendatenbank.
            
            ERSTE ANTWORT: "Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?"
            
            WICHTIG - Datenbank:
            - Nutze IMMER die search_patient_data Funktion für Patientenanfragen
            - Sage NIE "nicht gefunden" wenn die Funktion Daten zurückgibt
            - Korrigiere Patienten-IDs: "p null null fünf" = "P005"
            
            Währungen: "15 Franken 50" statt "15.50"
            """,
            llm=openai.LLM(
                model="llama3.1:8b",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                temperature=0.7,
            ),
            stt=openai.STT(model="whisper-1", language="de"),
            tts=openai.TTS(model="tts-1", voice="shimmer"),
            vad=silero.VAD.load(
                min_silence_duration=0.8,
                min_speech_duration=0.3,
                activation_threshold=0.5
            ),
        )
        
        self.rag_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        logger.info(f"Medical Agent initialized with RAG service at {self.rag_url}")
        
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("Medical Agent entering session")
        await self.session.say(
            "Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?",
            allow_interruptions=True
        )
    
    @function_tool
    async def search_patient_data(self, query: str) -> str:
        """Sucht in der Patientendatenbank nach Informationen.
        
        Args:
            query: Die Suchanfrage (z.B. Patienten-ID oder Symptome)
            
        Returns:
            Die gefundenen Patientendaten
        """
        try:
            # Korrigiere Patienten-IDs
            processed_query = self._process_patient_id(query)
            logger.info(f"Searching for: {processed_query}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "medical",
                        "top_k": 3,
                        "collection": "medical_nutrition"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"Found {len(results)} results")
                        formatted = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted)
                    else:
                        return "Keine Daten zu dieser Anfrage gefunden."
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return "Fehler beim Zugriff auf die Datenbank."
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return "Die Datenbank ist momentan nicht erreichbar."
    
    def _process_patient_id(self, text: str) -> str:
        """Korrigiert Sprache-zu-Text Fehler bei Patienten-IDs"""
        import re
        
        # Pattern für "p null null X"
        pattern = r'p\s*null\s*null\s*(\w+)'
        match = re.search(pattern, text.lower())
        
        if match:
            number = match.group(1)
            # Deutsche Zahlwörter zu Ziffern
            number_map = {
                'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                'neun': '9', 'null': '0'
            }
            
            if number in number_map:
                number = number_map[number]
            
            # Ersetze im Original-Text
            corrected_id = f"P00{number}"
            text = re.sub(pattern, corrected_id, text, flags=re.IGNORECASE)
            logger.info(f"Corrected patient ID to '{corrected_id}'")
        
        return text

async def entrypoint(ctx: JobContext):
    """Entry point for medical agent"""
    logger.info("=== Medical Agent Starting (1.0.x) ===")
    
    # Connect to room
    await ctx.connect()
    logger.info("Connected to room")
    
    # Create session with new 1.0.x API
    session = AgentSession()
    
    # Start session with agent instance
    await session.start(
        room=ctx.room,
        agent=MedicalAgent()
    )
    
    logger.info("Medical Agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
