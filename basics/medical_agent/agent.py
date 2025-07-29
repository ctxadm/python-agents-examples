# LiveKit Agents 1.0.0+ - Medical Agent
import logging
import os
import httpx
import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, deepgram, silero

logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

class MedicalAgentState:
    """State management for medical agent"""
    def __init__(self):
        self.patient_data: Optional[Dict] = None
        self.search_phase: int = 0  # 0=name, 1=year, 2=done

async def load_patient_data(
    name: str,
    birth_year: int
) -> str:
    """
    Lädt ALLE Patientendaten basierend auf Name und Geburtsjahr.
    
    Args:
        name: Vollständiger Name des Patienten
        birth_year: Geburtsjahr (z.B. 2010)
    
    Returns:
        Formatierte Patientendaten oder Fehlermeldung
    """
    logger.info(f"🔍 Loading patient data for: {name}, born {birth_year}")
    
    try:
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        async with httpx.AsyncClient() as client:
            # Schritt 1: Suche nach dem Patienten
            search_response = await client.post(
                f"{rag_url}/search",
                json={
                    "query": f"{name} {birth_year}",
                    "agent_type": "medical",
                    "top_k": 20
                },
                timeout=30.0
            )
            
            if search_response.status_code != 200:
                return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."
            
            results = search_response.json()
            documents = results.get("results", [])
            
            # Schritt 2: Finde den richtigen Patienten
            patient_data = None
            
            for doc in documents:
                payload = doc.get("payload", {})
                doc_name = payload.get("patient_name", "")
                doc_birth = payload.get("geburtsdatum", "")
                
                # Prüfe Name
                if name.lower() in doc_name.lower():
                    # Prüfe Geburtsjahr
                    if doc_birth and str(birth_year) in doc_birth:
                        patient_data = payload
                        break
            
            if not patient_data:
                return f"Ich konnte keinen Patienten namens {name} mit Geburtsjahr {birth_year} finden."
            
            # Schritt 3: Formatiere die Daten
            output = f"=== PATIENTENDATEN FÜR {patient_data.get('patient_name', name).upper()} ===\n\n"
            
            # Stammdaten
            output += "📋 STAMMDATEN:\n"
            output += f"Patient: {patient_data.get('patient_name', name)}\n"
            output += f"Geburtsdatum: {patient_data.get('geburtsdatum', '')}\n"
            output += f"Blutgruppe: {patient_data.get('blutgruppe', 'Nicht angegeben')}\n"
            
            # Allergien
            allergien = patient_data.get('allergien', [])
            if allergien:
                output += f"\n⚠️ ALLERGIEN:\n"
                for allergie in allergien:
                    output += f"- {allergie}\n"
            
            # Chronische Erkrankungen
            erkrankungen = patient_data.get('chronische_erkrankungen', [])
            if erkrankungen:
                output += f"\n🏥 CHRONISCHE ERKRANKUNGEN:\n"
                for erkrankung in erkrankungen:
                    output += f"- {erkrankung}\n"
            
            # Aktuelle Medikation
            output += f"\n💊 AKTUELLE MEDIKATION:\n"
            medikation = patient_data.get('aktuelle_medikation', [])
            if medikation:
                for med in medikation:
                    output += f"- {med.get('medikament', '')}: "
                    output += f"{med.get('dosierung', '')} "
                    output += f"für {med.get('grund', '')}\n"
            else:
                output += "- Keine aktuelle Medikation\n"
            
            # Letzte Behandlungen
            output += f"\n🏥 LETZTE BEHANDLUNGEN:\n"
            behandlungen = patient_data.get('letzte_behandlungen', [])
            if behandlungen:
                for behandlung in sorted(behandlungen, key=lambda x: x.get('datum', ''), reverse=True):
                    output += f"Datum: {behandlung.get('datum', '')}\n"
                    output += f"Behandlung: {behandlung.get('behandlung', '')}\n"
                    output += f"Befund: {behandlung.get('befund', '')}\n\n"
            else:
                output += "- Keine Behandlungen dokumentiert\n"
            
            # Notfallkontakt
            notfall = patient_data.get('notfallkontakt', '')
            if notfall:
                output += f"\n📞 NOTFALLKONTAKT:\n{notfall}\n"
            
            output += "\n=== ENDE DER PATIENTENDATEN ==="
            
            logger.info(f"✅ Successfully loaded data for {name}")
            return output
            
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."

async def entrypoint(ctx: JobContext):
    """Medical agent entrypoint für LiveKit 1.0.0+"""
    logger.info(f"Medical agent starting in room: {ctx.room.name}")
    
    # Initialize state
    state = MedicalAgentState()
    
    await ctx.connect()
    
    # System prompt
    system_prompt = """Du bist Lisa von der Klinik St. Anna. ANTWORTE NUR AUF DEUTSCH.

DEINE AUFGABE:
1. Frage nach dem Namen des Patienten
2. Frage nach dem Geburtsjahr
3. Verwende 'load_patient_data' um ALLE Daten zu laden
4. Lies die kompletten Daten vor (Medikation und letzte Behandlungen)

WICHTIG:
- Erfinde NIEMALS Daten
- Wenn die Funktion Daten zurückgibt, lies sie KOMPLETT vor
- Vergiss keine Details"""

    # Model selection based on environment
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    
    if llm_provider == "openai":
        model = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.0
        )
    else:
        # Ollama configuration
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        model = openai.LLM(
            model=ollama_model,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1",
            api_key="ollama",  # Ollama doesn't need a real API key
            temperature=0.0
        )
    
    # Create function context with the new API
    fnc_ctx = llm.FunctionContext()
    
    # Register the function with proper decorator
    @fnc_ctx.ai_callable()
    async def load_patient_data_callable(
        name: str,
        birth_year: int
    ) -> str:
        """
        Lädt ALLE Patientendaten basierend auf Name und Geburtsjahr.
        
        Args:
            name: Vollständiger Name des Patienten
            birth_year: Geburtsjahr (z.B. 2010)
        
        Returns:
            Formatierte Patientendaten oder Fehlermeldung
        """
        return await load_patient_data(name, birth_year)
    
    # Configure TTS (immer OpenAI für bessere Qualität)
    tts = openai.TTS(
        model="tts-1",
        voice="nova",
        speed=1.0
    )
    
    # Configure STT
    stt = deepgram.STT(
        model="nova-2-general",
        language="de"
    )
    
    # Create assistant with new API
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=stt,
        llm=model,
        tts=tts,
        chat_ctx=llm.ChatContext([
            llm.ChatMessage(role="system", content=system_prompt)
        ]),
        fnc_ctx=fnc_ctx,
        min_endpointing_delay=0.5
    )
    
    # Set up event handlers
    @assistant.on("state_changed")
    def on_state_changed(state):
        logger.info(f"🤖 Agent state: {state}")
    
    @assistant.on("function_calls_finished")
    def on_function_finished(event):
        logger.info(f"✅ Function call completed")
    
    # Start the assistant
    assistant.start(ctx.room)
    
    # Initial greeting
    await asyncio.sleep(1)
    await assistant.say(
        "Guten Tag! Ich bin Lisa von der Klinik St. Anna.\n\n"
        "Ich werde Ihnen die kompletten Patientendaten vorlesen, "
        "inklusive der aktuellen Medikation und der letzten Behandlungen.\n\n"
        "Dafür benötige ich:\n"
        "1. Den vollständigen Namen des Patienten\n"
        "2. Das Geburtsjahr\n\n"
        "Wie heißt der Patient?",
        allow_interruptions=True
    )

# Entry point for the worker
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
