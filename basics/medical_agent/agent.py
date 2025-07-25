# LiveKit Agents - Medical Agent (Moderne API wie Garage Agent)
import logging
import os
import httpx
import asyncio
import re
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-medical-3")

@dataclass
class MedicalUserData:
    """User data context für den Medical Agent"""
    authenticated_doctor: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    active_patient: Optional[str] = None


class MedicalAssistant(Agent):
    """Medical Assistant mit korrekter API-Nutzung"""
    
    def __init__(self) -> None:
        # Instructions klar und präzise für Llama 3.2
        super().__init__(instructions="""Du bist ein medizinischer Assistent mit Zugriff auf die Patientendatenbank.

WORKFLOW:
1. Deine erste Begrüßung wird automatisch gesendet. NICHT nochmal begrüßen.
2. Warte auf Anfragen des Arztes zu Patientendaten.
3. Nutze IMMER die search_patient_data Funktion für Patientenanfragen.
4. Sage NIE "nicht gefunden" wenn die Funktion Daten zurückgibt.
5. Korrigiere Patienten-IDs automatisch: "p null null fünf" = "P005"

WICHTIGE REGELN:
- Immer auf Deutsch antworten
- Währungen als "15 Franken 50" statt "15.50"
- Präzise medizinische Informationen aus der Datenbank wiedergeben
- Niemals eigene medizinische Diagnosen stellen
- Die Datenbank enthält: Patienten-IDs, Diagnosen, Behandlungen, Medikation
- Keine technischen Details oder Funktionen erwähnen""")
        logger.info("✅ MedicalAssistant initialized")

    @function_tool
    async def search_patient_data(self, 
                                 context: RunContext[MedicalUserData],
                                 query: str) -> str:
        """
        Sucht in der Patientendatenbank nach Informationen.
        
        Args:
            query: Die Suchanfrage (z.B. Patienten-ID oder Symptome)
        """
        logger.info(f"🔍 Searching for: {query}")
        
        try:
            # Korrigiere Patienten-IDs
            processed_query = self._process_patient_id(query)
            logger.info(f"🔎 Processed query: {processed_query}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "medical",
                        "top_k": 5,  # Mehr Ergebnisse für bessere Trefferquote
                        "collection": "medical_nutrition"
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} results")
                        
                        # Speichere aktuelle Patienten-ID wenn gefunden
                        patient_match = re.search(r'P\d{3}', processed_query)
                        if patient_match:
                            context.userdata.active_patient = patient_match.group()
                        
                        # Formatiere die Ergebnisse
                        formatted = []
                        for i, result in enumerate(results[:3]):  # Max 3 Ergebnisse
                            content = result.get("content", "").strip()
                            if content:
                                # Formatiere für bessere Lesbarkeit
                                content = self._format_medical_data(content)
                                formatted.append(f"[{i+1}] {content}")
                        
                        response_text = "Hier sind die Patientendaten:\n\n"
                        response_text += "\n\n".join(formatted)
                        return response_text
                    
                    return "Zu dieser Anfrage konnte ich keine Daten in der Patientendatenbank finden."
                    
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    return "Es gab einen Fehler beim Zugriff auf die Datenbank. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Die Patientendatenbank ist momentan nicht erreichbar. Bitte versuchen Sie es später noch einmal."
    
    def _process_patient_id(self, text: str) -> str:
        """Korrigiert Sprache-zu-Text Fehler bei Patienten-IDs"""
        # Pattern für verschiedene Varianten
        patterns = [
            r'p\s*null\s*null\s*(\w+)',
            r'patient\s*null\s*null\s*(\w+)',
            r'p\s*0\s*0\s*(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                number = match.group(1)
                
                # Deutsche Zahlwörter zu Ziffern
                number_map = {
                    'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
                    'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 
                    'neun': '9', 'null': '0', 'zehn': '10'
                }
                
                if number in number_map:
                    number = number_map[number]
                
                # Erstelle korrekte ID
                corrected_id = f"P{number.zfill(3)}"
                text = re.sub(pattern, corrected_id, text, flags=re.IGNORECASE)
                logger.info(f"✅ Corrected patient ID to '{corrected_id}'")
                break
        
        return text
    
    def _format_medical_data(self, content: str) -> str:
        """Formatiert medizinische Daten für bessere Lesbarkeit"""
        # Ersetze Unterstriche durch Leerzeichen
        content = content.replace('_', ' ')
        
        # Formatiere Währungen
        content = re.sub(r'(\d+)\.(\d{2})', r'\1 Franken \2', content)
        
        return content


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point mit moderner API wie im Garage Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚀 Starting Medical Agent Session: {session_id}")
    logger.info("="*50)
    
    session = None  # Session Variable für Cleanup
    session_closed = False  # Flag um doppeltes Cleanup zu vermeiden
    
    # Register disconnect handler FIRST
    def on_disconnect():
        nonlocal session_closed
        logger.info(f"[{session_id}] Room disconnected event received")
        session_closed = True
    
    if ctx.room:
        ctx.room.on("disconnected", on_disconnect)
    
    try:
        # 1. Connect FIRST (wie im Garage Agent!)
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")
        
        # Debug: Room Status
        logger.info(f"Room participants: {len(ctx.room.remote_participants)}")
        logger.info(f"Local participant: {ctx.room.local_participant.identity}")
        
        # Debug: Track Event Handler
        @ctx.room.on("track_published")
        def on_track_published(publication, participant):
            logger.info(f"[{session_id}] Track published: {publication.kind} from {participant.identity}")
        
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            logger.info(f"[{session_id}] Track subscribed: {track.kind} from {participant.identity}")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # === KRITISCH: AUF AUDIO TRACK WARTEN ===
        audio_track_received = False
        max_wait_time = 10  # 10 Sekunden maximal warten
        
        for i in range(max_wait_time):
            # Check ob Audio Track da ist
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found: {track_pub.sid}")
                    audio_track_received = True
                    
                    # Log track status
                    logger.info(f"📡 [{session_id}] Audio track - subscribed: {track_pub.subscribed}, muted: {track_pub.muted}")
                    
                    break
            
            if audio_track_received:
                break
                
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
            await asyncio.sleep(1)
        
        if not audio_track_received:
            logger.error(f"❌ [{session_id}] No audio track received from user after {max_wait_time}s!")
            # Trotzdem fortfahren, aber mit Warnung
        
        # === ENDE AUDIO TRACK WAIT ===
        # WICHTIG: Kein participant.on() hier - RemoteParticipant hat diese Methode nicht!
        
        # 3. LLM-Konfiguration - NUR LLAMA 3.2!
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # Immer Llama 3.2 verwenden
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
        
        # 4. Create session with userdata (wie im Garage Agent!)
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                authenticated_doctor=None,
                rag_url=rag_url,
                active_patient=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.8,  # Höher für medizinische Präzision
                min_speech_duration=0.3    # Angepasst für klare Sprache
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"  # Professionelle Stimme für medizinischen Kontext
            )
        )
        
        # 5. Create agent instance
        agent = MedicalAssistant()
        
        # 6. WICHTIG: Kurze Pause vor Session-Start
        await asyncio.sleep(0.5)
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Debug Event Handler für Session
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User input transcribed: {event.transcript} (final: {event.is_final})")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state changed to: {event.state}")
        
        @session.on("user_state_changed")
        def on_user_state(event):
            logger.info(f"[{session_id}] 👤 User state changed to: {event.state}")
        
        # 8. Initiale Begrüßung erzwingen
        await asyncio.sleep(1.0)  # Warte bis Session vollständig initialisiert
        
        # Sende Begrüßung direkt über die Session
        initial_greeting = "Guten Tag Herr Doktor, welche Patientendaten benötigen Sie?"
        logger.info(f"📢 [{session_id}] Sending initial greeting: {initial_greeting}")
        
        # Nutze die Session's TTS direkt
        try:
            # Option 1: Wenn session.say verfügbar ist
            if hasattr(session, 'say'):
                await session.say(initial_greeting)
            else:
                # Option 2: Direkte TTS-Synthese und Audio-Ausgabe
                tts_audio = await session.tts.synthesize(initial_greeting)
                # LiveKit sendet das Audio automatisch an den Room
                logger.info(f"✅ [{session_id}] Initial greeting sent via TTS")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not send initial greeting: {e}")
        
        logger.info(f"✅ [{session_id}] Medical Agent ready and listening!")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error in medical agent: {e}", exc_info=True)
        raise
        
    finally:
        # WICHTIGES CLEANUP - Wird IMMER ausgeführt
        logger.info(f"🧹 [{session_id}] Starting session cleanup...")
        
        if session is not None and not session_closed:
            try:
                # NUR drain wenn Session noch aktiv ist
                if hasattr(session, '_activity') and session._activity:
                    logger.info(f"🛑 [{session_id}] Session still active, attempting drain...")
                    try:
                        # Mit Timeout um Hängen zu vermeiden
                        await asyncio.wait_for(session.drain(), timeout=2.0)
                        logger.info(f"✅ [{session_id}] Session drained")
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ [{session_id}] Session drain timed out after 2s")
                    except Exception as e:
                        logger.warning(f"⚠️ [{session_id}] Session drain failed: {e}")
                
                # Session beenden
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ [{session_id}] Error closing session: {e}")
        elif session_closed:
            logger.info(f"ℹ️ [{session_id}] Session already closed by disconnect event")
        
        # Disconnect vom Room wenn noch verbunden
        try:
            if ctx.room and hasattr(ctx.room, 'connection_state') and ctx.room.connection_state == "connected":
                await ctx.room.disconnect()
                logger.info(f"✅ [{session_id}] Disconnected from room")
        except Exception as e:
            logger.warning(f"⚠️ [{session_id}] Error disconnecting from room: {e}")
        
        # Explizit Garbage Collection erzwingen
        import gc
        gc.collect()
        logger.info(f"♻️ [{session_id}] Forced garbage collection")
        
        logger.info(f"✅ [{session_id}] Session cleanup complete")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
