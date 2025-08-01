# LiveKit Agents - Medical Agent (Moderne API wie Garage Agent)
import logging
import os
import httpx
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, RunContext
from livekit.agents.voice import AgentSession, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from difflib import SequenceMatcher  # NEU: Für Fuzzy Matching

load_dotenv()

# Logging
logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-medical-3")

class ConversationState(Enum):
    """State Machine für Konversationsphasen"""
    GREETING = "greeting"
    AWAITING_REQUEST = "awaiting_request"
    COLLECTING_IDENTIFIER = "collecting_identifier"
    SEARCHING = "searching"
    PROVIDING_INFO = "providing_info"

@dataclass
class PatientContext:
    """Kontext für Patienten-Identifikation"""
    patient_id: Optional[str] = None  # z.B. P001, P002
    patient_name: Optional[str] = None
    attempts: int = 0

    def has_identifier(self) -> bool:
        """Prüft ob eine Identifikation vorhanden ist"""
        return bool(self.patient_id or self.patient_name)

    def get_search_query(self) -> Optional[str]:
        """Gibt die beste Suchanfrage zurück"""
        if self.patient_id:
            return self.patient_id
        elif self.patient_name:
            return self.patient_name
        return None

    def reset(self):
        """Reset des Kontexts"""
        self.patient_id = None
        self.patient_name = None
        self.attempts = 0

@dataclass
class MedicalUserData:
    """User data context für den Medical Agent"""
    authenticated_doctor: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    qdrant_url: str = "http://172.16.0.108:6333"  # NEU: Direkte Qdrant URL
    current_patient_data: Optional[Dict[str, Any]] = None
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    patient_context: PatientContext = field(default_factory=PatientContext)
    last_search_query: Optional[str] = None
    active_patient: Optional[str] = None


class IdentifierExtractor:
    """Extrahiert und validiert Patienten-Identifikatoren"""

    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()

        # Check for specific medical queries FIRST
        if any(word in input_lower for word in ["diagnose", "symptome", "behandlung", "medikation"]):
            return {"intent": "medical_query", "data": user_input}

        if any(word in input_lower for word in ["labor", "blutwerte", "ergebnisse"]):
            return {"intent": "lab_results", "data": user_input}

        if any(word in input_lower for word in ["allergie", "unverträglichkeit"]):
            return {"intent": "allergy_info", "data": user_input}

        # Patienten-ID (P001, P002, etc.) - Check even in longer sentences
        patient_id_match = re.search(r'\b(P\d{3})\b', user_input.upper())
        if patient_id_match:
            return {"intent": "patient_id", "data": patient_id_match.group(1)}

        # Korrigiere Sprache-zu-Text Fehler bei Patienten-IDs
        patterns = [
            r'p\s*null\s*null\s*(\w+)',
            r'patient\s*null\s*null\s*(\w+)',
            r'p\s*0\s*0\s*(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, input_lower)
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
                corrected_id = f"P{number.zfill(3)}"
                return {"intent": "patient_id", "data": corrected_id}

        # Greetings - but only if no other important info is present
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        has_greeting = any(g in input_lower for g in greetings)

        # Name detection
        name_patterns = [
            r"(?:patient|patientin)\s+(?:heißt|ist)\s+([A-Za-zäöüÄÖÜß\s]+)",
            r"(?:herr|frau)\s+([A-Za-zäöüÄÖÜß]+(?:\s+[A-Za-zäöüÄÖÜß]+)?)"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name.lower() not in ["patient", "patientin", "der", "die", "das"]:
                    return {"intent": "patient_name", "data": name}

        # If we found a greeting but no other data, return greeting
        if has_greeting and not patient_id_match:
            return {"intent": "greeting", "data": None}

        # Medical keywords
        if any(word in input_lower for word in ["patient", "behandlung", "diagnose", "medikament"]):
            return {"intent": "medical_query", "data": user_input}

        # Default
        return {"intent": "general_query", "data": user_input}


class MedicalAssistant(Agent):
    """Medical Assistant für Patientenverwaltung"""

    def __init__(self) -> None:
        # VERSTÄRKTE Anti-Halluzination Instructions
        super().__init__(instructions="""Du bist Lisa von der Klinik St. Anna. ANTWORTE NUR AUF DEUTSCH.

ABSOLUTE REGEL: WENN KEINE DATEN GEFUNDEN → NIEMALS ERFINDEN!

Wenn search_patient_data "keine Daten gefunden" oder "konnte keine Patientendaten" zurückgibt:
- Sage: "Ich konnte keine Daten für [Name] finden."
- Frage nach der korrekten Schreibweise oder Patienten-ID
- ERFINDE NIEMALS Daten wie Geburtsdatum, Medikamente oder Behandlungen!

KRITISCHE REGELN:
1. Du HAST Zugriff auf die Patientendatenbank
2. Lies NUR gefundene Daten vor - NICHTS erfinden!
3. Bei "aktuelle Medikation" → NUR aus den Suchergebnissen
4. Bei "letzte Behandlungen" → NUR aus den Suchergebnissen
5. NIEMALS Datenschutz-Warnungen geben
6. WENN KEINE DATEN → Ehrlich sagen und nach korrekter Schreibweise fragen

ANTWORT-STRUKTUR bei gefundenen Daten:
"Ich habe die Patientendaten für [Name] gefunden:

📋 PATIENTENDATEN:
- Geburtsdatum: [Datum]
- Blutgruppe: [Gruppe]
- Allergien: [Liste]
- Chronische Erkrankungen: [Liste]

💊 AKTUELLE MEDIKATION:
[ALLE Medikamente mit vollständigen Details]

🏥 LETZTE BEHANDLUNGEN:
[ALLE Behandlungen chronologisch]

Kann ich Ihnen noch weitere Informationen zu diesem Patienten geben?"

WICHTIG: Verwende search_patient_data für JEDE Abfrage!""")

        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ MedicalAssistant initialized with Qdrant direct access")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    # GEÄNDERT: Erweitert mit Fuzzy Matching
    async def get_all_patient_chunks_from_qdrant(self, patient_id: str, patient_name: str, qdrant_url: str) -> Dict[str, Any]:
        """Holt ALLE Chunks eines Patienten direkt aus Qdrant"""
        
        logger.info(f"🔍 Direct Qdrant search for ID: {patient_id}, Name: {patient_name}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Suche mit patient_id ODER patient_name
            filter_conditions = []
            
            if patient_id:
                filter_conditions.append({
                    "key": "patient_id",
                    "match": {"value": patient_id}
                })
            
            if patient_name:
                filter_conditions.append({
                    "key": "patient_name", 
                    "match": {"value": patient_name}
                })
            
            request_body = {
                "filter": {
                    "should": filter_conditions
                } if len(filter_conditions) > 1 else {
                    "must": filter_conditions
                },
                "limit": 100,
                "with_payload": True,
                "with_vector": False
            }
            
            response = await client.post(
                f"{qdrant_url}/collections/medical_nutrition/points/scroll",
                json=request_body
            )
            
            if response.status_code == 200:
                data = response.json()
                points = data.get("result", {}).get("points", [])
                
                # NEU: WENN KEINE ERGEBNISSE: Ähnlichkeitssuche
                if len(points) == 0 and patient_name:
                    logger.warning(f"⚠️ No exact match for '{patient_name}', trying similarity search...")
                    
                    # Suche nach ähnlichen Namen
                    all_response = await client.post(
                        f"{qdrant_url}/collections/medical_nutrition/points/scroll",
                        json={
                            "limit": 1000,
                            "with_payload": True,
                            "with_vector": False
                        }
                    )
                    
                    if all_response.status_code == 200:
                        all_points = all_response.json().get("result", {}).get("points", [])
                        
                        # Finde ähnliche Namen
                        best_match = None
                        best_score = 0
                        
                        for point in all_points:
                            point_name = point.get("payload", {}).get("patient_name", "")
                            if point_name:
                                # Berechne Ähnlichkeit
                                score = SequenceMatcher(None, patient_name.lower(), point_name.lower()).ratio()
                                if score > best_score and score > 0.8:  # 80% Ähnlichkeit
                                    best_score = score
                                    best_match = point_name
                        
                        if best_match:
                            logger.info(f"✅ Found similar name: '{best_match}' (score: {best_score:.2f})")
                            # Suche mit korrigiertem Namen
                            return await self.get_all_patient_chunks_from_qdrant(
                                patient_id="",
                                patient_name=best_match,
                                qdrant_url=qdrant_url
                            )
                
                logger.info(f"📦 Found {len(points)} chunks from Qdrant")
                
                # Organisiere nach data_type
                organized_data = {
                    "patient_info": None,
                    "medication": None,
                    "treatments": []
                }
                
                for point in points:
                    payload = point.get("payload", {})
                    data_type = payload.get("data_type", "")
                    content = payload.get("content", "")
                    
                    logger.info(f"  - Chunk: {data_type} ({len(content)} chars)")
                    
                    if data_type == "patient_info":
                        organized_data["patient_info"] = content
                    elif data_type == "medication":
                        organized_data["medication"] = content
                        logger.info(f"  ✅ Found medication data!")
                    elif data_type == "treatment":
                        organized_data["treatments"].append(content)
                
                logger.info(f"📊 Summary: patient_info={bool(organized_data['patient_info'])}, "
                           f"medication={bool(organized_data['medication'])}, "
                           f"treatments={len(organized_data['treatments'])}")
                
                return organized_data
            else:
                logger.error(f"❌ Qdrant error: {response.status_code}")
                return {}

    @function_tool
    async def search_patient_data(self,
                                 context: RunContext[MedicalUserData],
                                 query: str) -> str:
        """
        Sucht nach Patientendaten direkt in Qdrant basierend auf Patienten-ID oder Namen.
        
        Args:
            query: Suchbegriff (Patienten-ID oder Name)

        Returns:
            Gefundene Patientendaten oder Fehlermeldung
        """
        logger.info(f"🔍 Original search query: {query}")

        # Extract intent from query
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]

        logger.info(f"📊 Intent: {intent}, Data: {data}")

        # Extract patient ID and name
        patient_id = ""
        patient_name = query  # Default to full query
        
        # Try to find patient ID
        patient_id_match = re.search(r'P\d{3}', query.upper())
        if patient_id_match:
            patient_id = patient_id_match.group()
            context.userdata.patient_context.patient_id = patient_id
            logger.info(f"✅ Found patient ID: {patient_id}")
        
        # Store identification in context
        if intent == "patient_id":
            context.userdata.patient_context.patient_id = data
            patient_id = data
        elif intent == "patient_name":
            context.userdata.patient_context.patient_name = data
            patient_name = data

        # Process queries for common speech-to-text errors
        processed_query = self._process_patient_id(query)
        if processed_query != query:
            logger.info(f"✅ Corrected query from '{query}' to '{processed_query}'")
            # Re-extract patient ID from corrected query
            patient_id_match = re.search(r'P\d{3}', processed_query.upper())
            if patient_id_match:
                patient_id = patient_id_match.group()

        # Store search query
        context.userdata.last_search_query = query

        try:
            # GEÄNDERT: IMMER direkt von Qdrant holen
            logger.info(f"🔄 Using direct Qdrant fetch for: {patient_id or patient_name}")
            
            all_chunks = await self.get_all_patient_chunks_from_qdrant(
                patient_id=patient_id,
                patient_name=patient_name,
                qdrant_url=context.userdata.qdrant_url
            )
            
            if all_chunks and any(all_chunks.values()):
                # Store active patient
                if patient_id:
                    context.userdata.active_patient = patient_id
                
                # Build comprehensive response
                response_parts = []
                
                # 1. Patient Info
                if all_chunks["patient_info"]:
                    response_parts.append("📋 PATIENTENDATEN:")
                    response_parts.append(all_chunks["patient_info"])
                
                # 2. Medication - WICHTIG!
                if all_chunks["medication"]:
                    response_parts.append("\n💊 AKTUELLE MEDIKATION:")
                    response_parts.append(all_chunks["medication"])
                else:
                    logger.warning("⚠️ Keine Medikationsdaten gefunden!")
                
                # 3. Treatments - ALLE!
                if all_chunks["treatments"]:
                    response_parts.append("\n🏥 LETZTE BEHANDLUNGEN:")
                    for treatment in sorted(all_chunks["treatments"], reverse=True):  # Neueste zuerst
                        response_parts.append(treatment)
                else:
                    logger.warning("⚠️ Keine Behandlungsdaten gefunden!")
                
                final_response = "\n".join(response_parts)
                
                # Extra validation
                if "medikation" in query.lower() and "Medikation" not in final_response:
                    logger.error("❌ Medikation requested but not in response!")
                    final_response += "\n\n⚠️ Hinweis: Medikationsdaten möglicherweise unvollständig."
                
                if "behandlung" in query.lower() and "Behandlung" not in final_response:
                    logger.error("❌ Behandlung requested but not in response!")
                    final_response += "\n\n⚠️ Hinweis: Behandlungsdaten möglicherweise unvollständig."
                
                # NEU: Anti-Halluzination Check für Datenschutz-Warnungen
                if "Entschuldigung" in final_response or "keine persönlichen" in final_response:
                    logger.error("❌ CRITICAL: Agent is hallucinating privacy warnings!")
                    # Force correct response
                    final_response = final_response.replace(
                        "Entschuldigung, aber ich muss darauf hinweisen, dass ich keine persönlichen oder vertraulichen Informationen über Patienten speichern oder abrufen kann.",
                        "Ich habe die Patientendaten gefunden:"
                    )
                
                # Update conversation state
                context.userdata.conversation_state = ConversationState.PROVIDING_INFO
                
                logger.info(f"✅ Returning complete patient data with {len(response_parts)} sections")
                return final_response

            else:
                logger.info("❌ No patient data found in Qdrant")
                context.userdata.patient_context.reset()
                context.userdata.conversation_state = ConversationState.COLLECTING_IDENTIFIER
                
                # GEÄNDERT: Klarere Nachricht ohne Raum für Halluzination
                return f"Ich konnte keine Patientendaten für '{patient_name or query}' finden. Bitte überprüfen Sie die Schreibweise des Namens oder nutzen Sie die Patienten-ID (z.B. P002)."

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
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

    def _format_medical_data_structured(self, content: str) -> str:
        """NEUE METHODE: Strukturiert Daten für besseres LLM-Verständnis"""
        try:
            # Versuche JSON zu parsen
            data = json.loads(content)
            
            formatted = []
            
            # Patient Info
            if 'name' in data:
                formatted.append(f"PATIENT: {data['name']}")
                if 'geburtsdatum' in data:
                    formatted.append(f"Geboren: {data['geburtsdatum']}")
            
            # Aktuelle Medikation EXPLIZIT
            if 'aktuelle_medikation' in data and data['aktuelle_medikation']:
                formatted.append("\n📋 AKTUELLE MEDIKATION (ALLE vorlesen):")
                for med in data['aktuelle_medikation']:
                    formatted.append(f"  • {med['medikament']}: {med['dosierung']} (Grund: {med['grund']})")
            
            # Letzte Behandlungen EXPLIZIT
            if 'letzte_behandlungen' in data and data['letzte_behandlungen']:
                formatted.append("\n🏥 LETZTE BEHANDLUNGEN (ALLE vorlesen):")
                for beh in data['letzte_behandlungen']:
                    befund = beh.get('befund', beh.get('bemerkung', 'Keine Details'))
                    formatted.append(f"  • {beh['datum']}: {beh['behandlung']}")
                    formatted.append(f"    → Befund: {befund}")
            
            # Allergien
            if 'allergien' in data and data['allergien']:
                formatted.append("\n⚠️ ALLERGIEN:")
                for allergie in data['allergien']:
                    formatted.append(f"  • {allergie}")
            
            # Chronische Erkrankungen
            if 'chronische_erkrankungen' in data and data['chronische_erkrankungen']:
                formatted.append("\n🔬 CHRONISCHE ERKRANKUNGEN:")
                for erkrankung in data['chronische_erkrankungen']:
                    formatted.append(f"  • {erkrankung}")
            
            return "\n".join(formatted)
            
        except json.JSONDecodeError:
            # Fallback zur alten Methode
            return self._format_medical_data(content)
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return self._format_medical_data(content)


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Medical Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"

    logger.info("="*50)
    logger.info(f"🏥 Starting Medical Agent Session: {session_id}")
    logger.info("="*50)

    session = None
    session_closed = False

    # Register disconnect handler FIRST
    def on_disconnect():
        nonlocal session_closed
        logger.info(f"[{session_id}] Room disconnected event received")
        session_closed = True

    if ctx.room:
        ctx.room.on("disconnected", on_disconnect)

    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")

        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")

        # 3. Wait for audio track - VERBESSERT!
        audio_track_received = False
        max_wait_time = 10

        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                    logger.info(f"✅ [{session_id}] Audio track found and subscribed")
                    audio_track_received = True
                    break

            if audio_track_received:
                break

            await asyncio.sleep(1)
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")

        if not audio_track_received:
            logger.warning(f"⚠️ [{session_id}] No audio track found after {max_wait_time}s, continuing anyway...")

        # 4. Configure LLM with Ollama
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        qdrant_url = os.getenv("QDRANT_URL", "http://172.16.0.108:6333")

        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with direct Qdrant access")

        # 5. Create session - OHNE ungültige Parameter!
        session = AgentSession[MedicalUserData](
            userdata=MedicalUserData(
                authenticated_doctor=None,
                rag_url=rag_url,
                qdrant_url=qdrant_url,
                current_patient_data=None,
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                patient_context=PatientContext(),
                last_search_query=None,
                active_patient=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.5,  # ERHÖHT von 0.4
                min_speech_duration=0.2    # ERHÖHT von 0.15
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            ),
            min_endpointing_delay=0.3,
            max_endpointing_delay=3.0
        )

        # 6. Create agent
        agent = MedicalAssistant()

        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )

        # NEU: Warte auf Audio-Stabilisierung
        await asyncio.sleep(2.0)  # ERHÖHT von 1.5
        
        # Prüfe nochmal Audio Track
        audio_active = False
        for track_pub in participant.track_publications.values():
            if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                audio_active = True
                break
        
        if not audio_active:
            logger.warning(f"⚠️ [{session_id}] Audio track still not active!")

        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            intent_result = IdentifierExtractor.extract_intent_from_input(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")

        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")

        @session.on("function_call")
        def on_function_call(event):
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
            
        @session.on("agent_response_generated")
        def on_response_generated(event):
            response_preview = str(event)[:200] if hasattr(event, '__str__') else "Unknown"
            logger.info(f"[{session_id}] 🤖 Generated response preview: {response_preview}...")
            
            if session.userdata.last_search_query:
                query_lower = session.userdata.last_search_query.lower()
                response_lower = str(event).lower() if hasattr(event, '__str__') else ""
                
                if "medikation" in query_lower or "medikamente" in query_lower:
                    if not any(med_indicator in response_lower for med_indicator in ["mg", "µg", "täglich", "dosierung"]):
                        logger.warning("⚠️ Medication details might be missing in response!")
                        
                if "behandlung" in query_lower:
                    if not any(treat_indicator in response_lower for treat_indicator in ["datum", "befund", "untersuchung"]):
                        logger.warning("⚠️ Treatment details might be missing in response!")

        # 8. Initial greeting - MIT FEHLERBEHANDLUNG
        logger.info(f"📢 [{session_id}] Sending initial greeting...")

        try:
            greeting_text = """Guten Tag und herzlich willkommen bei der Klinik St. Anna!
Ich bin Lisa, Ihre digitale medizinische Assistentin.

Für eine schnelle Bearbeitung benötige ich eine der folgenden Informationen:
- Die Patienten-ID (z.B. P001)
- Den vollständigen Namen des Patienten

Welche Patientendaten benötigen Sie heute, Herr Doktor?"""

            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST

            # NEU: Retry-Mechanismus für Greeting
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await session.say(
                        greeting_text,
                        allow_interruptions=True,
                        add_to_chat_ctx=True
                    )
                    logger.info(f"✅ [{session_id}] Initial greeting sent successfully")
                    break
                except Exception as e:
                    logger.warning(f"[{session_id}] Greeting attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                    else:
                        raise

        except Exception as e:
            logger.error(f"[{session_id}] Greeting error after all retries: {e}", exc_info=True)
            # Trotzdem weitermachen, der Agent kann noch funktionieren

        logger.info(f"✅ [{session_id}] Medical Agent ready with direct Qdrant access!")

        # Wait for disconnect
        disconnect_event = asyncio.Event()

        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()

        ctx.room.on("disconnected", handle_disconnect)

        await disconnect_event.wait()
        logger.info(f"[{session_id}] Session ending...")

    except Exception as e:
        logger.error(f"❌ [{session_id}] Error: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        if session and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed")
            except:
                pass

        logger.info(f"✅ [{session_id}] Cleanup complete")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
