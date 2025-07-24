# LiveKit Agents 1.0.x Version - Secure Garage Agent
import logging
import os
import httpx
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from typing import Optional, Dict, Any

load_dotenv()

logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.INFO)

class GarageAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Du bist ein professioneller Werkstatt-Assistent mit strikten Datenschutz-Richtlinien.

## AUTHENTIFIZIERUNG (OBLIGATORISCH)
1. ERSTE ANTWORT: "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen zur Identifikation."
2. Nach Namensnennung: Validiere IMMER mit authenticate_customer("{Kundenname}")
3. Wenn Kunde nicht gefunden: "Entschuldigung, ich konnte Sie nicht in unserem System finden. Bitte wenden Sie sich an unsere Rezeption."
4. Wenn Kunde gefunden: "Guten Tag Herr/Frau {Name}, wie kann ich Ihnen heute helfen?"

## DATENSCHUTZ-REGELN (STRIKT EINHALTEN)
- NUR Informationen über den authentifizierten Kunden und SEINE Fahrzeuge
- NIEMALS Daten anderer Kunden erwähnen oder preisgeben
- Bei Fragen zu anderen Fahrzeugen/Kunden: "Aus Datenschutzgründen kann ich nur Auskunft über Ihre eigenen Fahrzeuge geben."
- Keine allgemeinen Werkstatt-Statistiken oder Vergleiche mit anderen Kunden

## ERLAUBTE FUNKTIONEN (NUR NACH AUTHENTIFIZIERUNG)
- Fahrzeugdaten des Kunden abrufen (Kennzeichen, Marke, Modell)
- Wartungshistorie der Kundenfahrzeuge einsehen
- Reparaturkosten für Kundenfahrzeuge kalkulieren
- Termine für den Kunden vereinbaren

## VERBOTENE THEMEN
- Persönliche Gespräche oder Small Talk
- Allgemeine Autowartungstipps (verweise auf Fachpersonal)
- Preisvergleiche mit anderen Werkstätten
- Technische Diagnosen ohne Fahrzeuginspektion
- Jegliche Informationen über andere Kunden oder deren Fahrzeuge

## KOMMUNIKATIONSREGELN
- Professionell und höflich bleiben
- Bei unklaren Anfragen: "Können Sie Ihre Anfrage präzisieren?"
- Bei nicht-werkstattbezogenen Fragen: "Ich bin ausschließlich für Werkstattanfragen zuständig."
- Nutze IMMER search_vehicle_data für ALLE Datenanfragen
- Korrigiere Kennzeichen: "ZH 12345" statt "zh eins zwei drei vier fünf"
- Währungsangaben: "850 Franken" statt "850.00"

## SICHERHEITSPROTOKOLL
- Bei verdächtigen Anfragen sofort abbrechen
- Keine Auskunft ohne erfolgreiche Authentifizierung
- Session beenden bei wiederholten unauthorisierten Zugriffsversuchen
- Dokumentiere alle Datenzugriffe intern

WICHTIG: Die Kundenauthentifizierung ist NICHT optional. Ohne bestätigten Kundennamen keine Datenweitergabe!
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
        
        # Session Management
        self.authenticated_customer = None
        self.session_id = None
        self.session_start_time = None
        self.last_activity_time = None
        self.failed_auth_attempts = 0
        self.max_failed_attempts = 3
        self.session_timeout_minutes = 30
        
        # Security Logging
        self.access_log = []
        
        logger.info(f"Secure Garage Agent initialized with RAG service at {self.rag_url}")
        
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("Garage Agent entering session")
        self.session_id = f"session_{int(time.time())}"
        self.session_start_time = datetime.now()
        self.last_activity_time = datetime.now()
        
        await self.session.say(
            "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen zur Identifikation.",
            allow_interruptions=True
        )
    
    @function_tool
    async def authenticate_customer(self, customer_name: str) -> str:
        """Authentifiziert einen Kunden anhand des Namens.
        
        Args:
            customer_name: Der Name des Kunden
            
        Returns:
            Authentifizierungsstatus und Kundendaten
        """
        try:
            # Check if already too many failed attempts
            if self.failed_auth_attempts >= self.max_failed_attempts:
                logger.warning(f"Too many failed auth attempts for session {self.session_id}")
                return "Zu viele fehlgeschlagene Anmeldeversuche. Bitte wenden Sie sich an die Rezeption."
            
            logger.info(f"Authenticating customer: {customer_name}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": f"Kunde: {customer_name}",
                        "agent_type": "garage",
                        "top_k": 1,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results and any(customer_name.lower() in str(r).lower() for r in results):
                        # Successful authentication
                        self.authenticated_customer = customer_name
                        self.failed_auth_attempts = 0
                        self.last_activity_time = datetime.now()
                        
                        # Log successful authentication
                        self._log_access("authentication", "success", {"customer": customer_name})
                        
                        logger.info(f"Customer {customer_name} successfully authenticated")
                        return f"Vielen Dank für die Identifikation, {customer_name}. Wie kann ich Ihnen heute helfen?"
                    else:
                        # Failed authentication
                        self.failed_auth_attempts += 1
                        self._log_access("authentication", "failed", {"attempt": self.failed_auth_attempts})
                        
                        remaining_attempts = self.max_failed_attempts - self.failed_auth_attempts
                        if remaining_attempts > 0:
                            return f"Kunde nicht gefunden. Sie haben noch {remaining_attempts} Versuche."
                        else:
                            return "Kunde nicht gefunden. Maximale Anzahl von Versuchen erreicht."
                else:
                    logger.error(f"Authentication API error: {response.status_code}")
                    return "Authentifizierungssystem momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return "Fehler bei der Authentifizierung. Bitte versuchen Sie es erneut."
    
    @function_tool
    async def search_vehicle_data(self, query: str) -> str:
        """Sucht in der Fahrzeugdatenbank nach Informationen.
        
        Args:
            query: Die Suchanfrage (z.B. Kennzeichen, Fahrzeugtyp, Problem)
            
        Returns:
            Die gefundenen Fahrzeugdaten
        """
        # Security check - ensure customer is authenticated
        if not self._check_authentication():
            return "Bitte authentifizieren Sie sich zuerst mit Ihrem Namen."
        
        # Check session timeout
        if self._is_session_expired():
            self._reset_session()
            return "Ihre Sitzung ist abgelaufen. Bitte authentifizieren Sie sich erneut."
        
        try:
            # Update last activity time
            self.last_activity_time = datetime.now()
            
            # Add customer filter to query for security
            customer_filtered_query = f"{query} Kunde:{self.authenticated_customer}"
            processed_query = self._process_license_plate(customer_filtered_query)
            
            logger.info(f"Searching vehicles for authenticated customer {self.authenticated_customer}: {processed_query}")
            
            # Log data access
            self._log_access("vehicle_search", "attempt", {"query": query})
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "garage",
                        "top_k": 3,
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Filter results to ensure they belong to authenticated customer
                    filtered_results = []
                    for result in results:
                        content = result.get("content", "")
                        # Only include results that seem to belong to the authenticated customer
                        if self.authenticated_customer.lower() in content.lower():
                            filtered_results.append(result)
                    
                    if filtered_results:
                        logger.info(f"Found {len(filtered_results)} vehicle results for {self.authenticated_customer}")
                        self._log_access("vehicle_search", "success", {"results_count": len(filtered_results)})
                        
                        formatted = []
                        for i, result in enumerate(filtered_results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted)
                    else:
                        self._log_access("vehicle_search", "no_results", {"query": query})
                        return "Keine Fahrzeugdaten zu dieser Anfrage für Ihr Konto gefunden."
                else:
                    logger.error(f"RAG search failed: {response.status_code}")
                    self._log_access("vehicle_search", "error", {"status_code": response.status_code})
                    return "Fehler beim Zugriff auf die Fahrzeugdatenbank."
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            self._log_access("vehicle_search", "exception", {"error": str(e)})
            return "Die Fahrzeugdatenbank ist momentan nicht erreichbar."
    
    def _check_authentication(self) -> bool:
        """Prüft ob ein Kunde authentifiziert ist"""
        if not self.authenticated_customer:
            logger.warning(f"Unauthenticated access attempt in session {self.session_id}")
            self._log_access("unauthorized_access", "blocked", {})
            return False
        return True
    
    def _is_session_expired(self) -> bool:
        """Prüft ob die Session abgelaufen ist"""
        if not self.last_activity_time:
            return True
        
        time_since_activity = datetime.now() - self.last_activity_time
        return time_since_activity > timedelta(minutes=self.session_timeout_minutes)
    
    def _reset_session(self):
        """Setzt die Session zurück"""
        logger.info(f"Resetting session {self.session_id}")
        self.authenticated_customer = None
        self.failed_auth_attempts = 0
        self.last_activity_time = None
        self._log_access("session_reset", "timeout", {})
    
    def _log_access(self, action: str, status: str, details: Dict[str, Any]):
        """Loggt alle Zugriffe für Audit-Zwecke"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "action": action,
            "status": status,
            "customer": self.authenticated_customer,
            "details": details
        }
        self.access_log.append(log_entry)
        logger.info(f"Access log: {log_entry}")
    
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
    logger.info("=== Secure Garage Agent Starting (1.0.x) ===")
    
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
    
    logger.info("Secure Garage Agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
