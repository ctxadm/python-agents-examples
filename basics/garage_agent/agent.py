# LiveKit Agents 1.0.x Version - Secure Garage Agent with Fuzzy Search
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
from difflib import get_close_matches, SequenceMatcher  # NEU: Fuzzy Search imports

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
5. Bei Fuzzy Match Vorschlag: Wenn gefragt "Meinten Sie {Name}?" und der Kunde mit "Ja" antwortet, authentifiziere mit dem vorgeschlagenen Namen.

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
        
        # NEU: Fuzzy Match tracking
        self.pending_fuzzy_match = None
        
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
        """Authentifiziert einen Kunden anhand des Namens mit Fuzzy Search Support.
        
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
            
            # Erste Suche mit dem Namen
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": f"besitzer: {customer_name}",  # Nur der Name, kein Präfix
                        "agent_type": "garage",
                        "top_k": 10,  # Mehr Ergebnisse für besseres Fuzzy Matching
                        "collection": "automotive_docs"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Sammle alle Besitzer-Namen aus den Ergebnissen
                    found_names = []
                    for result in results:
                        content = result.get("content", "")
                        # Extrahiere Namen aus dem Content (suche nach "besitzer" Feld)
                        besitzer_matches = re.findall(r'"besitzer":\s*"([^"]+)"', content)
                        found_names.extend(besitzer_matches)
                    
                    # Entferne Duplikate
                    found_names = list(set(found_names))
                    logger.info(f"Found {len(found_names)} unique customer names in database")
                    
                    # 1. Prüfe exakte Übereinstimmung (case-insensitive)
                    for name in found_names:
                        if customer_name.lower() == name.lower():
                            # Exakte Übereinstimmung gefunden
                            self.authenticated_customer = name
                            self.failed_auth_attempts = 0
                            self.last_activity_time = datetime.now()
                            self._log_access("authentication", "success", {"customer": name, "match_type": "exact"})
                            logger.info(f"Customer {name} successfully authenticated (exact match)")
                            return f"Vielen Dank für die Identifikation, {name}. Wie kann ich Ihnen heute helfen?"
                    
                    # 2. Keine exakte Übereinstimmung - versuche Fuzzy Matching
                    if found_names:
                        # Finde die ähnlichsten Namen
                        close_matches = get_close_matches(customer_name, found_names, n=3, cutoff=0.7)
                        
                        if close_matches:
                            best_match = close_matches[0]
                            similarity = SequenceMatcher(None, customer_name.lower(), best_match.lower()).ratio()
                            
                            if similarity >= 0.85:
                                # Sehr ähnlich - automatisch akzeptieren
                                self.authenticated_customer = best_match
                                self.failed_auth_attempts = 0
                                self.last_activity_time = datetime.now()
                                self._log_access("authentication", "success", {
                                    "customer": best_match, 
                                    "match_type": "fuzzy_auto",
                                    "similarity": f"{similarity:.2%}"
                                })
                                logger.info(f"Customer {best_match} authenticated via fuzzy match (similarity: {similarity:.2%})")
                                return f"Vielen Dank für die Identifikation, {best_match}. Wie kann ich Ihnen heute helfen?"
                            
                            elif similarity >= 0.7:
                                # Ähnlich genug zum Nachfragen
                                self.pending_fuzzy_match = best_match  # Speichere für spätere Bestätigung
                                self._log_access("authentication", "fuzzy_suggestion", {
                                    "suggested": best_match,
                                    "similarity": f"{similarity:.2%}"
                                })
                                return f"Meinten Sie {best_match}? Bitte bestätigen Sie mit 'Ja' oder nennen Sie Ihren Namen erneut."
                    
                    # 3. Versuche Nachnamen-Suche
                    if " " in customer_name:
                        last_name = customer_name.split()[-1]
                        # Suche alle Namen mit diesem Nachnamen
                        matching_lastnames = [name for name in found_names if last_name.lower() in name.lower()]
                        
                        if len(matching_lastnames) == 1:
                            # Nur eine Person mit diesem Nachnamen
                            matched_name = matching_lastnames[0]
                            self.authenticated_customer = matched_name
                            self.failed_auth_attempts = 0
                            self.last_activity_time = datetime.now()
                            self._log_access("authentication", "success", {
                                "customer": matched_name,
                                "match_type": "lastname"
                            })
                            logger.info(f"Customer {matched_name} authenticated via lastname match")
                            return f"Vielen Dank für die Identifikation, {matched_name}. Wie kann ich Ihnen heute helfen?"
                        
                        elif len(matching_lastnames) > 1:
                            # Mehrere Personen mit diesem Nachnamen
                            self._log_access("authentication", "multiple_matches", {
                                "lastname": last_name,
                                "matches": matching_lastnames
                            })
                            names_list = ", ".join(matching_lastnames[:3])  # Zeige max 3 Namen
                            return f"Ich habe mehrere Kunden mit dem Nachnamen {last_name} gefunden: {names_list}. Bitte nennen Sie Ihren vollständigen Namen."
                    
                    # 4. Keine Übereinstimmung gefunden
                    self.failed_auth_attempts += 1
                    self._log_access("authentication", "failed", {
                        "attempt": self.failed_auth_attempts,
                        "searched_for": customer_name
                    })
                    
                    remaining_attempts = self.max_failed_attempts - self.failed_auth_attempts
                    if remaining_attempts > 0:
                        # Gebe hilfreichen Hinweis
                        if found_names and len(found_names) < 20:
                            # Zeige ähnliche Namen als Hilfe
                            similar_initials = [n for n in found_names if n[0].lower() == customer_name[0].lower()][:3]
                            if similar_initials:
                                hint = f" Ähnliche Namen beginnen mit '{customer_name[0].upper()}'."
                            else:
                                hint = ""
                        else:
                            hint = ""
                        return f"Kunde nicht gefunden. Sie haben noch {remaining_attempts} Versuche.{hint}"
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
            customer_filtered_query = f"{query} besitzer:{self.authenticated_customer}"  # Verwende "besitzer" statt "Kunde"
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
        self.pending_fuzzy_match = None  # NEU: Reset auch pending fuzzy match
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
