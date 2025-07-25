# LiveKit Agents 1.0.x Version - Secure Garage Agent with Fuzzy Search - MISTRAL v0.3
import logging
import os
import httpx
import re
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from typing import Optional, Dict, Any
from difflib import get_close_matches, SequenceMatcher

load_dotenv()

# Enhanced logging
logger = logging.getLogger("garage-assistant")
logger.setLevel(logging.DEBUG)

# Add console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Enable LiveKit debug logs
logging.getLogger("livekit").setLevel(logging.DEBUG)

# Enable Mistral debugging
os.environ['MISTRAL_DEBUG'] = 'true'

class GarageAgent(Agent):
    def __init__(self):
        logger.info("Initializing GarageAgent with Mistral v0.3...")
        
        super().__init__(
            instructions="""Du bist ein professioneller Werkstatt-Assistent mit strikten Datenschutz-Richtlinien.

## WICHTIG FÜR MISTRAL FUNCTION CALLING:
- Bei Funktionsaufrufen IMMER das exakte Format verwenden
- Funktionsparameter müssen EXAKT den definierten Namen entsprechen
- search_vehicle_data benötigt Parameter "query" als String
- authenticate_customer benötigt Parameter "customer_name" als String
- confirm_fuzzy_match benötigt Parameter "confirmed_name" als String

## FUNCTION CALLING BEISPIELE:
- RICHTIG: search_vehicle_data mit query="nächster Service"
- RICHTIG: authenticate_customer mit customer_name="Marco Rossi"
- FALSCH: search_vehicle_data mit vehicle_id=123 oder id="xyz"

## AUTHENTIFIZIERUNG (OBLIGATORISCH)
1. ERSTE ANTWORT: "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen zur Identifikation."
2. Nach Namensnennung: Validiere IMMER mit authenticate_customer("{Kundenname}")
3. Wenn Kunde nicht gefunden: "Entschuldigung, ich konnte Sie nicht in unserem System finden. Bitte wenden Sie sich an unsere Rezeption."
4. Wenn Kunde gefunden: "Guten Tag Herr/Frau {Name}, wie kann ich Ihnen heute helfen?"
5. Bei Fuzzy Match Vorschlag: Wenn gefragt "Meinten Sie {Name}?" und der Kunde mit "Ja" antwortet, authentifiziere mit dem vorgeschlagenen Namen.

## FUZZY MATCH HANDLING
- Wenn pending_fuzzy_match gesetzt ist und Kunde "Ja" sagt, authentifiziere mit dem vorgeschlagenen Namen
- Bei "Ja"-Antwort nach Fuzzy-Vorschlag: Nutze authenticate_customer mit dem vorgeschlagenen Namen

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
                model="mistral:v0.3",  # Mistral v0.3
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                temperature=0.3,  # Niedriger für Mistral
                max_tokens=2048,  # Mistral bevorzugt kürzere Antworten
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
        
        # Fuzzy Match tracking
        self.pending_fuzzy_match = None
        
        # Security Logging
        self.access_log = []
        
        # Debug tracking
        self.transcription_count = 0
        self.vad_event_count = 0
        
        logger.info(f"✅ Secure Garage Agent initialized with Mistral v0.3 and RAG service at {self.rag_url}")
        
    async def on_enter(self):
        """Called when agent enters session"""
        logger.info("🎤 Garage Agent entering session")
        self.session_id = f"session_{int(time.time())}"
        self.session_start_time = datetime.now()
        self.last_activity_time = datetime.now()
        
        logger.debug(f"Session ID: {self.session_id}")
        logger.debug(f"Session start time: {self.session_start_time}")
        
        await self.session.say(
            "Willkommen in der Werkstatt! Bitte nennen Sie mir Ihren Namen zur Identifikation.",
            allow_interruptions=True
        )
        logger.info("✅ Initial greeting sent")
    
    # Debug: Override message handling to log transcriptions
    async def on_message_received(self, message: str):
        """Debug: Log when message is received"""
        self.transcription_count += 1
        logger.info(f"📝 Transcription #{self.transcription_count} received: '{message}'")
        
        # Check for fuzzy match confirmation
        if self.pending_fuzzy_match and message.lower() in ["ja", "ja bitte", "genau", "richtig", "korrekt"]:
            logger.info(f"✅ Fuzzy match confirmed for: {self.pending_fuzzy_match}")
            # Authenticate with the pending fuzzy match
            await self.authenticate_customer(self.pending_fuzzy_match)
            self.pending_fuzzy_match = None
            return
            
        await super().on_message_received(message)
    
    # Debug: Log VAD events
    async def on_vad_event(self, speaking: bool):
        """Debug: Log VAD events"""
        self.vad_event_count += 1
        logger.debug(f"🎙️ VAD Event #{self.vad_event_count}: Speaking = {speaking}")
    
    @function_tool
    async def confirm_fuzzy_match(self, confirmed_name: str) -> str:
        """Bestätigt einen Fuzzy Match Namen.
        
        MISTRAL WICHTIG: Der Parameter heißt 'confirmed_name' (String).
        
        Args:
            confirmed_name: Der bestätigte Name als String
            
        Returns:
            Bestätigung der Authentifizierung
        """
        logger.info(f"Confirming fuzzy match for: {confirmed_name}")
        return await self.authenticate_customer(confirmed_name)
    
    @function_tool
    async def authenticate_customer(self, customer_name: str) -> str:
        """Authentifiziert einen Kunden anhand des Namens mit Fuzzy Search Support.
        
        MISTRAL WICHTIG: Der Parameter heißt 'customer_name' (String).
        
        Verwende diese Funktion für:
        - Kundenidentifikation: customer_name="Marco Rossi"
        - Nach Namensnennung: customer_name="Thomas Meier"
        
        Args:
            customer_name: Der Name des Kunden als String
            
        Returns:
            Authentifizierungsstatus und Kundendaten
        """
        logger.info(f"🔐 Starting authentication for: '{customer_name}'")
        logger.debug(f"Parameter type check - customer_name: {type(customer_name)}")
        
        # Mistral-spezifische Validierung
        if not isinstance(customer_name, str):
            logger.warning(f"Mistral sent non-string parameter: {type(customer_name)}")
            customer_name = str(customer_name)
        
        try:
            # Check if already too many failed attempts
            if self.failed_auth_attempts >= self.max_failed_attempts:
                logger.warning(f"❌ Too many failed auth attempts for session {self.session_id}")
                return "Zu viele fehlgeschlagene Anmeldeversuche. Bitte wenden Sie sich an die Rezeption."
            
            # Clear pending fuzzy match if authenticating with a new name
            if self.pending_fuzzy_match and customer_name != self.pending_fuzzy_match:
                logger.debug("Clearing pending fuzzy match due to new authentication attempt")
                self.pending_fuzzy_match = None
            
            # Erste Suche mit dem Namen
            logger.debug(f"Searching RAG for customer: {customer_name}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": f"besitzer: {customer_name}",
                        "agent_type": "garage",
                        "top_k": 10,
                        "collection": "automotive_docs"
                    }
                )
                
                logger.debug(f"RAG response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    logger.debug(f"Found {len(results)} results from RAG")
                    
                    # Sammle alle Besitzer-Namen aus den Ergebnissen
                    found_names = []
                    for result in results:
                        content = result.get("content", "")
                        # Extrahiere Namen aus dem Content
                        besitzer_matches = re.findall(r'Besitzer:\s*([^\n]+)', content)
                        found_names.extend(besitzer_matches)
                    
                    # Entferne Duplikate
                    found_names = list(set(found_names))
                    logger.info(f"📋 Found {len(found_names)} unique customer names in database")
                    logger.debug(f"Names found: {found_names[:5]}...")  # Show first 5 for debugging
                    
                    # 1. Prüfe exakte Übereinstimmung (case-insensitive)
                    for name in found_names:
                        if customer_name.lower() == name.lower():
                            # Exakte Übereinstimmung gefunden
                            self.authenticated_customer = name
                            self.failed_auth_attempts = 0
                            self.last_activity_time = datetime.now()
                            self.pending_fuzzy_match = None  # Clear any pending match
                            self._log_access("authentication", "success", {"customer": name, "match_type": "exact"})
                            logger.info(f"✅ Customer {name} successfully authenticated (exact match)")
                            return f"Vielen Dank für die Identifikation, {name}. Wie kann ich Ihnen heute helfen?"
                    
                    # 2. Keine exakte Übereinstimmung - versuche Fuzzy Matching
                    if found_names:
                        logger.debug("No exact match found, trying fuzzy matching...")
                        close_matches = get_close_matches(customer_name, found_names, n=3, cutoff=0.7)
                        logger.debug(f"Close matches: {close_matches}")
                        
                        if close_matches:
                            best_match = close_matches[0]
                            similarity = SequenceMatcher(None, customer_name.lower(), best_match.lower()).ratio()
                            logger.debug(f"Best match: {best_match} (similarity: {similarity:.2%})")
                            
                            if similarity >= 0.85:
                                # Sehr ähnlich - automatisch akzeptieren
                                self.authenticated_customer = best_match
                                self.failed_auth_attempts = 0
                                self.last_activity_time = datetime.now()
                                self.pending_fuzzy_match = None
                                self._log_access("authentication", "success", {
                                    "customer": best_match, 
                                    "match_type": "fuzzy_auto",
                                    "similarity": f"{similarity:.2%}"
                                })
                                logger.info(f"✅ Customer {best_match} authenticated via fuzzy match (similarity: {similarity:.2%})")
                                return f"Vielen Dank für die Identifikation, {best_match}. Wie kann ich Ihnen heute helfen?"
                            
                            elif similarity >= 0.7:
                                # Ähnlich genug zum Nachfragen
                                self.pending_fuzzy_match = best_match
                                self._log_access("authentication", "fuzzy_suggestion", {
                                    "suggested": best_match,
                                    "similarity": f"{similarity:.2%}"
                                })
                                logger.info(f"❓ Suggesting fuzzy match: {best_match}")
                                return f"Meinten Sie {best_match}? Bitte bestätigen Sie mit 'Ja' oder nennen Sie Ihren Namen erneut."
                    
                    # 3. Versuche Nachnamen-Suche
                    if " " in customer_name:
                        last_name = customer_name.split()[-1]
                        logger.debug(f"Trying lastname search for: {last_name}")
                        matching_lastnames = [name for name in found_names if last_name.lower() in name.lower()]
                        
                        if len(matching_lastnames) == 1:
                            matched_name = matching_lastnames[0]
                            self.authenticated_customer = matched_name
                            self.failed_auth_attempts = 0
                            self.last_activity_time = datetime.now()
                            self.pending_fuzzy_match = None
                            self._log_access("authentication", "success", {
                                "customer": matched_name,
                                "match_type": "lastname"
                            })
                            logger.info(f"✅ Customer {matched_name} authenticated via lastname match")
                            return f"Vielen Dank für die Identifikation, {matched_name}. Wie kann ich Ihnen heute helfen?"
                        
                        elif len(matching_lastnames) > 1:
                            self._log_access("authentication", "multiple_matches", {
                                "lastname": last_name,
                                "matches": matching_lastnames
                            })
                            names_list = ", ".join(matching_lastnames[:3])
                            logger.info(f"⚠️ Multiple matches for lastname {last_name}")
                            return f"Ich habe mehrere Kunden mit dem Nachnamen {last_name} gefunden: {names_list}. Bitte nennen Sie Ihren vollständigen Namen."
                    
                    # 4. Keine Übereinstimmung gefunden
                    self.failed_auth_attempts += 1
                    self._log_access("authentication", "failed", {
                        "attempt": self.failed_auth_attempts,
                        "searched_for": customer_name
                    })
                    
                    remaining_attempts = self.max_failed_attempts - self.failed_auth_attempts
                    logger.warning(f"❌ Authentication failed. Remaining attempts: {remaining_attempts}")
                    
                    if remaining_attempts > 0:
                        if found_names and len(found_names) < 20:
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
                    logger.error(f"❌ Authentication API error: {response.status_code}")
                    return "Authentifizierungssystem momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}", exc_info=True)
            return "Fehler bei der Authentifizierung. Bitte versuchen Sie es erneut."
    
    @function_tool
    async def search_vehicle_data(self, query: str) -> str:
        """Sucht in der Fahrzeugdatenbank nach Informationen.
        
        MISTRAL WICHTIG: Der Parameter heißt 'query' (String).
        
        Verwende diese Funktion für:
        - Service-Informationen: query="nächster Service"
        - Fahrzeugdaten: query="meine Fahrzeuge"
        - Probleme: query="aktuelle Probleme"
        - Kennzeichen: query="LU 234567"
        
        Args:
            query: Die Suchanfrage als TEXT/STRING. NICHT vehicle_id oder andere Namen!
            
        Returns:
            Die gefundenen Fahrzeugdaten als Text
        """
        logger.info(f"🔍 Vehicle search requested: '{query}'")
        logger.debug(f"Parameter type check - query: {type(query)}")
        
        # Mistral-spezifische Validierung
        if not isinstance(query, str):
            logger.warning(f"Mistral sent non-string parameter: {type(query)}")
            query = str(query)
        
        # Security check - ensure customer is authenticated
        if not self._check_authentication():
            logger.warning("⚠️ Unauthorized vehicle search attempt")
            return "Bitte authentifizieren Sie sich zuerst mit Ihrem Namen."
        
        # Check session timeout
        if self._is_session_expired():
            logger.warning("⏰ Session expired")
            self._reset_session()
            return "Ihre Sitzung ist abgelaufen. Bitte authentifizieren Sie sich erneut."
        
        try:
            # Update last activity time
            self.last_activity_time = datetime.now()
            
            # SCHRITT 1: Kennzeichen des Kunden einmalig laden (mit Caching)
            if not hasattr(self, '_customer_kennzeichen'):
                logger.info(f"🔐 Loading vehicle data for {self.authenticated_customer}")
                
                async with httpx.AsyncClient() as client:
                    # Suche nach allen Fahrzeugen des Kunden
                    vehicle_response = await client.post(
                        f"{self.rag_url}/search",
                        json={
                            "query": f"besitzer: {self.authenticated_customer}",
                            "agent_type": "garage",
                            "top_k": 10,
                            "collection": "automotive_docs"
                        }
                    )
                    
                    self._customer_kennzeichen = []
                    if vehicle_response.status_code == 200:
                        vehicle_data = vehicle_response.json()
                        for result in vehicle_data.get("results", []):
                            content = result.get("content", "")
                            # Verbesserte Regex für verschiedene Kennzeichen-Formate
                            kz_patterns = [
                                r'[A-Z]{2}\s*\d+',  # Standard Format
                                r'Kennzeichen:\s*([A-Z]{2}\s*\d+)',  # Mit Label
                                r'\(([A-Z]{2}\s*\d+)\)',  # In Klammern
                            ]
                            
                            for pattern in kz_patterns:
                                matches = re.findall(pattern, content)
                                for match in matches:
                                    normalized = match.strip().upper()
                                    if normalized not in self._customer_kennzeichen:
                                        self._customer_kennzeichen.append(normalized)
                        
                        logger.info(f"✅ Found {len(self._customer_kennzeichen)} license plates for customer: {self._customer_kennzeichen}")
            
            # SCHRITT 2: Erweiterte Suchanfrage mit Kennzeichen
            # Füge alle Kundenkennzeichen zur Suche hinzu
            enhanced_query = query
            if hasattr(self, '_customer_kennzeichen') and self._customer_kennzeichen:
                # Füge Kennzeichen als zusätzliche Suchbegriffe hinzu
                kennzeichen_string = " ".join(self._customer_kennzeichen)
                enhanced_query = f"{query} {kennzeichen_string}"
            
            # Verarbeite die Query (z.B. Zahlen zu Ziffern)
            processed_query = self._process_license_plate(enhanced_query)
            
            logger.info(f"🔍 Enhanced search query: '{processed_query}'")
            logger.debug(f"Original query: '{query}'")
            
            # Log data access
            self._log_access("vehicle_search", "attempt", {"query": query})
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rag_url}/search",
                    json={
                        "query": processed_query,
                        "agent_type": "garage",
                        "top_k": 10,  # Erhöht für bessere Trefferquote
                        "collection": "automotive_docs"
                    }
                )
                
                logger.debug(f"RAG search response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    logger.debug(f"RAG returned {len(results)} results")
                    
                    # SCHRITT 3: Intelligenteres Filtern
                    filtered_results = []
                    kennzeichen_set = set(self._customer_kennzeichen) if hasattr(self, '_customer_kennzeichen') else set()
                    
                    for result in results:
                        content = result.get("content", "")
                        content_lower = content.lower()
                        
                        # Prüfe ob der Inhalt zum Kunden gehört
                        is_customer_data = False
                        
                        # 1. Check: Kundenname im Content
                        if self.authenticated_customer.lower() in content_lower:
                            is_customer_data = True
                            logger.debug(f"✓ Found customer name in content")
                        
                        # 2. Check: Eines der Kundenkennzeichen im Content
                        if not is_customer_data and kennzeichen_set:
                            for kennzeichen in kennzeichen_set:
                                if kennzeichen in content.upper():
                                    is_customer_data = True
                                    logger.debug(f"✓ Found customer license plate {kennzeichen} in content")
                                    break
                        
                        # 3. Check: Spezifische Muster für Service-Infos
                        if not is_customer_data:
                            # Prüfe ob es eine Service-Info mit einem der Kundenkennzeichen ist
                            service_patterns = [
                                r'Service für.*\(([A-Z]{2}\s*\d+)\)',
                                r'Nächster Service für.*\(([A-Z]{2}\s*\d+)\)',
                                r'Wartung.*([A-Z]{2}\s*\d+)',
                            ]
                            
                            for pattern in service_patterns:
                                match = re.search(pattern, content)
                                if match:
                                    found_kennzeichen = match.group(1).strip().upper()
                                    if found_kennzeichen in kennzeichen_set:
                                        is_customer_data = True
                                        logger.debug(f"✓ Found service info for customer vehicle {found_kennzeichen}")
                                        break
                        
                        if is_customer_data:
                            filtered_results.append(result)
                    
                    if filtered_results:
                        logger.info(f"✅ Found {len(filtered_results)} vehicle results for {self.authenticated_customer}")
                        self._log_access("vehicle_search", "success", {"results_count": len(filtered_results)})
                        
                        # Sortiere Ergebnisse nach Relevanz
                        # Priorisiere Ergebnisse mit höherem Score
                        filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                        
                        formatted = []
                        for i, result in enumerate(filtered_results[:5]):  # Maximal 5 Ergebnisse
                            content = result.get("content", "").strip()
                            score = result.get("score", 0)
                            if content:
                                # Füge Score zur Debugging hinzu (kann später entfernt werden)
                                formatted.append(f"[{i+1}] {content}")
                                logger.debug(f"Result {i+1} score: {score}")
                        
                        return "\n\n".join(formatted)
                    else:
                        # Falls keine Ergebnisse, versuche eine allgemeinere Suche
                        if "service" in query.lower() and hasattr(self, '_customer_kennzeichen') and self._customer_kennzeichen:
                            logger.info("ℹ️ No direct results, trying broader service search")
                            # Suche nur mit Kennzeichen
                            for kennzeichen in self._customer_kennzeichen[:1]:  # Versuche mit erstem Kennzeichen
                                retry_response = await client.post(
                                    f"{self.rag_url}/search",
                                    json={
                                        "query": f"Service {kennzeichen}",
                                        "agent_type": "garage",
                                        "top_k": 5,
                                        "collection": "automotive_docs"
                                    }
                                )
                                
                                if retry_response.status_code == 200:
                                    retry_data = retry_response.json()
                                    retry_results = retry_data.get("results", [])
                                    if retry_results:
                                        logger.info(f"✅ Found {len(retry_results)} results in retry")
                                        formatted = []
                                        for i, result in enumerate(retry_results[:3]):
                                            content = result.get("content", "").strip()
                                            if content and kennzeichen in content:
                                                formatted.append(f"[{i+1}] {content}")
                                        
                                        if formatted:
                                            return "\n\n".join(formatted)
                        
                        self._log_access("vehicle_search", "no_results", {"query": query})
                        logger.info("ℹ️ No results found for customer")
                        return "Keine Fahrzeugdaten zu dieser Anfrage für Ihr Konto gefunden."
                else:
                    logger.error(f"❌ RAG search failed: {response.status_code}")
                    self._log_access("vehicle_search", "error", {"status_code": response.status_code})
                    return "Fehler beim Zugriff auf die Fahrzeugdatenbank."
                    
        except Exception as e:
            logger.error(f"❌ Error searching RAG: {e}", exc_info=True)
            self._log_access("vehicle_search", "exception", {"error": str(e)})
            return "Die Fahrzeugdatenbank ist momentan nicht erreichbar."
    
    def _check_authentication(self) -> bool:
        """Prüft ob ein Kunde authentifiziert ist"""
        if not self.authenticated_customer:
            logger.warning(f"🚫 Unauthenticated access attempt in session {self.session_id}")
            self._log_access("unauthorized_access", "blocked", {})
            return False
        return True
    
    def _is_session_expired(self) -> bool:
        """Prüft ob die Session abgelaufen ist"""
        if not self.last_activity_time:
            return True
        
        time_since_activity = datetime.now() - self.last_activity_time
        is_expired = time_since_activity > timedelta(minutes=self.session_timeout_minutes)
        
        if is_expired:
            logger.debug(f"Session expired. Time since last activity: {time_since_activity}")
        
        return is_expired
    
    def _reset_session(self):
        """Setzt die Session zurück"""
        logger.info(f"🔄 Resetting session {self.session_id}")
        self.authenticated_customer = None
        self.failed_auth_attempts = 0
        self.last_activity_time = None
        self.pending_fuzzy_match = None
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
        logger.info(f"📝 Access log: {action} - {status}")
        logger.debug(f"Access log details: {log_entry}")
    
    def _process_license_plate(self, text: str) -> str:
        """Korrigiert Sprache-zu-Text Fehler bei Kennzeichen"""
        original_text = text
        
        # Deutsche Zahlwörter zu Ziffern
        number_map = {
            'null': '0', 'eins': '1', 'zwei': '2', 'drei': '3', 'vier': '4', 
            'fünf': '5', 'sechs': '6', 'sieben': '7', 'acht': '8', 'neun': '9'
        }
        
        # Ersetze Zahlwörter durch Ziffern
        for word, digit in number_map.items():
            text = text.replace(f" {word} ", f" {digit} ")
            text = text.replace(f" {word}", f" {digit}")
        
        # Normalisiere Kennzeichen Format
        pattern = r'([a-zA-Z]{2})\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            canton = match.group(1).upper()
            numbers = ''.join([match.group(i) for i in range(2, 7)])
            corrected = f"{canton} {numbers}"
            text = re.sub(pattern, corrected, text, flags=re.IGNORECASE)
            logger.info(f"🚗 Corrected license plate: '{original_text}' -> '{corrected}'")
        
        return text


async def entrypoint(ctx: JobContext):
    """Entry point for garage agent - MISTRAL VERSION"""
    logger.info("="*50)
    logger.info("🚀 Secure Garage Agent Starting (1.0.x) - MISTRAL v0.3")
    logger.info("="*50)
    
    # Log configuration
    logger.debug(f"Room name: {ctx.room.name}")
    logger.debug(f"Room ID: {ctx.room.sid}")
    logger.debug(f"Using Mistral v0.3 model")
    
    # SCHRITT 1: ZUERST connecten (OHNE AutoSubscribe!)
    logger.info("📡 Connecting to room...")
    await ctx.connect()  # Keine Parameter - nutzt die Defaults
    logger.info("✅ Connected to room")
    
    # SCHRITT 2: Auf Participant warten
    logger.info("👤 Waiting for participant...")
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ Participant joined: {participant.identity}")
    
    # Log participant details
    logger.debug(f"Participant SID: {participant.sid}")
    logger.debug(f"Participant name: {participant.name}")
    logger.debug(f"Number of tracks: {len(participant.track_publications)}")
    
    # SCHRITT 3: DANN Agent und Session erstellen
    logger.info("🎯 Creating agent and session with Mistral...")
    agent = GarageAgent()
    session = AgentSession()
    
    # SCHRITT 4: ZULETZT Session starten
    logger.info("🏁 Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=agent
    )
    
    logger.info("✅ Secure Garage Agent with Mistral v0.3 ready and listening!")
    logger.info("="*50)


if __name__ == "__main__":
    logger.info("🚀 Starting Garage Agent Worker with Mistral v0.3...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
