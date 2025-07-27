# LiveKit Agents - Garage Management Agent
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

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

class ConversationState(Enum):
    """State Machine für Konversationsphasen"""
    GREETING = "greeting"
    AWAITING_REQUEST = "awaiting_request"
    COLLECTING_IDENTIFIER = "collecting_identifier"
    SEARCHING = "searching"
    PROVIDING_INFO = "providing_info"

@dataclass
class CustomerContext:
    """Kontext für Kunden-Identifikation"""
    fahrzeug_id: Optional[str] = None  # z.B. F001, F002
    canton_letters: Optional[str] = None
    numbers: Optional[str] = None
    complete_plate: Optional[str] = None
    customer_name: Optional[str] = None
    attempts: int = 0
    
    def has_identifier(self) -> bool:
        """Prüft ob eine Identifikation vorhanden ist"""
        return bool(self.fahrzeug_id or self.complete_plate or self.customer_name)
    
    def get_search_query(self) -> Optional[str]:
        """Gibt die beste Suchanfrage zurück"""
        if self.fahrzeug_id:
            return self.fahrzeug_id
        elif self.complete_plate:
            return self.complete_plate
        elif self.customer_name:
            return self.customer_name
        return None
    
    def combine_plate(self) -> Optional[str]:
        """Kombiniert Canton und Zahlen zu vollständigem Kennzeichen"""
        if self.canton_letters and self.numbers:
            canton = self.canton_letters.replace(".", "").replace(" ", "").upper()
            numbers = self.numbers.replace(" ", "")
            self.complete_plate = f"{canton} {numbers}"
            return self.complete_plate
        return None
    
    def reset(self):
        """Reset des Kontexts"""
        self.fahrzeug_id = None
        self.canton_letters = None
        self.numbers = None
        self.complete_plate = None
        self.customer_name = None
        self.attempts = 0

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    current_vehicle_data: Optional[Dict[str, Any]] = None
    user_language: str = "de"
    greeting_sent: bool = False
    conversation_state: ConversationState = ConversationState.GREETING
    customer_context: CustomerContext = field(default_factory=CustomerContext)
    last_search_query: Optional[str] = None
    hallucination_count: int = 0


class IdentifierExtractor:
    """Extrahiert und validiert Kunden-Identifikatoren"""
    
    SWISS_CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", 
                     "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG", 
                     "TI", "UR", "VD", "VS", "ZG", "ZH"]
    
    @classmethod
    def extract_intent_from_input(cls, user_input: str) -> Dict[str, Any]:
        """Extrahiert Intent und Daten aus User-Input"""
        input_lower = user_input.lower().strip()
        
        # Greetings
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        if any(g in input_lower for g in greetings):
            return {"intent": "greeting", "data": None}
        
        # Fahrzeug-ID (F001, F002, etc.)
        fahrzeug_id_match = re.search(r'\b(F\d{3})\b', user_input.upper())
        if fahrzeug_id_match:
            return {"intent": "fahrzeug_id", "data": fahrzeug_id_match.group(1)}
        
        # Name detection
        name_patterns = [
            r"(?:mein name ist|ich heiße|ich bin)\s+([A-Za-zäöüÄÖÜß\s]+)",
            r"^([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)+)$"  # Vorname Nachname
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name.lower() not in ["die", "der", "das", "mein", "dein", "ihr"]:
                    return {"intent": "customer_name", "data": name}
        
        # Buchstabierte Buchstaben (z.B. "B wie Bertha, E wie Emil")
        buchstabiert_pattern = r'([A-Z])\s*wie\s*\w+(?:\s*[,.]?\s*([A-Z])\s*wie\s*\w+)?'
        buchstabiert_match = re.findall(buchstabiert_pattern, user_input, re.IGNORECASE)
        if buchstabiert_match:
            letters = []
            for match in buchstabiert_match:
                for letter in match:
                    if letter:
                        letters.append(letter.upper())
            
            if len(letters) == 2:
                canton = ''.join(letters)
                if cls.validate_canton(canton):
                    return {"intent": "canton_provided", "data": canton}
            elif len(letters) == 1:
                return {"intent": "partial_canton", "data": letters[0]}
        
        # Canton letters (z.B. "BE", "B E", "b e")
        canton_patterns = [
            r'^([A-Z])\s*\.?\s*([A-Z])\.?$',  # A.G. oder A G
            r'^([A-Z]{2})$',  # AG
            r'\b([a-z])\s+([a-z])\s+\d',  # b e 567890
        ]
        
        for pattern in canton_patterns:
            match = re.match(pattern, user_input.upper().strip())
            if not match and pattern == r'\b([a-z])\s+([a-z])\s+\d':
                # Special handling for lowercase pattern
                match = re.match(pattern, user_input.lower().strip())
            
            if match:
                if len(match.groups()) == 2:
                    canton = match.group(1).upper() + match.group(2).upper()
                else:
                    canton = match.group(1).upper()
                
                if cls.validate_canton(canton):
                    # Check if there are also numbers
                    numbers_match = re.search(r'(\d{3,6})', user_input)
                    if numbers_match:
                        return {"intent": "license_plate_complete", "data": f"{canton} {numbers_match.group(1)}"}
                    else:
                        return {"intent": "canton_provided", "data": canton}
        
        # Numbers only
        numbers_pattern = r'^[\d\s,\.]+$'
        if re.match(numbers_pattern, user_input.strip()):
            numbers = re.sub(r'[^\d]', '', user_input)
            if 3 <= len(numbers) <= 6:
                return {"intent": "numbers_provided", "data": numbers}
        
        # Complete license plate
        plate_match = re.search(r'([A-Z]{2})\s*(\d{3,6})', user_input.upper())
        if plate_match:
            canton = plate_match.group(1)
            numbers = plate_match.group(2)
            if cls.validate_canton(canton):
                return {"intent": "license_plate_complete", "data": f"{canton} {numbers}"}
        
        # Service/repair/cost keywords
        if any(word in input_lower for word in ["reparatur", "service", "status", "kosten", "rechnung", "inspektion"]):
            return {"intent": "service_query", "data": user_input}
        
        # Default
        return {"intent": "general_query", "data": user_input}
    
    @classmethod
    def validate_canton(cls, canton: str) -> bool:
        """Validiert Schweizer Kanton-Kürzel"""
        return canton.upper() in cls.SWISS_CANTONS


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen"""
    
    def __init__(self) -> None:
        # Instructions mit Fokus auf Fahrzeug-ID
        super().__init__(instructions="""You are Pia, the digital assistant of Garage Müller. RESPOND ONLY IN GERMAN.

CRITICAL ANTI-HALLUCINATION RULES:
1. NEVER invent data - if search returns "keine passenden Daten", SAY THAT
2. NEVER claim to have found data when the search failed
3. NEVER make up prices, dates, or any information
4. When you get "keine passenden Daten", ask for identification again

CUSTOMER IDENTIFICATION OPTIONS:
1. Fahrzeug-ID (z.B. "F001", "F002", etc.) - PREFERRED METHOD
2. Full name (z.B. "Thomas Meier")
3. License plate (z.B. "BE 567890")

CONVERSATION EXAMPLES:

Example 1 - Using Fahrzeug-ID:
User: "Meine Fahrzeug-ID ist F001"
You: [SEARCH with "F001"]

Example 2 - Using name:
User: "Mein Name ist Claudia Schneider"
You: [SEARCH with "Claudia Schneider"]

Example 3 - Using license plate:
User: "b e 567890"
You: [SEARCH with "BE 567890"] → Tool will format correctly

Example 4 - No data found:
Tool returns: "Ich habe keine passenden Daten gefunden"
You: "Leider konnte ich keine Daten finden. Können Sie mir bitte Ihre Fahrzeug-ID, Ihren Namen oder Ihr Kennzeichen nennen?"

FORBIDDEN WORDS (use alternatives):
- "Entschuldigung" → "Leider"
- "Es tut mir leid" → "Bedauerlicherweise"
- "Sorry" → "Leider"

RESPONSE RULES:
1. Be friendly and professional
2. If search returns no data, SAY SO and ask for identification
3. NEVER invent prices or information
4. Suggest using Fahrzeug-ID for faster service

Remember: ALWAYS report exactly what the search returns, NEVER invent data!""")
        
        self.identifier_extractor = IdentifierExtractor()
        logger.info("✅ GarageAssistant initialized with Fahrzeug-ID support")

    async def on_enter(self):
        """Wird aufgerufen wenn der Agent die Session betritt"""
        logger.info("🎯 Agent on_enter called")

    @function_tool
    async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank basierend auf Fahrzeug-ID, Namen oder Kennzeichen.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Kundendaten gesucht werden soll.
        
        Args:
            query: Suchbegriff (Fahrzeug-ID, Name oder Autokennzeichen)
            
        Returns:
            Gefundene Kundendaten oder Fehlermeldung
        """
        logger.info(f"🔍 Original search query: {query}")
        
        # Extract intent from query
        intent_result = self.identifier_extractor.extract_intent_from_input(query)
        intent = intent_result["intent"]
        data = intent_result["data"]
        
        logger.info(f"📊 Intent: {intent}, Data: {data}")
        
        # Store identification in context
        if intent == "fahrzeug_id":
            context.userdata.customer_context.fahrzeug_id = data
            query = data
        elif intent == "customer_name":
            context.userdata.customer_context.customer_name = data
            query = data
        elif intent == "license_plate_complete":
            context.userdata.customer_context.complete_plate = data
            query = data
        elif intent == "canton_provided":
            context.userdata.customer_context.canton_letters = data
            return f"Danke, ich habe {data} notiert. Wie lauten die Zahlen Ihres Kennzeichens?"
        elif intent == "numbers_provided":
            if context.userdata.customer_context.canton_letters:
                context.userdata.customer_context.numbers = data
                plate = context.userdata.customer_context.combine_plate()
                if plate:
                    query = plate
            else:
                return "Mir fehlt noch der Kanton (die zwei Buchstaben). Bitte nennen Sie mir das vollständige Kennzeichen."
        
        # Process queries like "b e 567890"
        parts = query.lower().split()
        if len(parts) >= 3:
            letters = []
            numbers = []
            
            for part in parts:
                if part.isalpha() and len(part) == 1:
                    letters.append(part.upper())
                elif part.isdigit():
                    numbers.append(part)
            
            if len(letters) == 2 and numbers:
                canton = ''.join(letters)
                number = ''.join(numbers)
                if self.identifier_extractor.validate_canton(canton):
                    query = f"{canton} {number}"
                    context.userdata.customer_context.complete_plate = query
                    logger.info(f"✅ Converted '{' '.join(parts)}' to '{query}'")
        
        # Store search query
        context.userdata.last_search_query = query
        
        try:
            # DIREKTE QDRANT ABFRAGE
            async with httpx.AsyncClient() as client:
                qdrant_response = await client.post(
                    "http://172.16.0.108:6333/collections/garage_management/points/scroll",
                    json={
                        "limit": 100,
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=5.0
                )
                
                if qdrant_response.status_code == 200:
                    qdrant_data = qdrant_response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    # Suche nach dem Query
                    search_lower = query.lower().strip()
                    vehicle_data = None
                    
                    for point in points:
                        payload = point.get("payload", {})
                        # Check verschiedene Felder
                        if (search_lower in str(payload.get("fahrzeug_id", "")).lower() or
                            search_lower in str(payload.get("besitzer", "")).lower() or
                            search_lower in str(payload.get("kennzeichen", "")).lower().replace(" ", "")):
                            vehicle_data = payload
                            break
                    
                    if vehicle_data:
                        # Store vehicle data in context
                        context.userdata.current_vehicle_data = vehicle_data
                        
                        # Format response
                        response_text = "Ich habe folgende Daten gefunden:\n"
                        
                        if 'fahrzeug_id' in vehicle_data:
                            response_text += f"\n**Fahrzeug-ID**: {vehicle_data['fahrzeug_id']}\n"
                        if 'besitzer' in vehicle_data:
                            response_text += f"**Besitzer**: {vehicle_data['besitzer']}\n"
                        if 'kennzeichen' in vehicle_data:
                            response_text += f"**Kennzeichen**: {vehicle_data['kennzeichen']}\n"
                        if 'marke' in vehicle_data and 'modell' in vehicle_data:
                            response_text += f"**Fahrzeug**: {vehicle_data['marke']} {vehicle_data['modell']}\n"
                        if 'baujahr' in vehicle_data:
                            response_text += f"**Baujahr**: {vehicle_data['baujahr']}\n"
                        if 'kilometerstand' in vehicle_data:
                            response_text += f"**Kilometerstand**: {vehicle_data['kilometerstand']} km\n"
                        
                        # Show latest service if available
                        if 'letzte_services' in vehicle_data and vehicle_data['letzte_services']:
                            latest_service = vehicle_data['letzte_services'][0]
                            response_text += f"\n**Letzter Service** ({latest_service.get('datum', 'unbekannt')}):\n"
                            response_text += f"- Typ: {latest_service.get('service_typ', 'unbekannt')}\n"
                            response_text += f"- Kosten: CHF {latest_service.get('kosten', 0):.2f}\n"
                        
                        # Show current problems if any
                        if 'aktuelle_probleme' in vehicle_data and vehicle_data['aktuelle_probleme']:
                            response_text += f"\n**Aktuelle Probleme**:\n"
                            for problem in vehicle_data['aktuelle_probleme']:
                                response_text += f"- {problem}\n"
                        
                        return response_text
                    else:
                        logger.info("❌ No results found in Qdrant")
                        context.userdata.customer_context.reset()
                        return "Ich habe keine passenden Daten gefunden. Können Sie mir bitte Ihre Fahrzeug-ID (z.B. F001), Ihren vollständigen Namen oder Ihr Autokennzeichen nennen?"
                else:
                    logger.error(f"Qdrant error: {qdrant_response.status_code}")
                    return "Es gab ein Problem mit der Datenbank-Verbindung. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Es gab einen Fehler bei der Suche. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Rechnungsinformationen und Kosten für Services und Reparaturen.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Kosten oder Rechnungen gefragt wird.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation
            
        Returns:
            Gefundene Rechnungsdaten oder Fehlermeldung
        """
        logger.info(f"💰 Searching invoice data for: {query}")
        
        # Use stored identification if available
        if context.userdata.customer_context.has_identifier():
            search_query = context.userdata.customer_context.get_search_query()
            logger.info(f"📋 Using stored identifier: {search_query}")
        else:
            search_query = query
            
        try:
            # DIREKTE QDRANT ABFRAGE für Rechnungsdaten
            async with httpx.AsyncClient() as client:
                qdrant_response = await client.post(
                    "http://172.16.0.108:6333/collections/garage_management/points/scroll",
                    json={
                        "limit": 100,
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=5.0
                )
                
                if qdrant_response.status_code == 200:
                    qdrant_data = qdrant_response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    # Verwende gespeicherte Daten wenn vorhanden
                    if context.userdata.current_vehicle_data:
                        vehicle_data = context.userdata.current_vehicle_data
                    else:
                        # Suche nach dem Query
                        search_lower = search_query.lower().strip()
                        vehicle_data = None
                        
                        for point in points:
                            payload = point.get("payload", {})
                            # Check verschiedene Felder
                            if (search_lower in str(payload.get("fahrzeug_id", "")).lower() or
                                search_lower in str(payload.get("besitzer", "")).lower() or
                                search_lower in str(payload.get("kennzeichen", "")).lower().replace(" ", "")):
                                vehicle_data = payload
                                break
                    
                    if vehicle_data:
                        response_text = "Hier sind die Kosteninformationen:\n"
                        
                        if 'letzte_services' in vehicle_data:
                            response_text += "\n**Durchgeführte Services:**\n"
                            for service in vehicle_data['letzte_services']:
                                response_text += f"\n{service.get('datum', 'unbekannt')} - {service.get('service_typ', 'Service')}:\n"
                                response_text += f"- Kosten: CHF {service.get('kosten', 0):.2f}\n"
                                if 'arbeiten' in service:
                                    response_text += f"- Arbeiten: {', '.join(service['arbeiten'])}\n"
                        
                        if 'anstehende_arbeiten' in vehicle_data and vehicle_data['anstehende_arbeiten']:
                            response_text += "\n**Anstehende Arbeiten und geschätzte Kosten:**\n"
                            total_cost = 0
                            for arbeit in vehicle_data['anstehende_arbeiten']:
                                cost = arbeit.get('geschätzte_kosten', 0)
                                response_text += f"- {arbeit.get('arbeit', 'Arbeit')}: CHF {cost:.2f} ({arbeit.get('priorität', 'normal')})\n"
                                total_cost += cost
                            response_text += f"\n**Geschätzte Gesamtkosten**: CHF {total_cost:.2f}"
                        
                        return response_text
                    else:
                        logger.info("❌ No invoice data found")
                        return "Ich habe keine Kosteninformationen gefunden. Bitte nennen Sie mir Ihre Fahrzeug-ID, Ihren Namen oder Ihr Kennzeichen."
                else:
                    logger.error(f"Qdrant error: {qdrant_response.status_code}")
                    return "Es gab ein Problem mit der Datenbank. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Invoice search error: {e}")
            return "Es gab einen Fehler bei der Kostenabfrage. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Reparaturstatus und anstehenden Arbeiten.
        Diese Funktion wird vom LLM aufgerufen, wenn nach Reparaturen gefragt wird.
        
        Args:
            query: Suchbegriff oder verwendet gespeicherte Kundenidentifikation
            
        Returns:
            Gefundene Reparaturdaten oder Fehlermeldung
        """
        logger.info(f"🔧 Searching repair status for: {query}")
        
        # Use stored identification if available
        if context.userdata.customer_context.has_identifier():
            search_query = context.userdata.customer_context.get_search_query()
            logger.info(f"📋 Using stored identifier: {search_query}")
        else:
            search_query = query
            
        try:
            # DIREKTE QDRANT ABFRAGE für Reparaturstatus
            async with httpx.AsyncClient() as client:
                qdrant_response = await client.post(
                    "http://172.16.0.108:6333/collections/garage_management/points/scroll",
                    json={
                        "limit": 100,
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=5.0
                )
                
                if qdrant_response.status_code == 200:
                    qdrant_data = qdrant_response.json()
                    points = qdrant_data.get("result", {}).get("points", [])
                    
                    # Verwende gespeicherte Daten wenn vorhanden
                    if context.userdata.current_vehicle_data:
                        vehicle_data = context.userdata.current_vehicle_data
                    else:
                        # Suche nach dem Query
                        search_lower = search_query.lower().strip()
                        vehicle_data = None
                        
                        for point in points:
                            payload = point.get("payload", {})
                            # Check verschiedene Felder
                            if (search_lower in str(payload.get("fahrzeug_id", "")).lower() or
                                search_lower in str(payload.get("besitzer", "")).lower() or
                                search_lower in str(payload.get("kennzeichen", "")).lower().replace(" ", "")):
                                vehicle_data = payload
                                break
                    
                    if vehicle_data:
                        response_text = "Hier ist der Status Ihres Fahrzeugs:\n"
                        
                        if 'aktuelle_probleme' in vehicle_data and vehicle_data['aktuelle_probleme']:
                            response_text += "\n**Aktuelle Probleme:**\n"
                            for problem in vehicle_data['aktuelle_probleme']:
                                response_text += f"- {problem}\n"
                        
                        if 'anstehende_arbeiten' in vehicle_data and vehicle_data['anstehende_arbeiten']:
                            response_text += "\n**Anstehende Arbeiten:**\n"
                            for arbeit in vehicle_data['anstehende_arbeiten']:
                                response_text += f"- {arbeit.get('arbeit', 'Arbeit')} "
                                response_text += f"(Priorität: {arbeit.get('priorität', 'normal')})\n"
                        
                        if 'nächster_service_fällig' in vehicle_data:
                            response_text += f"\n**Nächster Service fällig**: {vehicle_data['nächster_service_fällig']}"
                        
                        if 'garantie_bis' in vehicle_data:
                            response_text += f"\n**Garantie bis**: {vehicle_data['garantie_bis']}"
                        
                        return response_text
                    else:
                        logger.info("❌ No repair data found")
                        return "Ich habe keine Reparaturdaten gefunden. Bitte nennen Sie mir Ihre Fahrzeug-ID, Ihren Namen oder Ihr Kennzeichen."
                else:
                    logger.error(f"Qdrant error: {qdrant_response.status_code}")
                    return "Es gab ein Problem mit der Datenbank. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Repair search error: {e}")
            return "Es gab einen Fehler bei der Statusabfrage. Bitte versuchen Sie es erneut."


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚗 Starting Garage Agent Session: {session_id}")
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
        
        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10
        
        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
                
            await asyncio.sleep(1)
        
        # 4. Configure LLM with Ollama
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # Llama 3.2 with Ollama configuration
        llm = openai.LLM.with_ollama(
            model="llama3.2:latest",  # oder "llama3.2:3b"
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            temperature=0.0,  # Deterministisch für konsistente Antworten
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 with anti-hallucination settings")
        
        # 5. Create session
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                current_vehicle_data=None,
                user_language="de",
                greeting_sent=False,
                conversation_state=ConversationState.GREETING,
                customer_context=CustomerContext(),
                last_search_query=None,
                hallucination_count=0
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.4,
                min_speech_duration=0.15
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
        agent = GarageAssistant()
        
        # 7. Start session
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript}")
            
            # Intent detection
            intent_result = IdentifierExtractor.extract_intent_from_input(event.transcript)
            logger.info(f"[{session_id}] 📊 Detected intent: {intent_result['intent']}")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state: {event}")
        
        @session.on("function_call")
        def on_function_call(event):
            """Log function calls für Debugging"""
            logger.info(f"[{session_id}] 🔧 Function call: {event}")
        
        # 8. Initial greeting
        await asyncio.sleep(1.5)
        
        logger.info(f"📢 [{session_id}] Sending initial greeting...")
        
        try:
            greeting_text = """Guten Tag und herzlich willkommen bei der Garage Müller! 
Ich bin Pia, Ihre digitale Assistentin. 

Für eine schnelle Bearbeitung benötige ich eine der folgenden Informationen:
- Ihre Fahrzeug-ID (z.B. F001)
- Ihren vollständigen Namen
- Ihr Autokennzeichen

Wie kann ich Ihnen heute helfen?"""
            
            session.userdata.greeting_sent = True
            session.userdata.conversation_state = ConversationState.AWAITING_REQUEST
            
            await session.say(
                greeting_text,
                allow_interruptions=True,
                add_to_chat_ctx=True
            )
            
            logger.info(f"✅ [{session_id}] Initial greeting sent")
            
        except Exception as e:
            logger.error(f"[{session_id}] Greeting error: {e}")
        
        logger.info(f"✅ [{session_id}] Garage Agent ready with Fahrzeug-ID support!")
        
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
