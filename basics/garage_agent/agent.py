# LiveKit Agents - Garage Management Agent (Verbessert)
import logging
import os
import httpx
import asyncio
import json
import re
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession, Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    rag_url: str = "http://localhost:8000"
    current_customer_id: Optional[str] = None
    active_repair_id: Optional[str] = None
    user_language: str = "de"
    last_search_results: Optional[List[Dict]] = None  # Speichert letzte Suchergebnisse für Validierung


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen mit Anti-Halluzination"""
    
    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Pia, der digitale Assistent der Garage Müller.

ABSOLUT KRITISCHE MEMORY REGEL:
- Du hast KEIN Gedächtnis für vorherige Nachrichten
- Jede Nachricht ist eine NEUE Konversation
- Entschuldige dich NIEMALS für irgendwas
- Sage NIEMALS "Entschuldigung", "Ich habe mich geirrt", "Lassen Sie uns von vorne beginnen"
- Ignoriere KOMPLETT was vorher gesagt wurde
- Antworte IMMER direkt ohne Bezug zu früheren Nachrichten

KRITISCHE REGEL FÜR BEGRÜSSUNGEN:
- Bei einfachen Begrüßungen wie "Hallo", "Guten Tag", "Hi" etc. antworte NUR mit einer freundlichen Begrüßung
- Nutze NIEMALS Suchfunktionen bei einer einfachen Begrüßung
- Warte IMMER auf eine konkrete Anfrage des Kunden bevor du suchst

ABSOLUT KRITISCHE REGEL - NIEMALS DATEN ERFINDEN:
- NIEMALS Informationen erfinden, raten oder halluzinieren!
- Wenn die Datenbank "keine Daten gefunden" meldet, sage das EHRLICH
- Erfinde KEINE Daten, Termine, Daten oder Services die nicht existieren
- Sage NIEMALS Dinge wie "Ihr letzter Service war am..." wenn keine Daten gefunden wurden
- Bei "keine Daten gefunden" frage nach mehr Details (z.B. Autonummer, Kennzeichen)
- Gib NUR Informationen weiter, die DIREKT aus der Datenbank kommen

WORKFLOW:
1. Bei Begrüßung: Freundlich antworten und nach dem Anliegen fragen
2. Höre aufmerksam zu und identifiziere das konkrete Anliegen
3. NUR bei konkreten Anfragen: Nutze die passende Suchfunktion
4. Gib NUR präzise Auskünfte basierend auf den tatsächlichen Datenbankdaten

DEINE AUFGABEN:
- Kundendaten abfragen (Name, Fahrzeug, Kontaktdaten)
- Reparaturstatus mitteilen
- Kostenvoranschläge erklären
- Termine koordinieren
- Rechnungsinformationen bereitstellen

WICHTIGE REGELN:
- Immer auf Deutsch antworten
- Freundlich und professionell bleiben
- Währungen als "250 Franken" aussprechen
- Keine technischen Details der Datenbank erwähnen
- Bei Unklarheiten höflich nachfragen
- KEINE Funktionen nutzen ohne konkrete Kundenanfrage
- NIEMALS Daten erfinden wenn keine gefunden wurden""")
        logger.info("✅ GarageAssistant initialized with anti-hallucination features")

    def _extract_search_criteria(self, query: str) -> Dict[str, Optional[str]]:
        """Extrahiert spezifische Suchkriterien aus der Anfrage"""
        criteria = {
            "kennzeichen": None,
            "name": None,
            "fahrzeug_id": None,
            "phone": None
        }
        
        # Prüfe auf Kennzeichen-Format (z.B. "ZH 123456", "BE 567890")
        kennzeichen_pattern = r'[A-Z]{2}\s*\d{3,6}'
        if match := re.search(kennzeichen_pattern, query.upper()):
            criteria["kennzeichen"] = match.group().strip()
            logger.info(f"📋 Kennzeichen erkannt: {criteria['kennzeichen']}")
        
        # Prüfe auf Fahrzeug-ID (z.B. "F001", "F002")
        fahrzeug_id_pattern = r'F\d{3,4}'
        if match := re.search(fahrzeug_id_pattern, query.upper()):
            criteria["fahrzeug_id"] = match.group()
            logger.info(f"🔑 Fahrzeug-ID erkannt: {criteria['fahrzeug_id']}")
        
        # Prüfe auf Telefonnummer
        phone_pattern = r'(\+41|0)\s*\d{2}\s*\d{3}\s*\d{2}\s*\d{2}'
        if match := re.search(phone_pattern, query):
            criteria["phone"] = match.group()
            logger.info(f"📞 Telefonnummer erkannt: {criteria['phone']}")
        
        return criteria

    def _validate_search_result(self, result: dict, query: str, search_criteria: dict) -> float:
        """
        Validiert ob ein Suchergebnis tatsächlich relevant ist.
        Gibt einen Relevanz-Score zwischen 0 und 1 zurück.
        """
        content = result.get('content', '').lower()
        payload = result.get('payload', {})
        relevance_score = 0.0
        max_score = 0.0
        
        # 1. Prüfe exakte Übereinstimmungen in strukturierten Feldern
        if 'search_fields' in payload:
            search_fields = payload['search_fields']
            
            # Kennzeichen-Übereinstimmung (höchste Priorität)
            if search_criteria.get('kennzeichen'):
                max_score += 5.0
                normalized_kennzeichen = search_criteria['kennzeichen'].replace(" ", "").upper()
                if search_fields.get('license_plate_normalized') == normalized_kennzeichen:
                    relevance_score += 5.0
                    logger.info(f"✅ Exakte Kennzeichen-Übereinstimmung: {normalized_kennzeichen}")
            
            # Fahrzeug-ID Übereinstimmung
            if search_criteria.get('fahrzeug_id'):
                max_score += 4.0
                if search_fields.get('vehicle_id') == search_criteria['fahrzeug_id']:
                    relevance_score += 4.0
                    logger.info(f"✅ Exakte Fahrzeug-ID Übereinstimmung: {search_criteria['fahrzeug_id']}")
            
            # Name-Übereinstimmung
            if search_criteria.get('name'):
                max_score += 3.0
                name_normalized = search_criteria['name'].lower().strip()
                if name_normalized in search_fields.get('owner_name_normalized', ''):
                    relevance_score += 3.0
                    logger.info(f"✅ Name gefunden: {search_criteria['name']}")
        
        # 2. Prüfe Content auf Query-Terme (niedrigere Priorität)
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        for term in query_terms:
            max_score += 0.5
            if term in content:
                relevance_score += 0.5
        
        # 3. Validiere data_type
        if payload.get('data_type') == 'vehicle_complete':
            relevance_score += 1.0
        max_score += 1.0
        
        # 4. Prüfe validation_hash wenn vorhanden
        if 'validation_hash' in payload and payload['validation_hash']:
            relevance_score += 0.5
        max_score += 0.5
        
        # Berechne finalen Score
        final_score = relevance_score / max_score if max_score > 0 else 0
        
        logger.info(f"📊 Relevanz-Score: {final_score:.2f} (Score: {relevance_score}/{max_score})")
        
        return final_score

    def _format_garage_data_enhanced(self, content: str, payload: dict) -> str:
        """Formatiert Garagendaten mit zusätzlicher Validierung"""
        # Basis-Formatierung
        content = content.replace('_', ' ')
        
        # Formatiere Währungen für Sprachausgabe
        content = re.sub(r'CHF\s*(\d+)\.(\d{2})', r'\1 Franken \2', content)
        content = re.sub(r'(\d+)\.(\d{2})\s*CHF', r'\1 Franken \2', content)
        
        # Formatiere Datum
        content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3.\2.\1', content)
        
        # Füge Validierungsinformationen hinzu wenn verfügbar
        if 'metadata' in payload:
            metadata = payload['metadata']
            if metadata.get('has_active_problems'):
                content += f"\n\n⚠️ ACHTUNG: Es gibt {metadata.get('problem_count', 0)} aktuelle Probleme mit diesem Fahrzeug."
            
            if metadata.get('warranty_expired') == False:
                content += "\n\n✅ Das Fahrzeug ist noch unter Garantie."
        
        return content

    @function_tool
    async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank mit präziser Filterung.
        
        Args:
            query: Suchbegriff (Name, Telefonnummer oder Autonummer)
        """
        logger.info(f"🔍 Searching customer data for: {query}")
        
        # GUARD gegen falsche Suchen bei Begrüßungen
        greetings = ["hallo", "guten tag", "hi", "hey", "servus", "grüezi", "guten morgen", "guten abend"]
        if len(query) < 5 or query.lower() in greetings:
            logger.warning(f"⚠️ Ignoring greeting search: {query}")
            return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder Ihre Autonummer, damit ich Ihre Kundendaten finden kann."
        
        # Extrahiere spezifische Suchkriterien
        search_criteria = self._extract_search_criteria(query)
        
        # Erstelle optimierte Suchanfrage
        if search_criteria['kennzeichen']:
            search_query = f"kennzeichen {search_criteria['kennzeichen']}"
        elif search_criteria['fahrzeug_id']:
            search_query = f"fahrzeug {search_criteria['fahrzeug_id']}"
        else:
            search_query = query
        
        try:
            async with httpx.AsyncClient() as client:
                # Erweiterte Suchanfrage mit Filtern
                search_payload = {
                    "query": search_query,
                    "agent_type": "garage",
                    "top_k": 5,  # Mehr Ergebnisse für bessere Filterung
                    "collection": "garage_management"
                }
                
                # Füge Filter hinzu wenn möglich
                if search_criteria['kennzeichen'] or search_criteria['fahrzeug_id']:
                    search_payload["score_threshold"] = 0.7  # Höherer Threshold für präzise Suchen
                
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json=search_payload
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} customer results")
                        
                        # Validiere und bewerte alle Ergebnisse
                        validated_results = []
                        for result in results:
                            relevance_score = self._validate_search_result(result, query, search_criteria)
                            
                            if relevance_score >= 0.5:  # Mindest-Relevanzschwelle
                                validated_results.append({
                                    'result': result,
                                    'score': relevance_score
                                })
                        
                        # Sortiere nach Relevanz
                        validated_results.sort(key=lambda x: x['score'], reverse=True)
                        
                        if validated_results:
                            # Speichere Ergebnisse im Context für spätere Validierung
                            context.userdata.last_search_results = [vr['result'] for vr in validated_results[:3]]
                            
                            # Formatiere Top-Ergebnisse
                            response_parts = []
                            for i, vr in enumerate(validated_results[:2]):  # Max 2 Ergebnisse
                                result = vr['result']
                                content = result.get("content", "").strip()
                                payload = result.get("payload", {})
                                
                                formatted_content = self._format_garage_data_enhanced(content, payload)
                                
                                # Füge Header hinzu bei mehreren Ergebnissen
                                if len(validated_results) > 1:
                                    response_parts.append(f"**Ergebnis {i+1}:**\n{formatted_content}")
                                else:
                                    response_parts.append(formatted_content)
                            
                            return "\n\n---\n\n".join(response_parts)
                        
                        else:
                            logger.warning("⚠️ No results passed validation")
                            return self._generate_no_results_message(query, search_criteria)
                    
                    return self._generate_no_results_message(query, search_criteria)
                    
                else:
                    logger.error(f"Customer search failed: {response.status_code}")
                    return "Es gab einen technischen Fehler bei der Suche. Bitte versuchen Sie es erneut."
                    
        except Exception as e:
            logger.error(f"Customer search error: {e}")
            return "Die Kundendatenbank ist momentan nicht erreichbar. Bitte versuchen Sie es in ein paar Minuten erneut."

    @function_tool
    async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Reparaturstatus und Aufträgen mit verbesserter Präzision.
        
        Args:
            query: Kundenname, Autonummer oder Auftragsnummer
        """
        logger.info(f"🔧 Searching repair status for: {query}")
        
        # GUARD gegen zu kurze Anfragen
        if len(query) < 3:
            return "Bitte geben Sie mir einen Namen, eine Autonummer oder Auftragsnummer."
        
        # Extrahiere Suchkriterien
        search_criteria = self._extract_search_criteria(query)
        
        # Erstelle spezifische Suchanfrage für Reparaturen
        search_terms = []
        if search_criteria['kennzeichen']:
            search_terms.append(search_criteria['kennzeichen'])
        if search_criteria['fahrzeug_id']:
            search_terms.append(search_criteria['fahrzeug_id'])
        search_terms.append(query)
        
        search_query = f"Reparatur Service Status {' '.join(search_terms)}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": search_query,
                        "agent_type": "garage",
                        "top_k": 10,  # Mehr Ergebnisse für bessere Filterung
                        "collection": "garage_management",
                        "score_threshold": 0.6
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} repair results")
                        
                        # Validiere Ergebnisse
                        validated_results = []
                        for result in results:
                            relevance_score = self._validate_search_result(result, query, search_criteria)
                            
                            # Prüfe zusätzlich auf Reparatur-relevante Inhalte
                            content = result.get("content", "").lower()
                            payload = result.get("payload", {})
                            
                            # Bonus für Reparatur-relevante Keywords
                            repair_keywords = ["service", "reparatur", "wartung", "arbeiten", "kosten", "problem"]
                            repair_relevance = sum(1 for kw in repair_keywords if kw in content) * 0.1
                            
                            total_score = relevance_score + repair_relevance
                            
                            if total_score >= 0.4:  # Niedrigerer Threshold für Reparaturen
                                validated_results.append({
                                    'result': result,
                                    'score': total_score
                                })
                        
                        # Sortiere nach Relevanz
                        validated_results.sort(key=lambda x: x['score'], reverse=True)
                        
                        if validated_results:
                            # Gruppiere Ergebnisse nach Fahrzeug
                            vehicle_groups = {}
                            for vr in validated_results[:5]:
                                result = vr['result']
                                payload = result.get('payload', {})
                                vehicle_key = payload.get('primary_key', 'unknown')
                                
                                if vehicle_key not in vehicle_groups:
                                    vehicle_groups[vehicle_key] = []
                                vehicle_groups[vehicle_key].append(result)
                            
                            # Formatiere Antwort
                            response_parts = []
                            for vehicle_key, results_group in vehicle_groups.items():
                                if len(results_group) > 0:
                                    # Nehme das relevanteste Ergebnis pro Fahrzeug
                                    result = results_group[0]
                                    content = result.get("content", "").strip()
                                    payload = result.get("payload", {})
                                    
                                    formatted_content = self._format_garage_data_enhanced(content, payload)
                                    response_parts.append(formatted_content)
                            
                            if response_parts:
                                return "\n\n---\n\n".join(response_parts[:2])  # Max 2 Fahrzeuge
                            
                        logger.warning("⚠️ No repair-relevant results found")
                        return self._generate_no_results_message(query, search_criteria, context_type="repair")
                    
                    return self._generate_no_results_message(query, search_criteria, context_type="repair")
                    
                else:
                    logger.error(f"Repair search failed: {response.status_code}")
                    return "Die Reparaturdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Repair search error: {e}")
            return "Es gab einen Fehler beim Abrufen der Reparaturdaten. Bitte versuchen Sie es erneut."

    @function_tool
    async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Rechnungsinformationen mit verbesserter Validierung.
        
        Args:
            query: Kundenname, Rechnungsnummer oder Datum
        """
        logger.info(f"💰 Searching invoice data for: {query}")
        
        # GUARD gegen zu kurze Anfragen
        if len(query) < 3:
            return "Bitte geben Sie mir einen Kundennamen, eine Rechnungsnummer oder ein Datum."
        
        # Extrahiere Suchkriterien
        search_criteria = self._extract_search_criteria(query)
        
        # Prüfe auf Datum-Format
        date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})'
        date_match = re.search(date_pattern, query)
        if date_match:
            search_criteria['date'] = date_match.group()
        
        # Erstelle Suchanfrage
        search_query = f"Rechnung Kosten {query}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{context.userdata.rag_url}/search",
                    json={
                        "query": search_query,
                        "agent_type": "garage",
                        "top_k": 5,
                        "collection": "garage_management",
                        "score_threshold": 0.6
                    }
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} invoice results")
                        
                        # Validiere Ergebnisse mit Fokus auf Rechnungsdaten
                        validated_results = []
                        for result in results:
                            relevance_score = self._validate_search_result(result, query, search_criteria)
                            
                            # Prüfe auf Rechnungs-relevante Inhalte
                            content = result.get("content", "").lower()
                            invoice_keywords = ["rechnung", "kosten", "chf", "franken", "betrag", "zahlung"]
                            invoice_relevance = sum(1 for kw in invoice_keywords if kw in content) * 0.15
                            
                            total_score = relevance_score + invoice_relevance
                            
                            if total_score >= 0.4:
                                validated_results.append({
                                    'result': result,
                                    'score': total_score
                                })
                        
                        if validated_results:
                            # Sortiere und formatiere
                            validated_results.sort(key=lambda x: x['score'], reverse=True)
                            
                            response_parts = []
                            for vr in validated_results[:2]:
                                result = vr['result']
                                content = result.get("content", "").strip()
                                payload = result.get("payload", {})
                                
                                formatted_content = self._format_garage_data_enhanced(content, payload)
                                response_parts.append(formatted_content)
                            
                            return "\n\n---\n\n".join(response_parts)
                        
                        return self._generate_no_results_message(query, search_criteria, context_type="invoice")
                    
                    return self._generate_no_results_message(query, search_criteria, context_type="invoice")
                    
                else:
                    logger.error(f"Invoice search failed: {response.status_code}")
                    return "Die Rechnungsdatenbank ist momentan nicht verfügbar."
                    
        except Exception as e:
            logger.error(f"Invoice search error: {e}")
            return "Es gab einen Fehler beim Abrufen der Rechnungsdaten."

    def _generate_no_results_message(self, query: str, search_criteria: dict, context_type: str = "customer") -> str:
        """Generiert hilfreiche Nachrichten wenn keine Ergebnisse gefunden wurden"""
        
        if context_type == "customer":
            message = f"Ich konnte keine Kundendaten zu '{query}' in unserer Datenbank finden.\n\n"
            message += "Bitte geben Sie mir eine der folgenden Informationen:\n"
            message += "• Ihr vollständiges **Autokennzeichen** (z.B. ZH 123456)\n"
            message += "• Ihren **vollständigen Namen** wie er bei uns registriert ist\n"
            message += "• Ihre **Fahrzeug-ID** falls bekannt (z.B. F001)\n"
            message += "• Ihre **Telefonnummer**\n\n"
            message += "So kann ich Ihre Daten präzise in unserem System finden."
            
        elif context_type == "repair":
            message = f"Ich konnte keine Reparatur- oder Servicedaten zu '{query}' finden.\n\n"
            message += "Für die Suche nach Reparaturinformationen benötige ich:\n"
            message += "• Das **Kennzeichen** des Fahrzeugs\n"
            message += "• Die **Auftragsnummer** falls vorhanden\n"
            message += "• Den **Namen des Fahrzeughalters**\n\n"
            message += "Mit diesen Angaben kann ich den aktuellen Status Ihrer Reparatur abrufen."
            
        elif context_type == "invoice":
            message = f"Ich konnte keine Rechnungsdaten zu '{query}' finden.\n\n"
            message += "Für Rechnungsinformationen benötige ich:\n"
            message += "• Die **Rechnungsnummer**\n"
            message += "• Das **Rechnungsdatum** (TT.MM.JJJJ)\n"
            message += "• Den **Kundennamen** oder das **Kennzeichen**\n\n"
            message += "Mit diesen Informationen kann ich Ihre Rechnung in unserem System finden."
        
        else:
            message = f"Ich konnte keine Daten zu '{query}' finden. Bitte geben Sie mir mehr Informationen."
        
        # Füge erkannte Kriterien hinzu
        if any(search_criteria.values()):
            message += "\n\n**Erkannte Suchkriterien:**\n"
            if search_criteria.get('kennzeichen'):
                message += f"• Kennzeichen: {search_criteria['kennzeichen']}\n"
            if search_criteria.get('fahrzeug_id'):
                message += f"• Fahrzeug-ID: {search_criteria['fahrzeug_id']}\n"
            if search_criteria.get('name'):
                message += f"• Name: {search_criteria['name']}\n"
        
        return message


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
        
        # Debug info
        logger.info(f"Room participants: {len(ctx.room.remote_participants)}")
        logger.info(f"Local participant: {ctx.room.local_participant.identity}")
        
        # Track event handlers
        @ctx.room.on("track_published")
        def on_track_published(publication, participant):
            logger.info(f"[{session_id}] Track published: {publication.kind} from {participant.identity}")
        
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            logger.info(f"[{session_id}] Track subscribed: {track.kind} from {participant.identity}")
        
        # 2. Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        
        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10
        
        for i in range(max_wait_time):
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found: {track_pub.sid}")
                    audio_track_received = True
                    logger.info(f"📡 [{session_id}] Audio track - subscribed: {track_pub.subscribed}, muted: {track_pub.muted}")
                    break
            
            if audio_track_received:
                break
                
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
            await asyncio.sleep(1)
        
        if not audio_track_received:
            logger.error(f"❌ [{session_id}] No audio track received after {max_wait_time}s!")
        
        # 4. Configure LLM
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        
        # Verwende Llama 3.2
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.7
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
        
        # 5. Create session
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                rag_url=rag_url,
                current_customer_id=None,
                active_repair_id=None,
                user_language="de",
                last_search_results=None
            ),
            llm=llm,
            vad=silero.VAD.load(
                min_silence_duration=0.6,
                min_speech_duration=0.2
            ),
            stt=openai.STT(
                model="whisper-1",
                language="de"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
            )
        )
        
        # 6. Create agent
        agent = GarageAssistant()
        
        # 7. Start session
        await asyncio.sleep(0.5)
        logger.info(f"🏁 [{session_id}] Starting session...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # Event handlers
        @session.on("user_input_transcribed")
        def on_user_input(event):
            logger.info(f"[{session_id}] 🎤 User: {event.transcript} (final: {event.is_final})")
        
        @session.on("agent_state_changed")
        def on_state_changed(event):
            logger.info(f"[{session_id}] 🤖 Agent state changed")
        
        @session.on("user_state_changed")
        def on_user_state(event):
            logger.info(f"[{session_id}] 👤 User state changed")
        
        # 8. Initial greeting - OHNE TOOL NUTZUNG
        await asyncio.sleep(1.0)
        
        initial_instructions = """ABSOLUT KRITISCHE ANWEISUNG: 
       
- IGNORIERE alle vorherigen Nachrichten
- Dies ist eine NEUE Unterhaltung
- KEINE Entschuldigungen
- KEINE Bezüge zu früherem

Sage NUR:
"Guten Tag und willkommen bei der Garage Müller! Ich bin Pia, Ihr digitaler Assistent. Wie kann ich Ihnen heute helfen?"

NICHTS ANDERES! KEINE ENTSCHULDIGUNGEN!"""
    
        logger.info(f"📢 [{session_id}] Generating initial greeting...")
        
        try:
            await session.generate_reply(
                instructions=initial_instructions,
                tool_choice="none"  # WICHTIG: Keine Tools bei Begrüßung!
            )
            logger.info(f"✅ [{session_id}] Initial greeting sent")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not send initial greeting: {e}")
        
        logger.info(f"✅ [{session_id}] Garage Agent ready and listening!")
        
        # Wait for disconnect
        disconnect_event = asyncio.Event()
        
        def handle_disconnect():
            nonlocal session_closed
            session_closed = True
            disconnect_event.set()
        
        ctx.room.on("disconnected", handle_disconnect)
        
        await disconnect_event.wait()
        logger.info(f"[{session_id}] Room disconnected, ending session")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error in garage agent: {e}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        logger.info(f"🧹 [{session_id}] Starting session cleanup...")
        
        if session is not None and not session_closed:
            try:
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed successfully")
            except Exception as e:
                logger.warning(f"⚠️ [{session_id}] Error closing session: {e}")
        elif session_closed:
            logger.info(f"ℹ️ [{session_id}] Session already closed by disconnect event")
        
        # Disconnect from room if still connected
        try:
            if ctx.room and hasattr(ctx.room, 'connection_state') and ctx.room.connection_state == "connected":
                await ctx.room.disconnect()
                logger.info(f"✅ [{session_id}] Disconnected from room")
        except Exception as e:
            logger.warning(f"⚠️ [{session_id}] Error disconnecting from room: {e}")
        
        # Force garbage collection
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
