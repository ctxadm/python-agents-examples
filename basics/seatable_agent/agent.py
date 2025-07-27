import logging
import os
import httpx
import asyncio
import json
import re
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
logger = logging.getLogger("garage-seatable-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-seatable")

@dataclass
class GarageUserData:
    """User data context für den Garage Agent"""
    authenticated_user: Optional[str] = None
    seatable_base_token: str = ""
    seatable_server_url: str = ""
    seatable_access_token: Optional[str] = None
    current_customer_id: Optional[str] = None
    active_repair_id: Optional[str] = None
    user_language: str = "de"


class SeaTableClient:
    """Client für SeaTable API Operationen"""
    
    def __init__(self, server_url: str, base_token: str):
        self.server_url = server_url.rstrip('/')
        self.base_token = base_token
        self.access_token = None
        self.dtable_uuid = None
        self.dtable_server = None
        
    async def get_access_token(self) -> str:
        """Holt den Access Token mit dem Base Token"""
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(
                f"{self.server_url}/api/v2.1/dtable/app-access-token/",
                headers={"Authorization": f"Bearer {self.base_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                self.dtable_uuid = data.get("dtable_uuid")
                self.dtable_server = data.get("dtable_server", self.server_url)
                logger.info(f"✅ SeaTable access token obtained")
                return self.access_token
            else:
                logger.error(f"Failed to get access token: {response.status_code}")
                raise Exception(f"Failed to get SeaTable access token")
    
    async def query_unified_table(self, table_name: str, filters: Dict[str, str] = None) -> List[Dict]:
        """Query the unified table with optional filters"""
        try:
            if not self.access_token:
                await self.get_access_token()
            
            # Get all rows from the unified table
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(
                    f"{self.server_url}/dtable-server/api/v1/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={"table_name": table_name}
                )
                
            if response.status_code == 200:
                data = response.json()
                rows = data.get("rows", [])
                logger.info(f"✅ Retrieved {len(rows)} rows from table {table_name}")
                
                # Apply filters if provided
                if filters:
                    filtered_rows = []
                    for row in rows:
                        match = True
                        for field, value in filters.items():
                            if field in row and value.lower() not in str(row.get(field, '')).lower():
                                match = False
                                break
                        if match:
                            filtered_rows.append(row)
                    rows = filtered_rows
                    logger.info(f"✅ Filtered to {len(rows)} rows")
                
                return rows
            else:
                logger.error(f"Failed to fetch rows: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return []


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen"""
    
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
- Bei "Hallo", "Guten Tag", "Hi" antworte mit: "Guten Tag! Wie kann ich Ihnen helfen?"
- KEINE langen Erklärungen
- KEINE Suchfunktionen bei Begrüßungen
- Kurz und direkt antworten

ABSOLUT KRITISCHE REGEL - NIEMALS DATEN ERFINDEN:
- NIEMALS Informationen erfinden, raten oder halluzinieren!
- Wenn die Datenbank "keine Daten gefunden" meldet, sage das EHRLICH
- Erfinde KEINE Daten, Termine, Daten oder Services die nicht existieren
- Sage NIEMALS Dinge wie "Ihr letzter Service war am..." wenn keine Daten gefunden wurden
- Bei "keine Daten gefunden" frage nach mehr Details (z.B. Autonummer, Kennzeichen)
- Gib NUR Informationen weiter, die DIREKT aus der Datenbank kommen

KRITISCHE REGEL FÜR PREISAUSKÜNFTE:
- Bei Preisanfragen IMMER nur tatsächliche Daten aus der Datenbank nennen
- NIEMALS Preise aufschlüsseln oder berechnen, die nicht explizit so in der Datenbank stehen
- Bei allgemeinen Preisanfragen ohne konkrete Kundendaten sage: 
  "Ich kann Ihnen gerne Beispiele aus vergangenen Services nennen, aber für aktuelle Preise sollten Sie direkt bei uns anrufen."
- Wenn Preise gefunden werden, sage IMMER dazu: "Das war der Preis bei diesem spezifischen Service"
- ERFINDE NIEMALS Paketpreise oder Rabatte, die nicht in der Datenbank stehen

WORKFLOW:
1. Bei Begrüßung: Freundlich antworten und nach dem Anliegen fragen
2. Höre aufmerksam zu und identifiziere das konkrete Anliegen
3. NUR bei konkreten Anfragen: Nutze die passende Suchfunktion
4. Gib NUR präzise Auskünfte basierend auf den tatsächlichen Datenbankdaten

DEINE AUFGABEN:
- Kundendaten abfragen (Name, Fahrzeug, Kontaktdaten)
- Fahrzeugdaten anzeigen (Kennzeichen, Marke, Modell, Kilometerstand)
- Service-Historie mitteilen
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
        logger.info("✅ GarageAssistant initialized")

    @function_tool
    async def search_customer_data(self, 
                                  context: RunContext[GarageUserData],
                                  query: str) -> str:
        """
        Sucht nach Kundendaten in der Garage-Datenbank.
        
        Args:
            query: Suchbegriff (Name, Telefonnummer oder Email)
        """
        logger.info(f"🔍 Searching customer data for: {query}")
        
        # GUARD gegen falsche Suchen bei Begrüßungen
        if len(query) < 3 or query.lower() in ["hallo", "guten tag", "hi", "hey", "servus", "grüezi"]:
            logger.warning(f"⚠️ Ignoring greeting search: {query}")
            return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder Email-Adresse, damit ich Ihre Kundendaten finden kann."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Suche in der vereinheitlichten Tabelle
            results = await client.query_unified_table("Garage_Gesamtdaten")
            
            # Filtere nach Kundennamen, Email oder Telefon
            matching_customers = []
            seen_customers = set()
            
            for row in results:
                kunde_name = row.get('Kunden_Name', '')
                email = row.get('Email', '')
                telefon = row.get('Telefon', '')
                
                # Check if query matches
                if (query.lower() in kunde_name.lower() or 
                    query.lower() in email.lower() or 
                    query.lower() in telefon.lower()):
                    
                    # Avoid duplicates
                    if kunde_name not in seen_customers:
                        seen_customers.add(kunde_name)
                        matching_customers.append({
                            'Name': kunde_name,
                            'Email': email,
                            'Telefon': telefon,
                            'Kunde_seit': row.get('Kunde_seit', '-'),
                            'Notizen': row.get('Notizen', ''),
                            'Fahrzeuge': row.get('Fahrzeuge_Details', ''),
                            'Kennzeichen': row.get('Fahrzeuge_Kennzeichen', '')
                        })
            
            if matching_customers:
                logger.info(f"✅ Found {len(matching_customers)} customer results")
                
                response_text = "Ich habe folgende Kundendaten gefunden:\n\n"
                for customer in matching_customers[:5]:
                    response_text += f"**{customer['Name']}**\n"
                    response_text += f"- Email: {customer['Email']}\n"
                    response_text += f"- Telefon: {customer['Telefon']}\n"
                    response_text += f"- Kunde seit: {customer['Kunde_seit']}\n"
                    if customer['Fahrzeuge']:
                        response_text += f"- Fahrzeuge: {customer['Fahrzeuge']}\n"
                    if customer['Kennzeichen']:
                        response_text += f"- Kennzeichen: {customer['Kennzeichen']}\n"
                    if customer['Notizen']:
                        response_text += f"- Notizen: {customer['Notizen']}\n"
                    response_text += "\n"
                
                # Speichere Kundennamen
                if len(matching_customers) == 1:
                    context.userdata.authenticated_user = matching_customers[0]['Name']
                
                return response_text
            else:
                return "Ich konnte keine Kundendaten zu Ihrer Anfrage finden. Können Sie mir bitte Ihren vollständigen Namen oder Ihre Telefonnummer nennen?"
                
        except Exception as e:
            logger.error(f"Customer search error: {e}")
            return "Die Kundendatenbank ist momentan nicht erreichbar."

    @function_tool
    async def search_vehicle_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
        """
        Sucht nach Fahrzeugdaten in der Datenbank.
        
        Args:
            query: Kennzeichen oder Besitzername
        """
        logger.info(f"🚗 Searching vehicle data for: {query}")
        
        if len(query) < 2:
            return "Bitte geben Sie mir ein Kennzeichen oder einen Kundennamen."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Suche in der vereinheitlichten Tabelle
            results = await client.query_unified_table("Garage_Gesamtdaten")
            
            # Filtere nach Kennzeichen oder Besitzer
            matching_vehicles = []
            seen_vehicles = set()
            
            for row in results:
                kennzeichen = row.get('Kennzeichen', '')
                besitzer = row.get('Besitzer_Name', '')
                
                # Check if query matches
                if (query.lower() in kennzeichen.lower() or 
                    query.lower() in besitzer.lower()):
                    
                    # Avoid duplicates
                    vehicle_key = f"{kennzeichen}_{besitzer}"
                    if vehicle_key not in seen_vehicles:
                        seen_vehicles.add(vehicle_key)
                        matching_vehicles.append({
                            'Kennzeichen': kennzeichen,
                            'Besitzer': besitzer,
                            'Marke': row.get('Marke', '-'),
                            'Modell': row.get('Modell', '-'),
                            'Baujahr': row.get('Baujahr', '-'),
                            'Farbe': row.get('Farbe', '-'),
                            'Kraftstoff': row.get('Kraftstoff', '-'),
                            'Kilometerstand': row.get('Kilometerstand', '-'),
                            'Letzte_Inspektion': row.get('Letzte_Inspektion', '-'),
                            'VIN': row.get('VIN', '-')
                        })
            
            if matching_vehicles:
                logger.info(f"✅ Found {len(matching_vehicles)} vehicle results")
                
                response_text = "Hier sind die Fahrzeugdaten:\n\n"
                for vehicle in matching_vehicles[:5]:
                    response_text += f"**{vehicle['Kennzeichen']}**\n"
                    response_text += f"- Besitzer: {vehicle['Besitzer']}\n"
                    response_text += f"- Marke/Modell: {vehicle['Marke']} {vehicle['Modell']}\n"
                    response_text += f"- Baujahr: {vehicle['Baujahr']}\n"
                    response_text += f"- Farbe: {vehicle['Farbe']}\n"
                    response_text += f"- Kraftstoff: {vehicle['Kraftstoff']}\n"
                    response_text += f"- Kilometerstand: {vehicle['Kilometerstand']} km\n"
                    response_text += f"- Letzte Inspektion: {vehicle['Letzte_Inspektion']}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return "Ich konnte keine Fahrzeugdaten zu Ihrer Anfrage finden. Bitte geben Sie das vollständige Kennzeichen oder den Namen des Besitzers an."
                
        except Exception as e:
            logger.error(f"Vehicle search error: {e}")
            return "Die Fahrzeugdatenbank ist momentan nicht verfügbar."

    @function_tool
    async def search_service_history(self, 
                                   context: RunContext[GarageUserData],
                                   query: str) -> str:
        """
        Sucht nach Service-Historie und Reparaturen.
        
        Args:
            query: Kundenname, Kennzeichen oder Datum
        """
        logger.info(f"🔧 Searching service history for: {query}")
        
        if len(query) < 2:
            return "Bitte geben Sie mir einen Namen, ein Kennzeichen oder ein Datum."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Suche in der vereinheitlichten Tabelle
            results = await client.query_unified_table("Garage_Gesamtdaten")
            
            # Filtere nach Service-Einträgen
            matching_services = []
            
            for row in results:
                kunde = row.get('Kunden_Name', '')
                kennzeichen = row.get('Kennzeichen', '')
                service_datum = row.get('Service_Datum', '')
                service_beschreibung = row.get('Service_Beschreibung', '')
                
                # Skip rows without service data
                if not service_datum or not service_beschreibung:
                    continue
                
                # Check if query matches
                if (query.lower() in kunde.lower() or 
                    query.lower() in kennzeichen.lower() or 
                    query.lower() in str(service_datum).lower()):
                    
                    matching_services.append({
                        'Datum': service_datum,
                        'Kunde': kunde,
                        'Kennzeichen': kennzeichen,
                        'Marke_Modell': f"{row.get('Marke', '')} {row.get('Modell', '')}",
                        'Beschreibung': service_beschreibung,
                        'Kosten': row.get('Service_Kosten', 0),
                        'Mechaniker': row.get('Service_Mechaniker', '-'),
                        'Status': row.get('Service_Status', '-')
                    })
            
            # Sort by date (newest first)
            matching_services.sort(key=lambda x: x['Datum'], reverse=True)
            
            if matching_services:
                logger.info(f"✅ Found {len(matching_services)} service results")
                
                response_text = "Hier ist die Service-Historie:\n\n"
                for service in matching_services[:5]:
                    response_text += f"**Service vom {service['Datum']}**\n"
                    response_text += f"- Fahrzeug: {service['Kennzeichen']} ({service['Marke_Modell']})\n"
                    response_text += f"- Beschreibung: {service['Beschreibung']}\n"
                    response_text += f"- Status: {service['Status']}\n"
                    response_text += f"- Mechaniker: {service['Mechaniker']}\n"
                    if service['Kosten']:
                        response_text += f"- Kosten: {service['Kosten']} Franken\n"
                    response_text += "\n"
                
                # Berechne Gesamtkosten für diesen Kunden
                if len(set(s['Kunde'] for s in matching_services)) == 1:
                    total_cost = sum(float(s['Kosten']) for s in matching_services if s['Kosten'])
                    response_text += f"\n**Gesamtkosten aller Services: {total_cost} Franken**"
                
                return response_text
            else:
                return "Ich konnte keine Service-Einträge zu Ihrer Anfrage finden. Bitte geben Sie ein Kennzeichen oder einen Kundennamen an."
                
        except Exception as e:
            logger.error(f"Service search error: {e}")
            return "Die Service-Datenbank ist momentan nicht verfügbar."

    @function_tool
    async def get_price_information(self, 
                                  context: RunContext[GarageUserData],
                                  service_type: str) -> str:
        """
        Sucht nach Preisinformationen für bestimmte Services.
        
        Args:
            service_type: Art des Services (z.B. "Ölwechsel", "Inspektion")
        """
        logger.info(f"💰 Searching price information for: {service_type}")
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Suche in der vereinheitlichten Tabelle nach ähnlichen Services
            results = await client.query_unified_table("Garage_Gesamtdaten")
            
            # Filtere nach Service-Typ
            matching_services = []
            
            for row in results:
                service_beschreibung = row.get('Service_Beschreibung', '')
                service_kosten = row.get('Service_Kosten', 0)
                
                # Skip rows without service data
                if not service_beschreibung or not service_kosten:
                    continue
                
                # Check if service type matches
                if service_type.lower() in service_beschreibung.lower():
                    matching_services.append({
                        'Beschreibung': service_beschreibung,
                        'Kosten': service_kosten,
                        'Datum': row.get('Service_Datum', '-'),
                        'Fahrzeug': f"{row.get('Marke', '')} {row.get('Modell', '')}"
                    })
            
            if matching_services:
                logger.info(f"✅ Found {len(matching_services)} price examples")
                
                # Gruppiere nach Service-Typ und berechne Durchschnitt
                from collections import defaultdict
                service_groups = defaultdict(list)
                
                for service in matching_services:
                    service_groups[service['Beschreibung']].append(float(service['Kosten']))
                
                response_text = f"Hier sind Beispiele für '{service_type}' aus unserer Historie:\n\n"
                
                for beschreibung, kosten_liste in service_groups.items():
                    avg_cost = sum(kosten_liste) / len(kosten_liste)
                    min_cost = min(kosten_liste)
                    max_cost = max(kosten_liste)
                    
                    response_text += f"**{beschreibung}**\n"
                    if len(kosten_liste) > 1:
                        response_text += f"- Durchschnittspreis: {avg_cost:.2f} Franken\n"
                        response_text += f"- Preisspanne: {min_cost:.2f} - {max_cost:.2f} Franken\n"
                        response_text += f"- Anzahl Services: {len(kosten_liste)}\n"
                    else:
                        response_text += f"- Preis: {kosten_liste[0]} Franken\n"
                    response_text += "\n"
                
                response_text += "\n⚠️ **Hinweis**: Dies sind historische Preise. Für aktuelle Preise und ein individuelles Angebot kontaktieren Sie uns bitte direkt."
                
                return response_text
            else:
                return f"Ich konnte keine Preisbeispiele für '{service_type}' in unserer Historie finden. Bitte rufen Sie uns für eine individuelle Preisauskunft an."
                
        except Exception as e:
            logger.error(f"Price search error: {e}")
            return "Die Preisdatenbank ist momentan nicht verfügbar."


async def request_handler(ctx: JobContext):
    """Request handler - KRITISCH für Job-Akzeptierung!"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    
    # KRITISCH: Job MUSS akzeptiert werden!
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent mit SeaTable"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚗 Starting Garage SeaTable Agent Session: {session_id}")
    logger.info("="*50)
    
    # DEBUG: Environment Check
    logger.info("🔍 Environment Check:")
    logger.info(f"  LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'NOT SET')}")
    logger.info(f"  SEATABLE_BASE_TOKEN: {'***' if os.getenv('SEATABLE_BASE_TOKEN') else 'NOT SET'}")
    logger.info(f"  SEATABLE_SERVER_URL: {os.getenv('SEATABLE_SERVER_URL', 'NOT SET')}")
    
    # SeaTable Config aus .env
    seatable_token = os.getenv("SEATABLE_BASE_TOKEN")
    seatable_url = os.getenv("SEATABLE_SERVER_URL", "https://cloud.seatable.io")
    
    if not seatable_token:
        logger.error("❌ SEATABLE_BASE_TOKEN not set!")
        raise ValueError("SeaTable configuration missing")
    
    session = None
    session_closed = False
    disconnect_event = asyncio.Event()
    
    # Single disconnect handler
    def on_disconnect():
        nonlocal session_closed
        if not session_closed:
            logger.info(f"[{session_id}] 🔌 Room disconnected event received")
            session_closed = True
            disconnect_event.set()
    
    try:
        # 1. Connect to room
        await ctx.connect()
        logger.info(f"✅ [{session_id}] Connected to room")
        
        # Register disconnect handler AFTER connection
        if ctx.room:
            ctx.room.on("disconnected", on_disconnect)
        
        # Debug info
        logger.info(f"Room participants: {len(ctx.room.remote_participants)}")
        logger.info(f"Local participant: {ctx.room.local_participant.identity}")
        
        # 2. Wait for participant with timeout
        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                timeout=30.0  # 30 second timeout
            )
            logger.info(f"✅ [{session_id}] Participant joined: {participant.identity}")
        except asyncio.TimeoutError:
            logger.error(f"❌ [{session_id}] No participant joined within 30 seconds")
            return
        
        # 3. Wait for audio track
        audio_track_received = False
        max_wait_time = 10
        
        for i in range(max_wait_time):
            if session_closed:
                logger.info(f"[{session_id}] Session closed during audio wait")
                return
                
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ [{session_id}] Audio track found: {track_pub.sid}")
                    audio_track_received = True
                    break
            
            if audio_track_received:
                break
                
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
            await asyncio.sleep(1)
        
        if not audio_track_received:
            logger.error(f"❌ [{session_id}] No audio track received after {max_wait_time}s!")
            return
        
        # 4. Configure LLM
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.1
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
        
        # 5. Create session
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                seatable_base_token=seatable_token,
                seatable_server_url=seatable_url,
                current_customer_id=None,
                active_repair_id=None,
                user_language="de"
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
        
        # 8. Initial greeting
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
                tool_choice="none"
            )
            logger.info(f"✅ [{session_id}] Initial greeting sent")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not send initial greeting: {e}")
        
        logger.info(f"✅ [{session_id}] Garage SeaTable Agent ready and listening!")
        
        # Wait for disconnect with timeout
        try:
            await asyncio.wait_for(
                disconnect_event.wait(),
                timeout=300.0  # 5 minute maximum session time
            )
            logger.info(f"[{session_id}] Disconnect event received")
        except asyncio.TimeoutError:
            logger.info(f"[{session_id}] Session timeout after 5 minutes")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error in garage agent: {e}", exc_info=True)
        
    finally:
        # CRITICAL: Comprehensive cleanup
        logger.info(f"🧹 [{session_id}] Starting comprehensive cleanup...")
        
        # 1. Close session first
        if session is not None and not session_closed:
            try:
                logger.info(f"[{session_id}] Closing agent session...")
                await session.aclose()
                logger.info(f"✅ [{session_id}] Session closed successfully")
            except Exception as e:
                logger.warning(f"⚠️ [{session_id}] Error closing session: {e}")
        
        # 2. CRITICAL: Disconnect from room explicitly
        if ctx.room and not session_closed:
            try:
                logger.info(f"[{session_id}] Disconnecting from room...")
                await ctx.room.disconnect()
                logger.info(f"✅ [{session_id}] Room disconnected successfully")
            except Exception as e:
                logger.warning(f"⚠️ [{session_id}] Error disconnecting room: {e}")
        
        # 3. Clear all references
        session = None
        agent = None
        
        # 4. Force garbage collection
        import gc
        gc.collect()
        logger.info(f"♻️ [{session_id}] Forced garbage collection")
        
        # 5. Small delay to ensure resources are fully released
        await asyncio.sleep(0.5)
        
        logger.info(f"✅ [{session_id}] Cleanup complete - room freed")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
