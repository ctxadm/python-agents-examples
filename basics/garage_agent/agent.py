# LiveKit Agents - Garage Management Agent mit SeaTable
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
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

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
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.server_url}/api/v2.1/dtable/app-access-token/",
                headers={"Authorization": f"Token {self.base_token}"}  # Token statt Bearer für API Token
            )
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                self.dtable_uuid = data.get("dtable_uuid")
                # Nutze dtable_server falls vorhanden, sonst server_url
                self.dtable_server = data.get("dtable_server", self.server_url)
                logger.info(f"✅ SeaTable access token obtained - UUID: {self.dtable_uuid}")
                return self.access_token
            else:
                logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
                raise Exception(f"Failed to get SeaTable access token: {response.text}")
    
    async def query_table(self, sql_query: str) -> List[Dict]:
        """Führt eine SQL-ähnliche Abfrage auf SeaTable aus"""
        if not self.access_token:
            await self.get_access_token()
            
        async with httpx.AsyncClient() as client:
            # Nutze den neuen API Gateway Endpoint für bessere Performance
            try:
                # Versuche zuerst den neuen API Gateway Endpoint
                response = await client.post(
                    f"{self.server_url}/api-gateway/api/v2/dtables/{self.dtable_uuid}/sql",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={"sql": sql_query}
                )
            except Exception as e:
                logger.info("API Gateway nicht verfügbar, nutze Legacy Endpoint")
                # Fallback zum alten Endpoint
                response = await client.post(
                    f"{self.dtable_server}/dtable-db/api/v1/query/{self.dtable_uuid}/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={"sql": sql_query}
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("results", [])
            else:
                logger.error(f"SeaTable query failed: {response.status_code} - {response.text}")
                return []
    
    async def get_rows(self, table_name: str, view_name: Optional[str] = None) -> List[Dict]:
        """Holt alle Zeilen aus einer Tabelle"""
        if not self.access_token:
            await self.get_access_token()
            
        async with httpx.AsyncClient() as client:
            params = {"table_name": table_name}
            if view_name:
                params["view_name"] = view_name
                
            try:
                # Nutze neuen API Gateway Endpoint
                response = await client.get(
                    f"{self.server_url}/api-gateway/api/v2/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params=params
                )
            except Exception as e:
                # Fallback zum alten Endpoint
                response = await client.get(
                    f"{self.dtable_server}/api/v1/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params=params
                )
            
            if response.status_code == 200:
                return response.json().get("rows", [])
            else:
                logger.error(f"Failed to get rows: {response.text}")
                return []
    
    async def add_row(self, table_name: str, row_data: Dict) -> bool:
        """Fügt eine neue Zeile zu einer Tabelle hinzu"""
        if not self.access_token:
            await self.get_access_token()
            
        async with httpx.AsyncClient() as client:
            # API Gateway Endpoint für Rows
            try:
                response = await client.post(
                    f"{self.server_url}/api-gateway/api/v2/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={
                        "table_name": table_name,
                        "rows": [row_data]  # API Gateway erwartet Array
                    }
                )
            except Exception as e:
                # Fallback zum alten Endpoint
                response = await client.post(
                    f"{self.dtable_server}/api/v1/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={
                        "table_name": table_name,
                        "row": row_data
                    }
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Row added successfully to {table_name}")
                return True
            else:
                logger.error(f"Failed to add row: {response.status_code} - {response.text}")
                return False
    
    async def update_row(self, table_name: str, row_id: str, updates: Dict) -> bool:
        """Aktualisiert eine Zeile in einer Tabelle"""
        if not self.access_token:
            await self.get_access_token()
            
        async with httpx.AsyncClient() as client:
            try:
                # API Gateway Endpoint
                response = await client.put(
                    f"{self.server_url}/api-gateway/api/v2/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={
                        "table_name": table_name,
                        "row_id": row_id,
                        "row": updates
                    }
                )
            except Exception as e:
                # Fallback
                response = await client.put(
                    f"{self.dtable_server}/api/v1/dtables/{self.dtable_uuid}/rows/",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    json={
                        "table_name": table_name,
                        "row_id": row_id,
                        "row": updates
                    }
                )
            
            return response.status_code == 200


class GarageAssistant(Agent):
    """Garage Assistant für Kundenverwaltung und Reparaturen mit SeaTable"""
    
    def __init__(self) -> None:
        super().__init__(instructions="""Du bist Pia, der digitale Assistent der Garage Müller.

ABSOLUT KRITISCHE MEMORY REGEL:
- Du hast KEIN Gedächtnis für vorherige Nachrichten
- Jede Nachricht ist eine NEUE Konversation
- Entschuldige dich NIEMALS für irgendwas
- Ignoriere KOMPLETT was vorher gesagt wurde

KRITISCHE REGEL FÜR BEGRÜSSUNGEN:
- Bei einfachen Begrüßungen wie "Hallo", "Guten Tag", "Hi" antworte NUR mit einer freundlichen Begrüßung
- Nutze NIEMALS Suchfunktionen bei einer einfachen Begrüßung

ABSOLUT KRITISCHE REGEL - NIEMALS DATEN ERFINDEN:
- NIEMALS Informationen erfinden oder halluzinieren!
- Wenn die Datenbank "keine Daten gefunden" meldet, sage das EHRLICH
- Gib NUR Informationen weiter, die DIREKT aus SeaTable kommen

WORKFLOW:
1. Bei Begrüßung: Freundlich antworten und nach dem Anliegen fragen
2. Bei konkreten Anfragen: Nutze die SeaTable-Funktionen
3. Gib präzise Auskünfte basierend auf den Datenbankdaten

DEINE AUFGABEN:
- Kundendaten aus SeaTable abfragen
- Fahrzeugdaten und Service-Historie anzeigen
- Reparaturstatus mitteilen
- Termine koordinieren
- Neue Termine in SeaTable eintragen

WICHTIGE REGELN:
- Immer auf Deutsch antworten
- Währungen als "250 Franken" aussprechen
- Keine technischen Details erwähnen
- NIEMALS Daten erfinden wenn keine gefunden wurden""")
        logger.info("✅ GarageAssistant with SeaTable initialized")

    @function_tool
    async def search_customer_data(self, 
                                  context: RunContext[GarageUserData],
                                  query: str) -> str:
        """
        Sucht nach Kundendaten in SeaTable.
        
        Args:
            query: Suchbegriff (Name, Telefonnummer oder E-Mail)
        """
        logger.info(f"🔍 Searching customer data in SeaTable for: {query}")
        
        # GUARD gegen Begrüßungen
        if len(query) < 3 or query.lower() in ["hallo", "guten tag", "hi", "hey", "servus", "grüezi"]:
            logger.warning(f"⚠️ Ignoring greeting search: {query}")
            return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder E-Mail-Adresse."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # SQL-Abfrage für Kundensuche
            sql_query = f"""
            SELECT Name, Email, Telefon, Kunde_seit, Notizen 
            FROM Kunden 
            WHERE Name LIKE '%{query}%' 
               OR Email LIKE '%{query}%' 
               OR Telefon LIKE '%{query}%'
            LIMIT 5
            """
            
            results = await client.query_table(sql_query)
            
            if results:
                logger.info(f"✅ Found {len(results)} customers in SeaTable")
                
                response_text = "Ich habe folgende Kundendaten gefunden:\n\n"
                for customer in results:
                    response_text += f"**{customer.get('Name', 'Unbekannt')}**\n"
                    response_text += f"- E-Mail: {customer.get('Email', '-')}\n"
                    response_text += f"- Telefon: {customer.get('Telefon', '-')}\n"
                    response_text += f"- Kunde seit: {customer.get('Kunde_seit', '-')}\n"
                    if customer.get('Notizen'):
                        response_text += f"- Notizen: {customer.get('Notizen')}\n"
                    response_text += "\n"
                
                # Speichere Kundennamen für weitere Abfragen
                if len(results) == 1:
                    context.userdata.authenticated_user = results[0].get('Name')
                
                return response_text
            else:
                return "Ich konnte keine Kundendaten zu Ihrer Anfrage finden. Bitte überprüfen Sie die Schreibweise oder geben Sie eine andere Information an."
                
        except Exception as e:
            logger.error(f"SeaTable customer search error: {e}")
            return "Es gab einen Fehler beim Zugriff auf die Kundendatenbank."

    @function_tool
    async def search_vehicle_data(self, 
                                 context: RunContext[GarageUserData],
                                 customer_name: Optional[str] = None) -> str:
        """
        Sucht nach Fahrzeugdaten eines Kunden in SeaTable.
        
        Args:
            customer_name: Name des Kunden (optional, nutzt authenticated_user wenn nicht angegeben)
        """
        search_name = customer_name or context.userdata.authenticated_user
        
        if not search_name:
            return "Bitte nennen Sie mir zuerst Ihren Namen, damit ich Ihre Fahrzeuge finden kann."
        
        logger.info(f"🚗 Searching vehicles for: {search_name}")
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # SQL-Abfrage für Fahrzeuge
            sql_query = f"""
            SELECT Kennzeichen, Marke, Modell, Baujahr, Kilometerstand, 
                   Kraftstoff, Letzte_Inspektion, VIN, Farbe
            FROM Fahrzeuge 
            WHERE Besitzer = '{search_name}'
            """
            
            vehicles = await client.query_table(sql_query)
            
            if vehicles:
                logger.info(f"✅ Found {len(vehicles)} vehicles")
                
                response_text = f"Fahrzeuge von {search_name}:\n\n"
                for vehicle in vehicles:
                    response_text += f"**{vehicle.get('Marke', '')} {vehicle.get('Modell', '')}**\n"
                    response_text += f"- Kennzeichen: {vehicle.get('Kennzeichen', '-')}\n"
                    response_text += f"- Baujahr: {vehicle.get('Baujahr', '-')}\n"
                    response_text += f"- Kilometerstand: {vehicle.get('Kilometerstand', '-')} km\n"
                    response_text += f"- Kraftstoff: {vehicle.get('Kraftstoff', '-')}\n"
                    response_text += f"- Letzte Inspektion: {self._format_date(vehicle.get('Letzte_Inspektion', '-'))}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return f"Ich konnte keine Fahrzeuge für {search_name} in der Datenbank finden."
                
        except Exception as e:
            logger.error(f"Vehicle search error: {e}")
            return "Es gab einen Fehler beim Abrufen der Fahrzeugdaten."

    @function_tool
    async def search_service_history(self, 
                                   context: RunContext[GarageUserData],
                                   kennzeichen: Optional[str] = None) -> str:
        """
        Sucht nach Service-Historie eines Fahrzeugs.
        
        Args:
            kennzeichen: Kennzeichen des Fahrzeugs
        """
        if not kennzeichen and not context.userdata.authenticated_user:
            return "Bitte nennen Sie mir ein Kennzeichen oder Ihren Namen."
        
        logger.info(f"🔧 Searching service history for: {kennzeichen or context.userdata.authenticated_user}")
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # SQL-Abfrage für Service-Historie
            if kennzeichen:
                sql_query = f"""
                SELECT Datum, Beschreibung, Kosten, Mechaniker, Status, Dauer_Stunden
                FROM Service 
                WHERE Fahrzeug_Kennzeichen = '{kennzeichen}'
                ORDER BY Datum DESC
                LIMIT 10
                """
            else:
                sql_query = f"""
                SELECT s.Datum, s.Fahrzeug_Kennzeichen, s.Beschreibung, 
                       s.Kosten, s.Mechaniker, s.Status
                FROM Service s
                WHERE s.Kunde = '{context.userdata.authenticated_user}'
                ORDER BY s.Datum DESC
                LIMIT 10
                """
            
            services = await client.query_table(sql_query)
            
            if services:
                logger.info(f"✅ Found {len(services)} service entries")
                
                response_text = "Service-Historie:\n\n"
                for service in services:
                    response_text += f"**{self._format_date(service.get('Datum', '-'))}**\n"
                    if not kennzeichen:
                        response_text += f"- Fahrzeug: {service.get('Fahrzeug_Kennzeichen', '-')}\n"
                    response_text += f"- Arbeiten: {service.get('Beschreibung', '-')}\n"
                    response_text += f"- Status: {service.get('Status', '-')}\n"
                    response_text += f"- Kosten: {self._format_currency(service.get('Kosten', 0))}\n"
                    response_text += f"- Mechaniker: {service.get('Mechaniker', '-')}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return "Ich konnte keine Service-Einträge in der Datenbank finden."
                
        except Exception as e:
            logger.error(f"Service history error: {e}")
            return "Es gab einen Fehler beim Abrufen der Service-Historie."

    @function_tool
    async def check_appointments(self,
                               context: RunContext[GarageUserData],
                               customer_name: Optional[str] = None) -> str:
        """
        Prüft anstehende Termine eines Kunden.
        
        Args:
            customer_name: Name des Kunden
        """
        search_name = customer_name or context.userdata.authenticated_user
        
        if not search_name:
            return "Bitte nennen Sie mir Ihren Namen, um Ihre Termine zu prüfen."
        
        logger.info(f"📅 Checking appointments for: {search_name}")
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Hole aktuelle und zukünftige Termine
            today = datetime.now().strftime('%Y-%m-%d')
            sql_query = f"""
            SELECT Datum, Uhrzeit, Fahrzeug, Service_Art, Mechaniker, 
                   Status, Geschätzte_Dauer, Notizen
            FROM Termine 
            WHERE Kunde = '{search_name}' 
              AND Datum >= '{today}'
              AND Status != 'Abgesagt'
            ORDER BY Datum, Uhrzeit
            """
            
            appointments = await client.query_table(sql_query)
            
            if appointments:
                response_text = f"Ihre Termine:\n\n"
                for apt in appointments:
                    response_text += f"**{self._format_date(apt.get('Datum', '-'))} um {apt.get('Uhrzeit', '-')} Uhr**\n"
                    response_text += f"- Fahrzeug: {apt.get('Fahrzeug', '-')}\n"
                    response_text += f"- Service: {apt.get('Service_Art', '-')}\n"
                    response_text += f"- Mechaniker: {apt.get('Mechaniker', '-')}\n"
                    response_text += f"- Status: {apt.get('Status', '-')}\n"
                    response_text += f"- Dauer: ca. {apt.get('Geschätzte_Dauer', '-')} Stunden\n"
                    if apt.get('Notizen'):
                        response_text += f"- Hinweis: {apt.get('Notizen')}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return f"Sie haben aktuell keine anstehenden Termine bei uns."
                
        except Exception as e:
            logger.error(f"Appointment check error: {e}")
            return "Es gab einen Fehler beim Abrufen der Termine."

    @function_tool
    async def create_appointment(self,
                               context: RunContext[GarageUserData],
                               datum: str,
                               uhrzeit: str,
                               service_art: str,
                               fahrzeug: str,
                               notizen: Optional[str] = None) -> str:
        """
        Erstellt einen neuen Termin in SeaTable.
        
        Args:
            datum: Datum im Format YYYY-MM-DD
            uhrzeit: Uhrzeit im Format HH:MM
            service_art: Art des Services
            fahrzeug: Kennzeichen des Fahrzeugs
            notizen: Optionale Notizen
        """
        if not context.userdata.authenticated_user:
            return "Bitte nennen Sie mir zuerst Ihren Namen."
        
        logger.info(f"📝 Creating appointment for {context.userdata.authenticated_user}")
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # Generiere Termin-ID
            termin_id = f"T-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Neuer Termin
            new_appointment = {
                "Termin_ID": termin_id,
                "Datum": datum,
                "Uhrzeit": uhrzeit,
                "Kunde": context.userdata.authenticated_user,
                "Fahrzeug": fahrzeug,
                "Service_Art": service_art,
                "Mechaniker": "Wird zugeteilt",
                "Status": "Angefragt",
                "Geschätzte_Dauer": 2.0,
                "Notizen": notizen or ""
            }
            
            success = await client.add_row("Termine", new_appointment)
            
            if success:
                return f"""Termin erfolgreich angelegt:
                
📅 Datum: {self._format_date(datum)}
⏰ Uhrzeit: {uhrzeit} Uhr
🚗 Fahrzeug: {fahrzeug}
🔧 Service: {service_art}
📋 Status: Angefragt

Wir werden uns bei Ihnen melden, um den Termin zu bestätigen."""
            else:
                return "Es gab einen Fehler beim Anlegen des Termins. Bitte rufen Sie uns an."
                
        except Exception as e:
            logger.error(f"Appointment creation error: {e}")
            return "Es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."

    def _format_date(self, date_str: str) -> str:
        """Formatiert Datum für deutsche Ausgabe"""
        if not date_str or date_str == '-':
            return '-'
        try:
            # Versuche verschiedene Formate
            for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%Y-%m-%dT%H:%M:%S']:
                try:
                    date_obj = datetime.strptime(date_str.split('T')[0], fmt)
                    return date_obj.strftime('%d.%m.%Y')
                except:
                    continue
            return date_str
        except:
            return date_str

    def _format_currency(self, amount: Any) -> str:
        """Formatiert Währungsbeträge"""
        try:
            if isinstance(amount, (int, float)):
                return f"{amount:.2f} Franken"
            elif isinstance(amount, str):
                # Entferne CHF und formatiere
                amount_clean = amount.replace('CHF', '').replace(',', '.').strip()
                amount_float = float(amount_clean)
                return f"{amount_float:.2f} Franken"
            else:
                return "0.00 Franken"
        except:
            return str(amount)


async def request_handler(ctx: JobContext):
    """Request handler ohne Hash-Assignment"""
    logger.info(f"[{AGENT_NAME}] 📨 Job request received")
    logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
    await ctx.accept()


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent mit SeaTable"""
    room_name = ctx.room.name if ctx.room else "unknown"
    session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
    
    logger.info("="*50)
    logger.info(f"🚗 Starting Garage Agent with SeaTable - Session: {session_id}")
    logger.info("="*50)
    
    session = None
    session_closed = False
    
    # SeaTable Konfiguration aus Umgebungsvariablen
    seatable_token = os.getenv("SEATABLE_BASE_TOKEN", "")
    seatable_url = os.getenv("SEATABLE_SERVER_URL", "https://cloud.seatable.io")
    
    if not seatable_token:
        logger.error("❌ SEATABLE_BASE_TOKEN not set in environment!")
        raise ValueError("SeaTable configuration missing")
    
    # Register disconnect handler
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
                
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
            await asyncio.sleep(1)
        
        # 4. Configure LLM
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.7
        )
        logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
        
        # 5. Create session with SeaTable config
        session = AgentSession[GarageUserData](
            userdata=GarageUserData(
                authenticated_user=None,
                seatable_base_token=seatable_token,
                seatable_server_url=seatable_url,
                seatable_access_token=None,
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
        logger.info(f"🏁 [{session_id}] Starting session with SeaTable...")
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        # 8. Initial greeting
        await asyncio.sleep(1.0)
        
        initial_instructions = """Sage NUR:
"Guten Tag und willkommen bei der Garage Müller! Ich bin Pia, Ihr digitaler Assistent. Wie kann ich Ihnen heute helfen?"

NICHTS ANDERES!"""
    
        logger.info(f"📢 [{session_id}] Generating initial greeting...")
        
        try:
            await session.generate_reply(
                instructions=initial_instructions,
                tool_choice="none"
            )
            logger.info(f"✅ [{session_id}] Initial greeting sent")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not send initial greeting: {e}")
        
        logger.info(f"✅ [{session_id}] Garage Agent with SeaTable ready!")
        
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
        
        # Disconnect from room if still connected
        try:
            if ctx.room and hasattr(ctx.room, 'connection_state') and ctx.room.connection_state == "connected":
                await ctx.room.disconnect()
                logger.info(f"✅ [{session_id}] Disconnected from room")
        except Exception as e:
            logger.warning(f"⚠️ [{session_id}] Error disconnecting from room: {e}")
        
        logger.info(f"✅ [{session_id}] Session cleanup complete")
        logger.info("="*50)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_handler=request_handler
    ))
