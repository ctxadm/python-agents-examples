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
    
    async def query_table(self, sql_query: str) -> List[Dict]:
        """Execute a query and return results - NOW USING ROW API"""
        try:
            if not self.access_token:
                await self.get_access_token()
                
            # Parse SQL to extract table name and conditions
            import re
            
            # Extract table name
            table_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            if not table_match:
                logger.error(f"Could not extract table name from query: {sql_query}")
                return []
                
            table_name = table_match.group(1)
            
            # Get all rows from table using Row API
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
                
                # Simple WHERE clause filtering
                where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s*$)', sql_query, re.IGNORECASE)
                if where_match:
                    condition = where_match.group(1)
                    filtered_rows = []
                    
                    # Split multiple conditions by OR
                    or_conditions = re.split(r'\s+OR\s+', condition, flags=re.IGNORECASE)
                    
                    for row in rows:
                        for or_cond in or_conditions:
                            # Handle LIKE conditions
                            if 'LIKE' in or_cond.upper():
                                field_match = re.search(r'(\w+)\s+LIKE\s+[\'"]%(.+?)%[\'"]', or_cond, re.IGNORECASE)
                                if field_match:
                                    field, value = field_match.groups()
                                    if value.lower() in str(row.get(field, '')).lower():
                                        filtered_rows.append(row)
                                        break
                            
                            # Handle = conditions
                            elif '=' in or_cond:
                                field_match = re.search(r'(\w+)\s*=\s*[\'"](.+?)[\'"]', or_cond)
                                if field_match:
                                    field, value = field_match.groups()
                                    if str(row.get(field, '')).lower() == value.lower():
                                        filtered_rows.append(row)
                                        break
                    
                    rows = filtered_rows
                    logger.info(f"✅ Filtered to {len(rows)} rows matching conditions")
                
                # Handle ORDER BY
                order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(DESC|ASC))?', sql_query, re.IGNORECASE)
                if order_match:
                    field = order_match.group(1)
                    desc = order_match.group(2) and order_match.group(2).upper() == 'DESC'
                    rows.sort(key=lambda x: x.get(field, ''), reverse=desc)
                
                # Handle LIMIT
                limit_match = re.search(r'LIMIT\s+(\d+)', sql_query, re.IGNORECASE)
                if limit_match:
                    limit = int(limit_match.group(1))
                    rows = rows[:limit]
                    
                return rows
            else:
                logger.error(f"Failed to fetch rows: {response.status_code} - {response.text}")
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
            
            # SeaTable SQL Query - KORRIGIERT mit tatsächlichen Feldnamen
            sql_query = f"""
            SELECT Name, Email, Telefon, Kunde_seit, Notizen 
            FROM Kunden 
            WHERE Name LIKE '%{query}%' 
               OR Telefon LIKE '%{query}%'
               OR Email LIKE '%{query}%'
            LIMIT 5
            """
            
            results = await client.query_table(sql_query)
            
            if results:
                logger.info(f"✅ Found {len(results)} customer results")
                
                response_text = "Ich habe folgende Kundendaten gefunden:\n\n"
                for customer in results:
                    response_text += f"**{customer.get('Name', 'Unbekannt')}**\n"
                    if customer.get('Email'):
                        response_text += f"- Email: {customer.get('Email')}\n"
                    if customer.get('Telefon'):
                        response_text += f"- Telefon: {customer.get('Telefon')}\n"
                    if customer.get('Kunde_seit'):
                        response_text += f"- Kunde seit: {customer.get('Kunde_seit')}\n"
                    if customer.get('Notizen'):
                        response_text += f"- Notizen: {customer.get('Notizen')}\n"
                    response_text += "\n"
                
                # Speichere Kundennamen
                if len(results) == 1:
                    context.userdata.authenticated_user = results[0].get('Name')
                
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
        
        if len(query) < 3:
            return "Bitte geben Sie mir ein Kennzeichen oder einen Kundennamen."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            sql_query = f"""
            SELECT Kennzeichen, Marke, Modell, Baujahr, 
                   Besitzer, Kilometerstand, Letzte_Inspektion, Farbe, Kraftstoff
            FROM Fahrzeuge 
            WHERE Kennzeichen LIKE '%{query}%' 
               OR Besitzer LIKE '%{query}%'
            LIMIT 5
            """
            
            results = await client.query_table(sql_query)
            
            if results:
                logger.info(f"✅ Found {len(results)} vehicle results")
                
                response_text = "Hier sind die Fahrzeugdaten:\n\n"
                for vehicle in results:
                    response_text += f"**{vehicle.get('Kennzeichen', '-')}**\n"
                    response_text += f"- Besitzer: {vehicle.get('Besitzer', '-')}\n"
                    response_text += f"- Marke/Modell: {vehicle.get('Marke', '-')} {vehicle.get('Modell', '-')}\n"
                    response_text += f"- Baujahr: {vehicle.get('Baujahr', '-')}\n"
                    response_text += f"- Farbe: {vehicle.get('Farbe', '-')}\n"
                    response_text += f"- Kraftstoff: {vehicle.get('Kraftstoff', '-')}\n"
                    response_text += f"- Kilometerstand: {vehicle.get('Kilometerstand', '-')} km\n"
                    response_text += f"- Letzte Inspektion: {vehicle.get('Letzte_Inspektion', '-')}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return "Ich konnte keine Fahrzeugdaten zu Ihrer Anfrage finden. Bitte geben Sie das vollständige Kennzeichen an."
                
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
        
        if len(query) < 3:
            return "Bitte geben Sie mir einen Namen, ein Kennzeichen oder ein Datum."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            # KORRIGIERT: Service statt Reparaturen, richtige Feldnamen
            sql_query = f"""
            SELECT Datum, Kunde, Fahrzeug_Kennzeichen, Status, 
                   Beschreibung, Kosten, Mechaniker, Dauer_Stunden
            FROM Service 
            WHERE Kunde LIKE '%{query}%' 
               OR Fahrzeug_Kennzeichen LIKE '%{query}%'
               OR Datum LIKE '%{query}%'
            ORDER BY Datum DESC
            LIMIT 5
            """
            
            results = await client.query_table(sql_query)
            
            if results:
                logger.info(f"✅ Found {len(results)} service results")
                
                response_text = "Hier ist die Service-Historie:\n\n"
                for service in results[:3]:
                    response_text += f"**Service vom {service.get('Datum', '-')}**\n"
                    response_text += f"- Fahrzeug: {service.get('Fahrzeug_Kennzeichen', '-')}\n"
                    response_text += f"- Beschreibung: {service.get('Beschreibung', '-')}\n"
                    response_text += f"- Status: {service.get('Status', '-')}\n"
                    response_text += f"- Mechaniker: {service.get('Mechaniker', '-')}\n"
                    if service.get('Kosten'):
                        response_text += f"- Kosten: {service.get('Kosten')} Franken\n"
                    if service.get('Dauer_Stunden'):
                        response_text += f"- Dauer: {service.get('Dauer_Stunden')} Stunden\n"
                    response_text += "\n"
                
                return response_text
            else:
                return "Ich konnte keine Service-Einträge zu Ihrer Anfrage finden. Bitte geben Sie ein Kennzeichen oder einen Kundennamen an."
                
        except Exception as e:
            logger.error(f"Service search error: {e}")
            return "Die Service-Datenbank ist momentan nicht verfügbar."

    @function_tool
    async def search_appointments(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
        """
        Sucht nach Terminen in der Datenbank.
        
        Args:
            query: Kundenname, Kennzeichen oder Datum
        """
        logger.info(f"📅 Searching appointments for: {query}")
        
        if len(query) < 3:
            return "Bitte geben Sie mir einen Namen, ein Kennzeichen oder ein Datum."
        
        try:
            client = SeaTableClient(
                context.userdata.seatable_server_url,
                context.userdata.seatable_base_token
            )
            
            sql_query = f"""
            SELECT Termin_ID, Datum, Uhrzeit, Kunde, Fahrzeug, 
                   Service_Art, Mechaniker, Status, Geschätzte_Dauer
            FROM Termine 
            WHERE Kunde LIKE '%{query}%' 
               OR Fahrzeug LIKE '%{query}%'
               OR Datum LIKE '%{query}%'
            ORDER BY Datum DESC
            LIMIT 5
            """
            
            results = await client.query_table(sql_query)
            
            if results:
                logger.info(f"✅ Found {len(results)} appointment results")
                
                response_text = "Hier sind die Termine:\n\n"
                for termin in results[:3]:
                    response_text += f"**Termin am {termin.get('Datum', '-')} um {termin.get('Uhrzeit', '-')} Uhr**\n"
                    response_text += f"- Kunde: {termin.get('Kunde', '-')}\n"
                    response_text += f"- Fahrzeug: {termin.get('Fahrzeug', '-')}\n"
                    response_text += f"- Service: {termin.get('Service_Art', '-')}\n"
                    response_text += f"- Mechaniker: {termin.get('Mechaniker', '-')}\n"
                    response_text += f"- Status: {termin.get('Status', '-')}\n"
                    if termin.get('Geschätzte_Dauer'):
                        response_text += f"- Geschätzte Dauer: {termin.get('Geschätzte_Dauer')}\n"
                    response_text += "\n"
                
                return response_text
            else:
                return "Ich konnte keine Termine zu Ihrer Anfrage finden."
                
        except Exception as e:
            logger.error(f"Appointment search error: {e}")
            return "Die Termin-Datenbank ist momentan nicht verfügbar."


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
                    break
            
            if audio_track_received:
                break
                
            logger.info(f"⏳ [{session_id}] Waiting for audio track... ({i+1}/{max_wait_time})")
            await asyncio.sleep(1)
        
        if not audio_track_received:
            logger.error(f"❌ [{session_id}] No audio track received after {max_wait_time}s!")
        
        # 4. Configure LLM
        llm = openai.LLM(
            model="llama3.2:latest",
            base_url=os.getenv("OLLAMA_URL", "http://172.16.0.146:11434/v1"),
            api_key="ollama",
            temperature=0.3
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
