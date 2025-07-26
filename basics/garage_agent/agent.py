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
               headers={"Authorization": f"Token {self.base_token}"}
           )
           if response.status_code == 200:
               data = response.json()
               self.access_token = data["access_token"]
               self.dtable_uuid = data.get("dtable_uuid")
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
   
   async def add_row(self, table_name: str, row_data: Dict) -> bool:
       """Fügt eine neue Zeile zu einer Tabelle hinzu"""
       if not self.access_token:
           await self.get_access_token()
           
       async with httpx.AsyncClient() as client:
           try:
               response = await client.post(
                   f"{self.server_url}/api-gateway/api/v2/dtables/{self.dtable_uuid}/rows/",
                   headers={"Authorization": f"Bearer {self.access_token}"},
                   json={
                       "table_name": table_name,
                       "rows": [row_data]
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


class GarageAssistant(Agent):
   """Garage Assistant für SeaTable Integration"""
   
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
- Wenn keine Daten gefunden wurden, sage das EHRLICH
- Erfinde KEINE Daten, Termine oder Services die nicht existieren
- Bei "keine Daten gefunden" frage nach mehr Details

WORKFLOW:
1. Bei Begrüßung: Freundlich antworten und nach dem Anliegen fragen
2. Höre aufmerksam zu und identifiziere das konkrete Anliegen
3. NUR bei konkreten Anfragen: Nutze die passende Suchfunktion
4. Gib NUR präzise Auskünfte basierend auf den tatsächlichen Datenbankdaten

DEINE AUFGABEN:
- Kundendaten abfragen (Name, E-Mail, Telefon)
- Fahrzeugdaten anzeigen
- Service-Historie zeigen
- Termine erstellen

WICHTIGE REGELN:
- Immer auf Deutsch antworten
- Freundlich und professionell bleiben
- Währungen als "250 Franken" aussprechen
- Keine technischen Details der Datenbank erwähnen
- Bei Unklarheiten höflich nachfragen""")
       logger.info("✅ GarageAssistant initialized")

   @function_tool
   async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
       """
       Sucht nach Kundendaten in SeaTable.
       
       Args:
           query: Suchbegriff (Name, Telefonnummer oder E-Mail)
       """
       logger.info(f"🔍 Searching customer data for: {query}")
       
       if len(query) < 3:
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
               logger.info(f"✅ Found {len(results)} customers")
               
               response_text = "Ich habe folgende Kundendaten gefunden:\n\n"
               for customer in results:
                   response_text += f"**{customer.get('Name', 'Unbekannt')}**\n"
                   response_text += f"- E-Mail: {customer.get('Email', '-')}\n"
                   response_text += f"- Telefon: {customer.get('Telefon', '-')}\n"
                   response_text += f"- Kunde seit: {self._format_date(customer.get('Kunde_seit', '-'))}\n"
                   if customer.get('Notizen'):
                       response_text += f"- Notizen: {customer.get('Notizen')}\n"
                   response_text += "\n"
               
               # Speichere Kundennamen für weitere Abfragen
               if len(results) == 1:
                   context.userdata.authenticated_user = results[0].get('Name')
               
               return response_text
           else:
               return "Ich konnte keine Kundendaten zu Ihrer Anfrage finden."
               
       except Exception as e:
           logger.error(f"SeaTable customer search error: {e}")
           return "Es gab einen Fehler beim Zugriff auf die Kundendatenbank."

   @function_tool
   async def search_vehicle_data(self, 
                                context: RunContext[GarageUserData],
                                customer_name: Optional[str] = None) -> str:
       """
       Sucht nach Fahrzeugdaten eines Kunden.
       
       Args:
           customer_name: Name des Kunden (optional, nutzt gespeicherten Namen)
       """
       search_name = customer_name or context.userdata.authenticated_user
       
       if not search_name:
           return "Bitte nennen Sie mir zuerst Ihren Namen."
       
       logger.info(f"🚗 Searching vehicles for: {search_name}")
       
       try:
           client = SeaTableClient(
               context.userdata.seatable_server_url,
               context.userdata.seatable_base_token
           )
           
           sql_query = f"""
           SELECT Kennzeichen, Marke, Modell, Baujahr, Kilometerstand, 
                  Kraftstoff, Letzte_Inspektion
           FROM Fahrzeuge 
           WHERE Besitzer = '{search_name}'
           """
           
           vehicles = await client.query_table(sql_query)
           
           if vehicles:
               response_text = f"Fahrzeuge von {search_name}:\n\n"
               for vehicle in vehicles:
                   response_text += f"**{vehicle.get('Marke', '')} {vehicle.get('Modell', '')}**\n"
                   response_text += f"- Kennzeichen: {vehicle.get('Kennzeichen', '-')}\n"
                   response_text += f"- Baujahr: {vehicle.get('Baujahr', '-')}\n"
                   response_text += f"- Kilometerstand: {vehicle.get('Kilometerstand', '-')} km\n"
                   response_text += f"- Letzte Inspektion: {self._format_date(vehicle.get('Letzte_Inspektion', '-'))}\n"
                   response_text += "\n"
               
               return response_text
           else:
               return f"Ich konnte keine Fahrzeuge für {search_name} finden."
               
       except Exception as e:
           logger.error(f"Vehicle search error: {e}")
           return "Es gab einen Fehler beim Abrufen der Fahrzeugdaten."

   @function_tool
   async def search_service_history(self, 
                                  context: RunContext[GarageUserData],
                                  kennzeichen: Optional[str] = None) -> str:
       """
       Sucht nach Service-Historie.
       
       Args:
           kennzeichen: Fahrzeug-Kennzeichen (optional)
       """
       if not kennzeichen and not context.userdata.authenticated_user:
           return "Bitte nennen Sie mir ein Kennzeichen oder Ihren Namen."
       
       logger.info(f"🔧 Searching service history")
       
       try:
           client = SeaTableClient(
               context.userdata.seatable_server_url,
               context.userdata.seatable_base_token
           )
           
           if kennzeichen:
               sql_query = f"""
               SELECT Datum, Beschreibung, Kosten, Mechaniker, Status
               FROM Service 
               WHERE Fahrzeug_Kennzeichen = '{kennzeichen}'
               ORDER BY Datum DESC
               LIMIT 10
               """
           else:
               sql_query = f"""
               SELECT s.Datum, s.Fahrzeug_Kennzeichen, s.Beschreibung, 
                      s.Kosten, s.Status
               FROM Service s
               WHERE s.Kunde = '{context.userdata.authenticated_user}'
               ORDER BY s.Datum DESC
               LIMIT 10
               """
           
           services = await client.query_table(sql_query)
           
           if services:
               response_text = "Service-Historie:\n\n"
               for service in services:
                   response_text += f"**{self._format_date(service.get('Datum', '-'))}**\n"
                   response_text += f"- Arbeiten: {service.get('Beschreibung', '-')}\n"
                   response_text += f"- Status: {service.get('Status', '-')}\n"
                   response_text += f"- Kosten: {self._format_currency(service.get('Kosten', 0))}\n"
                   response_text += "\n"
               
               return response_text
           else:
               return "Keine Service-Einträge gefunden."
               
       except Exception as e:
           logger.error(f"Service history error: {e}")
           return "Fehler beim Abrufen der Service-Historie."

   @function_tool
   async def create_appointment(self, 
                              context: RunContext[GarageUserData],
                              datum: str,
                              uhrzeit: str,
                              service_art: str,
                              fahrzeug: str) -> str:
       """
       Erstellt einen neuen Termin.
       
       Args:
           datum: Datum des Termins (YYYY-MM-DD)
           uhrzeit: Uhrzeit des Termins
           service_art: Art des Services
           fahrzeug: Kennzeichen des Fahrzeugs
       """
       if not context.userdata.authenticated_user:
           return "Bitte nennen Sie mir zuerst Ihren Namen."
       
       logger.info(f"📝 Creating appointment")
       
       try:
           client = SeaTableClient(
               context.userdata.seatable_server_url,
               context.userdata.seatable_base_token
           )
           
           termin_id = f"T-{datetime.now().strftime('%Y%m%d%H%M%S')}"
           
           new_appointment = {
               "Termin_ID": termin_id,
               "Datum": datum,
               "Uhrzeit": uhrzeit,
               "Kunde": context.userdata.authenticated_user,
               "Fahrzeug": fahrzeug,
               "Service_Art": service_art,
               "Mechaniker": "Wird zugeteilt",
               "Status": "Angefragt",
               "Geschätzte_Dauer": 2.0
           }
           
           success = await client.add_row("Termine", new_appointment)
           
           if success:
               return f"Termin erfolgreich angelegt für {self._format_date(datum)} um {uhrzeit} Uhr."
           else:
               return "Fehler beim Anlegen des Termins."
               
       except Exception as e:
           logger.error(f"Appointment creation error: {e}")
           return "Technischer Fehler beim Terminanlegen."

   def _format_date(self, date_str: str) -> str:
       """Formatiert Datum für deutsche Ausgabe"""
       if not date_str or date_str == '-':
           return '-'
       try:
           date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
           return date_obj.strftime('%d.%m.%Y')
       except:
           return date_str

   def _format_currency(self, amount: Any) -> str:
       """Formatiert Währungsbeträge"""
       try:
           if isinstance(amount, (int, float)):
               return f"{amount:.2f} Franken"
           return str(amount)
       except:
           return str(amount)


async def request_handler(ctx: JobContext):
   """Request handler für Garage Agent"""
   logger.info(f"[{AGENT_NAME}] 📨 Job request received")
   logger.info(f"[{AGENT_NAME}] Room: {ctx.room.name}")
   await ctx.accept()


async def entrypoint(ctx: JobContext):
   """Entry point für den Garage Agent mit SeaTable"""
   room_name = ctx.room.name if ctx.room else "unknown"
   session_id = f"{room_name}_{int(asyncio.get_event_loop().time())}"
   
   logger.info("="*50)
   logger.info(f"🚗 Starting Garage Agent Session with SeaTable: {session_id}")
   logger.info("="*50)
   
   # SeaTable Konfiguration
   seatable_token = os.getenv("SEATABLE_BASE_TOKEN", "")
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
           model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
           temperature=0.3
       )
       logger.info(f"🤖 [{session_id}] Using LLM: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
       
       # 5. Create session with SeaTable context
       session = AgentSession[GarageUserData](
           userdata=GarageUserData(
               seatable_base_token=seatable_token,
               seatable_server_url=seatable_url,
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
- KEINE Tools verwenden

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
       
       logger.info(f"✅ [{session_id}] Garage Agent with SeaTable ready and listening!")
       
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
