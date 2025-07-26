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
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero

load_dotenv()

# Logging
logger = logging.getLogger("garage-agent")
logger.setLevel(logging.INFO)

# Agent Name für Multi-Worker Setup
AGENT_NAME = os.getenv("AGENT_NAME", "agent-garage-1")

@dataclass
class AssistantContext:
    """Context für den Garage Assistant"""
    authenticated_user: Optional[str] = None
    seatable_base_token: str = ""
    seatable_server_url: str = ""
    seatable_access_token: Optional[str] = None
    current_customer_id: Optional[str] = None
    active_repair_id: Optional[str] = None


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


# Function Tools für den LLM
async def search_customer_data(
    query: str,
    context: AssistantContext
) -> str:
    """Sucht nach Kundendaten in SeaTable."""
    logger.info(f"🔍 Searching customer data for: {query}")
    
    if len(query) < 3:
        return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder E-Mail-Adresse."
    
    try:
        client = SeaTableClient(
            context.seatable_server_url,
            context.seatable_base_token
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
                response_text += f"- Kunde seit: {format_date(customer.get('Kunde_seit', '-'))}\n"
                if customer.get('Notizen'):
                    response_text += f"- Notizen: {customer.get('Notizen')}\n"
                response_text += "\n"
            
            # Speichere Kundennamen für weitere Abfragen
            if len(results) == 1:
                context.authenticated_user = results[0].get('Name')
            
            return response_text
        else:
            return "Ich konnte keine Kundendaten zu Ihrer Anfrage finden."
            
    except Exception as e:
        logger.error(f"SeaTable customer search error: {e}")
        return "Es gab einen Fehler beim Zugriff auf die Kundendatenbank."


async def search_vehicle_data(
    customer_name: Optional[str],
    context: AssistantContext
) -> str:
    """Sucht nach Fahrzeugdaten eines Kunden."""
    search_name = customer_name or context.authenticated_user
    
    if not search_name:
        return "Bitte nennen Sie mir zuerst Ihren Namen."
    
    logger.info(f"🚗 Searching vehicles for: {search_name}")
    
    try:
        client = SeaTableClient(
            context.seatable_server_url,
            context.seatable_base_token
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
                response_text += f"- Letzte Inspektion: {format_date(vehicle.get('Letzte_Inspektion', '-'))}\n"
                response_text += "\n"
            
            return response_text
        else:
            return f"Ich konnte keine Fahrzeuge für {search_name} finden."
            
    except Exception as e:
        logger.error(f"Vehicle search error: {e}")
        return "Es gab einen Fehler beim Abrufen der Fahrzeugdaten."


async def search_service_history(
    kennzeichen: Optional[str],
    context: AssistantContext
) -> str:
    """Sucht nach Service-Historie."""
    if not kennzeichen and not context.authenticated_user:
        return "Bitte nennen Sie mir ein Kennzeichen oder Ihren Namen."
    
    logger.info(f"🔧 Searching service history")
    
    try:
        client = SeaTableClient(
            context.seatable_server_url,
            context.seatable_base_token
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
            WHERE s.Kunde = '{context.authenticated_user}'
            ORDER BY s.Datum DESC
            LIMIT 10
            """
        
        services = await client.query_table(sql_query)
        
        if services:
            response_text = "Service-Historie:\n\n"
            for service in services:
                response_text += f"**{format_date(service.get('Datum', '-'))}**\n"
                response_text += f"- Arbeiten: {service.get('Beschreibung', '-')}\n"
                response_text += f"- Status: {service.get('Status', '-')}\n"
                response_text += f"- Kosten: {format_currency(service.get('Kosten', 0))}\n"
                response_text += "\n"
            
            return response_text
        else:
            return "Keine Service-Einträge gefunden."
            
    except Exception as e:
        logger.error(f"Service history error: {e}")
        return "Fehler beim Abrufen der Service-Historie."


async def create_appointment(
    datum: str,
    uhrzeit: str,
    service_art: str,
    fahrzeug: str,
    context: AssistantContext
) -> str:
    """Erstellt einen neuen Termin."""
    if not context.authenticated_user:
        return "Bitte nennen Sie mir zuerst Ihren Namen."
    
    logger.info(f"📝 Creating appointment")
    
    try:
        client = SeaTableClient(
            context.seatable_server_url,
            context.seatable_base_token
        )
        
        termin_id = f"T-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        new_appointment = {
            "Termin_ID": termin_id,
            "Datum": datum,
            "Uhrzeit": uhrzeit,
            "Kunde": context.authenticated_user,
            "Fahrzeug": fahrzeug,
            "Service_Art": service_art,
            "Mechaniker": "Wird zugeteilt",
            "Status": "Angefragt",
            "Geschätzte_Dauer": 2.0
        }
        
        success = await client.add_row("Termine", new_appointment)
        
        if success:
            return f"Termin erfolgreich angelegt für {format_date(datum)} um {uhrzeit} Uhr."
        else:
            return "Fehler beim Anlegen des Termins."
            
    except Exception as e:
        logger.error(f"Appointment creation error: {e}")
        return "Technischer Fehler beim Terminanlegen."


# Hilfsfunktionen
def format_date(date_str: str) -> str:
    """Formatiert Datum für deutsche Ausgabe"""
    if not date_str or date_str == '-':
        return '-'
    try:
        date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
        return date_obj.strftime('%d.%m.%Y')
    except:
        return date_str


def format_currency(amount: Any) -> str:
    """Formatiert Währungsbeträge"""
    try:
        if isinstance(amount, (int, float)):
            return f"{amount:.2f} Franken"
        return str(amount)
    except:
        return str(amount)


async def entrypoint(ctx: JobContext):
    """Entry point für den Garage Agent"""
    logger.info("="*50)
    logger.info("🚗 Starting Garage Agent with SeaTable")
    logger.info("="*50)
    
    # SeaTable Konfiguration
    seatable_token = os.getenv("SEATABLE_BASE_TOKEN", "")
    seatable_url = os.getenv("SEATABLE_SERVER_URL", "https://cloud.seatable.io")
    
    if not seatable_token:
        logger.error("❌ SEATABLE_BASE_TOKEN not set!")
        raise ValueError("SeaTable configuration missing")
    
    # Context erstellen
    assistant_context = AssistantContext(
        seatable_base_token=seatable_token,
        seatable_server_url=seatable_url
    )
    
    # Warte auf Verbindung
    await ctx.connect()
    logger.info("✅ Connected to LiveKit room")
    
    # LLM Setup
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""Du bist Pia, der digitale Assistent der Garage Müller.

WICHTIGE REGELN:
- Antworte IMMER auf Deutsch
- Bei Begrüßungen: Freundlich antworten und nach dem Anliegen fragen
- Nutze die Funktionen um auf SeaTable-Daten zuzugreifen
- Erfinde NIEMALS Daten - sage ehrlich wenn nichts gefunden wurde
- Währungen als "X Franken" aussprechen

DEINE AUFGABEN:
- Kundendaten abfragen
- Fahrzeugdaten anzeigen
- Service-Historie zeigen
- Termine erstellen

Begrüße den Kunden freundlich und frage nach seinem Anliegen."""
    )
    
    # Voice Assistant erstellen
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(language="de"),
        llm=openai.LLM(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.7
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="nova"
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=assistant_context,
    )
    
    # Funktionen registrieren
    assistant.llm.add_function(search_customer_data)
    assistant.llm.add_function(search_vehicle_data)
    assistant.llm.add_function(search_service_history)
    assistant.llm.add_function(create_appointment)
    
    # Session starten
    assistant.start(ctx.room)
    
    # Initiale Begrüßung
    await asyncio.sleep(1)
    await assistant.say(
        "Guten Tag und willkommen bei der Garage Müller! Ich bin Pia, Ihr digitaler Assistent. Wie kann ich Ihnen heute helfen?",
        allow_interruptions=True
    )
    
    logger.info("✅ Garage Agent ready!")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
