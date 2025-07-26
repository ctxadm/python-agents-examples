# LiveKit Agents - Garage Management Agent
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
   rag_url: str = "http://localhost:8000"
   current_customer_id: Optional[str] = None
   active_repair_id: Optional[str] = None
   user_language: str = "de"


class GarageAssistant(Agent):
   """Garage Assistant für Kundenverwaltung und Reparaturen"""
   
   def __init__(self) -> None:
       # Instructions VERBESSERT - Klare Anweisungen gegen Halluzinationen
       super().__init__(instructions="""Du bist Pia, der digitale Assistent der Garage Müller.

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
       logger.info("✅ GarageAssistant initialized")

   @function_tool
   async def search_customer_data(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
       """
       Sucht nach Kundendaten in der Garage-Datenbank.
       
       Args:
           query: Suchbegriff (Name, Telefonnummer oder Autonummer)
       """
       logger.info(f"🔍 Searching customer data for: {query}")
       
       # GUARD gegen falsche Suchen bei Begrüßungen
       if len(query) < 5 or query.lower() in ["hallo", "guten tag", "hi", "hey", "servus", "grüezi"]:
           logger.warning(f"⚠️ Ignoring greeting search: {query}")
           return "Bitte geben Sie mir Ihren Namen, Ihre Telefonnummer oder Ihre Autonummer, damit ich Ihre Kundendaten finden kann."
       
       # GUARD gegen halluzinierte Anfragen
       if "möchte gerne meine kundendaten" in query.lower() and len(query) > 30:
           logger.warning(f"⚠️ Detected hallucinated query: {query}")
           return "Bitte nennen Sie mir Ihren Namen oder Ihre Autonummer, damit ich Ihre Daten suchen kann."
       
       try:
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{context.userdata.rag_url}/search",
                   json={
                       "query": query,
                       "agent_type": "garage",
                       "top_k": 3,
                       "collection": "garage_management"
                   }
               )
               
               if response.status_code == 200:
                   results = response.json().get("results", [])
                   
                   if results:
                       logger.info(f"✅ Found {len(results)} customer results")
                       
                       # Prüfe ob die Ergebnisse relevant sind
                       relevant_results = []
                       for result in results[:3]:
                           content = result.get("content", "").strip()
                           # Filter irrelevante Ergebnisse (z.B. Funkfrequenzen)
                           if content and not any(word in content.lower() for word in ["mhz", "bakom", "funk", "frequenz"]):
                               content = self._format_garage_data(content)
                               relevant_results.append(content)
                       
                       if relevant_results:
                           response_text = "\n\n".join(relevant_results)
                           return response_text
                       else:
                           logger.warning("⚠️ Only irrelevant results found")
                           return "Ich konnte keine relevanten Kundendaten zu Ihrer Anfrage finden. Bitte geben Sie mir mehr Informationen wie Ihre Autonummer oder Telefonnummer."
                   
                   return "Ich konnte keine Kundendaten zu Ihrer Anfrage finden. Können Sie mir bitte Ihre Autonummer oder Telefonnummer nennen?"
                   
               else:
                   logger.error(f"Customer search failed: {response.status_code}")
                   return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."
                   
       except Exception as e:
           logger.error(f"Customer search error: {e}")
           return "Die Kundendatenbank ist momentan nicht erreichbar."

   @function_tool
   async def search_repair_status(self, 
                                 context: RunContext[GarageUserData],
                                 query: str) -> str:
       """
       Sucht nach Reparaturstatus und Aufträgen.
       
       Args:
           query: Kundenname, Autonummer oder Auftragsnummer
       """
       logger.info(f"🔧 Searching repair status for: {query}")
       
       # GUARD gegen zu kurze Anfragen
       if len(query) < 3:
           return "Bitte geben Sie mir einen Namen, eine Autonummer oder Auftragsnummer."
       
       try:
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{context.userdata.rag_url}/search",
                   json={
                       "query": f"Reparatur Status Service {query}",
                       "agent_type": "garage",
                       "top_k": 5,
                       "collection": "garage_management"
                   }
               )
               
               if response.status_code == 200:
                   results = response.json().get("results", [])
                   
                   if results:
                       logger.info(f"✅ Found {len(results)} repair results")
                       
                       # DEBUG: Log ALL results before filtering
                       for i, result in enumerate(results):
                           content = result.get("content", "").strip()
                           logger.info(f"RAG Result {i}: {content[:150]}...")  # First 150 chars
                       
                       # Sammle ALLE Ergebnisse, lockerer Filter
                       all_results = []
                       for result in results:
                           content = result.get("content", "").strip()
                           # Viel lockererer Filter - zeige fast alles außer offensichtlichen Fehlern
                           if content and not any(word in content.lower() for word in ["mhz", "bakom", "funk", "frequenz"]):
                               content = self._format_garage_data(content)
                               all_results.append(content)
                       
                       if all_results:
                           # Zeige ALLE gefundenen Daten
                           response_text = f"Ich habe {len(all_results)} Einträge in der Datenbank gefunden:\n\n"
                           response_text += "\n\n".join(all_results[:3])  # Max 3 Einträge
                           
                           # Spezifischer Hinweis wenn nichts passt
                           if not any(word in response_text.lower() for word in ["reparatur", "status", "service", "wartung", query.lower()]):
                               response_text += "\n\nHINWEIS: Diese Einträge scheinen nicht direkt zu Ihrer Anfrage zu passen. Können Sie mir bitte die Autonummer oder Auftragsnummer nennen?"
                           
                           return response_text
                       else:
                           # Wenn wirklich NICHTS gefunden wurde
                           return f"Ich konnte keine Daten zu '{query}' in unserer Datenbank finden. Bitte geben Sie mir die genaue Autonummer (z.B. ZH 123456) oder die Auftragsnummer."
                   
                   return "Ich konnte keine Reparatur- oder Servicedaten zu Ihrer Anfrage finden. Können Sie mir bitte die Autonummer oder Auftragsnummer nennen?"
                   
               else:
                   logger.error(f"Repair search failed: {response.status_code}")
                   return "Die Reparaturdatenbank ist momentan nicht verfügbar."
                   
       except Exception as e:
           logger.error(f"Repair search error: {e}")
           return "Es gab einen Fehler beim Abrufen der Reparaturdaten."

   @function_tool
   async def search_invoice_data(self, 
                                context: RunContext[GarageUserData],
                                query: str) -> str:
       """
       Sucht nach Rechnungsinformationen.
       
       Args:
           query: Kundenname, Rechnungsnummer oder Datum
       """
       logger.info(f"💰 Searching invoice data for: {query}")
       
       # GUARD gegen zu kurze Anfragen
       if len(query) < 3:
           return "Bitte geben Sie mir einen Kundennamen, eine Rechnungsnummer oder ein Datum."
       
       try:
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{context.userdata.rag_url}/search",
                   json={
                       "query": f"Rechnung Kosten {query}",
                       "agent_type": "garage",
                       "top_k": 3,
                       "collection": "garage_management"
                   }
               )
               
               if response.status_code == 200:
                   results = response.json().get("results", [])
                   
                   if results:
                       logger.info(f"✅ Found {len(results)} invoice results")
                       
                       # DEBUG: Log results
                       for i, result in enumerate(results):
                           content = result.get("content", "").strip()
                           logger.info(f"Invoice Result {i}: {content[:150]}...")
                       
                       # Lockerer Filter
                       all_invoices = []
                       for result in results:
                           content = result.get("content", "").strip()
                           # Zeige fast alles was kein offensichtlicher Fehler ist
                           if content and not any(word in content.lower() for word in ["mhz", "bakom", "funk"]):
                               content = self._format_garage_data(content)
                               all_invoices.append(content)
                       
                       if all_invoices:
                           response_text = f"Ich habe {len(all_invoices)} Rechnungseinträge gefunden:\n\n"
                           response_text += "\n\n".join(all_invoices[:2])
                           return response_text
                       else:
                           return "Ich konnte keine Rechnungsdaten zu Ihrer Anfrage finden. Bitte nennen Sie mir die Rechnungsnummer oder das genaue Datum."
                   
                   return "Ich konnte keine Rechnungsdaten zu Ihrer Anfrage finden. Haben Sie eine Rechnungsnummer?"
                   
               else:
                   logger.error(f"Invoice search failed: {response.status_code}")
                   return "Die Rechnungsdatenbank ist momentan nicht verfügbar."
                   
       except Exception as e:
           logger.error(f"Invoice search error: {e}")
           return "Es gab einen Fehler beim Abrufen der Rechnungsdaten."

   def _format_garage_data(self, content: str) -> str:
       """Formatiert Garagendaten für bessere Lesbarkeit"""
       # Ersetze Unterstriche
       content = content.replace('_', ' ')
       
       # Formatiere Währungen für Sprachausgabe
       content = re.sub(r'CHF\s*(\d+)\.(\d{2})', r'\1 Franken \2', content)
       content = re.sub(r'(\d+)\.(\d{2})\s*CHF', r'\1 Franken \2', content)
       
       # Formatiere Datum
       content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3.\2.\1', content)
       
       return content


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
           temperature=0.3
       )
       logger.info(f"🤖 [{session_id}] Using Llama 3.2 via Ollama")
       
       # 5. Create session
       session = AgentSession[GarageUserData](
           userdata=GarageUserData(
               authenticated_user=None,
               rag_url=rag_url,
               current_customer_id=None,
               active_repair_id=None,
               user_language="de"
           ),
           llm=llm,
           vad=silero.VAD.load(
               min_silence_duration=0.6,  # Etwas schneller als Medical
               min_speech_duration=0.2
           ),
           stt=openai.STT(
               model="whisper-1",
               language="de"
           ),
           tts=openai.TTS(
               model="tts-1",
               voice="nova"  # Freundliche Stimme für Kundenkontakt
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
       
       # Event handlers (simplified)
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
       
       initial_instructions = """WICHTIG: Nutze KEINE Suchfunktionen für diese Begrüßung!

- Dies ist eine NEUE Konversation
- Ignoriere alle vorherigen Nachrichten
- Nutze KEINE Suchfunktionen für diese Begrüßung!
- Entschuldige dich NICHT für vorherige Fehler

Begrüße den Kunden freundlich mit: 
'Guten Tag und willkommen bei der Garage Müller! Ich bin Pia, Ihr digitaler Assistent. 
Wie kann ich Ihnen heute helfen? Möchten Sie den Status einer Reparatur erfahren, 
Kundendaten abfragen oder haben Sie Fragen zu einer Rechnung?'

KEINE TOOLS VERWENDEN! KEINE ENTSCHULDIGUNGEN!"""
       
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
