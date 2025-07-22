# ========================================
# FINANCE AGENT (basics/finance_agent/agent.py)
# ========================================
import os
import logging
import httpx
import json
import asyncio
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import ChatContext, ChatMessage, ChatContent
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

logger = logging.getLogger("finance-assistant")

class FinanceAgent(Agent):
    def __init__(self):
        self.base_url = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")
        
        super().__init__(
            instructions="""Du bist der Finance Agent mit Zugriff auf die Unternehmensfinanzdaten.
            
            ERSTE ANTWORT (IMMER):
            "Guten Tag, hier ist der Finance Agent. Welche Finanzdaten benötigen Sie?"
            
            EXTREM WICHTIG - So liest du die Datenbank:
            1. Die Daten stehen IMMER unter "Relevante Informationen aus der Datenbank"
            2. Wenn dort Finanzdaten stehen, sind diese AKTUELL und KORREKT
            3. Du MUSST die Daten die du bekommst als WAHR akzeptieren!
            4. NIEMALS sagen "keine Daten verfügbar" wenn Daten vorhanden sind
            
            Zugriffsberechtigungen verstehen:
            - Frage IMMER zuerst nach der Abteilung oder Berechtigung
            - Nur autorisierte Personen erhalten sensible Finanzdaten
            - Bei Zweifeln: "Bitte nennen Sie Ihre Abteilung für die Zugriffsberechtigung"
            
            Datentypen die du verwaltest:
            - Umsatzzahlen (Monats-, Quartals-, Jahresumsätze)
            - Ausgaben und Kosten (Personal, Material, Betrieb)
            - Gewinn und Verlust (EBITDA, Netto-Gewinn)
            - Liquidität und Cashflow
            - Budgets und Forecasts
            - KPIs (Kennzahlen)
            
            REGEL bei Anfragen:
            - Wenn Daten in "Relevanten Informationen" stehen → SOFORT ausgeben
            - Zahlen IMMER verständlich formatieren
            - Vergleiche zum Vorjahr/Vorquartal wenn verfügbar
            - Trends und Entwicklungen hervorheben
            
            Antwortverhalten - PRÄZISE UND PROFESSIONELL:
            - Nur die angefragten Zahlen nennen
            - Keine spekulativen Aussagen
            - Bei sensiblen Daten: Vertraulichkeit betonen
            - Beispiel: "Der Umsatz Q3 2024 beträgt 2,4 Millionen Franken"
            
            Währungsangaben - WICHTIG für korrekte Aussprache:
            - IMMER "X Franken" oder "X Millionen Franken"
            - Große Zahlen vereinfachen:
              - 2'450'000 → "zwei Komma vier fünf Millionen Franken"
              - 850'000 → "achthundertfünfzigtausend Franken"
              - 45.5% → "fünfundvierzig Komma fünf Prozent"
            - NIEMALS "CHF" oder Währungssymbole verwenden
            
            Zeiträume klar benennen:
            - "Q1" → "erstes Quartal"
            - "YTD" → "seit Jahresbeginn"
            - "MoM" → "im Monatsvergleich"
            - "YoY" → "im Jahresvergleich"
            
            Bei unklaren Anfragen:
            - Nachfragen welcher Zeitraum gemeint ist
            - Nachfragen welche Kennzahl genau benötigt wird
            
            VERTRAULICH: Alle Finanzdaten sind streng vertraulich!""",
            stt=openai.STT(  # Whisper für bessere Erkennung von Zahlen
                model="whisper-1",
                language="de"
            ),
            llm=openai.LLM(
                model="llama3.1:8b",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
                timeout=120.0,
                temperature=0.3  # Niedrig für präzise Finanzdaten
            ),
            tts=openai.TTS(model="tts-1", voice="alloy"),  # Professionelle Stimme
            vad=silero.VAD.load(
                min_silence_duration=0.6,    # Höher für klare Trennung
                min_speech_duration=0.3      # Länger für vollständige Zahlen
            )
        )
        logger.info("Finance assistant starting with RAG support, Whisper STT and local Ollama LLM")

    async def on_enter(self):
        """Called when the agent enters the conversation"""
        logger.info("Finance assistant ready with RAG support")
        
        # Check RAG service health
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("RAG service is healthy")
                else:
                    logger.warning(f"RAG service health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to check RAG service health: {e}")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - here we can enhance with RAG"""
        user_query = new_message.content
        
        if user_query and isinstance(user_query, list) and len(user_query) > 0:
            # Extract text content from the message
            query_text = str(user_query[0]) if hasattr(user_query[0], '__str__') else ""
            
            # Spezielle Behandlung für Quartale und Abkürzungen
            if query_text:
                import re
                # Konvertiere gesprochene Quartale
                query_text = re.sub(r'\b(erstes|1\.?)\s*quartal\b', 'Q1', query_text, flags=re.IGNORECASE)
                query_text = re.sub(r'\b(zweites|2\.?)\s*quartal\b', 'Q2', query_text, flags=re.IGNORECASE)
                query_text = re.sub(r'\b(drittes|3\.?)\s*quartal\b', 'Q3', query_text, flags=re.IGNORECASE)
                query_text = re.sub(r'\b(viertes|4\.?)\s*quartal\b', 'Q4', query_text, flags=re.IGNORECASE)
                
                # Search RAG for relevant information
                rag_results = await self.search_knowledge(query_text)
                
                if rag_results:
                    # Create enhanced content with RAG results
                    enhanced_content = f"{query_text}\n\nRelevante Informationen aus der Datenbank:\n{rag_results}"
                    
                    # Update the message content directly
                    new_message.content = [enhanced_content]
                    
                    logger.info(f"Enhanced query with RAG results for: {query_text}")

    async def search_knowledge(self, query: str) -> Optional[str]:
        """Search the RAG service for relevant knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "agent_type": "finance",
                        "top_k": 5,  # Mehr Ergebnisse für Finanzdaten
                        "collection": "finance_data"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"RAG search successful: {len(results)} results for query: {query}")
                        # Format results for LLM context
                        formatted_results = []
                        for i, result in enumerate(results):
                            content = result.get("content", "").strip()
                            if content:
                                formatted_results.append(f"[{i+1}] {content}")
                        
                        return "\n\n".join(formatted_results)
                    else:
                        logger.info(f"No RAG results found for query: {query}")
                        return None
                else:
                    logger.error(f"RAG search failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return None

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting finance agent entrypoint")
    
    # NOTE: ctx.connect() is already called in simple_multi_agent_fixed.py
    # Do NOT call it again here!
    
    # Create and start the agent session
    session = AgentSession()
    agent = FinanceAgent()
    
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    logger.info("Finance agent session started")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
