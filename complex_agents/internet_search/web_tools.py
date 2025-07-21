import asyncio
import httpx
import logging
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from readability import Document
from typing import List, Dict, Optional
from livekit.agents.llm import function_tool

logger = logging.getLogger(__name__)

class WebSearchTools:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; LiveKit-Agent/1.0)'},
            follow_redirects=True
        )
    
    @function_tool
    async def search_web(self, query: str, max_results: int = 3) -> str:
        """Sucht im Internet nach aktuellen Informationen zu einer Frage oder einem Thema"""
        try:
            logger.info(f"Suche im Web nach: {query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return "Keine Suchergebnisse gefunden."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. **{result['title']}**\n"
                    f"   {result['body'][:200]}{'...' if len(result['body']) > 200 else ''}\n"
                    f"   Quelle: {result['href']}\n"
                )
            
            search_summary = f"Suchergebnisse für '{query}':\n\n" + "\n".join(formatted_results)
            logger.info(f"Web-Suche erfolgreich: {len(results)} Ergebnisse gefunden")
            return search_summary
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Fehler bei der Websuche: {str(e)}"
    
    @function_tool
    async def fetch_webpage(self, url: str) -> str:
        """Lädt den vollständigen Inhalt einer Webseite und extrahiert den Haupttext"""
        try:
            logger.info(f"Lade Webseite: {url}")
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Content mit Readability extrahieren
            doc = Document(response.text)
            title = doc.title()
            content = BeautifulSoup(doc.summary(), 'html.parser').get_text()
            
            # Text kürzen für LLM (aber großzügiger als vorher)
            if len(content) > 3000:
                content = content[:3000] + "...\n[Inhalt gekürzt]"
            
            result = f"**{title}**\n\n{content}\n\nQuelle: {url}"
            logger.info(f"Webseite erfolgreich geladen: {len(content)} Zeichen")
            return result
            
        except Exception as e:
            logger.error(f"Webpage fetch error: {e}")
            return f"Fehler beim Laden der Webseite {url}: {str(e)}"
    
    @function_tool
    async def search_news(self, topic: str, max_results: int = 3) -> str:
        """Sucht nach aktuellen Nachrichten und News zu einem bestimmten Thema"""
        try:
            logger.info(f"Suche News zu: {topic}")
            
            with DDGS() as ddgs:
                results = list(ddgs.news(topic, max_results=max_results))
            
            if not results:
                return f"Keine aktuellen Nachrichten zu '{topic}' gefunden."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. **{result['title']}**\n"
                    f"   {result['body'][:250]}{'...' if len(result['body']) > 250 else ''}\n"
                    f"   Datum: {result.get('date', 'Unbekannt')}\n"
                    f"   Quelle: {result['url']}\n"
                )
            
            news_summary = f"Aktuelle Nachrichten zu '{topic}':\n\n" + "\n".join(formatted_results)
            logger.info(f"News-Suche erfolgreich: {len(results)} Artikel gefunden")
            return news_summary
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return f"Fehler bei der Nachrichtensuche: {str(e)}"
    
    @function_tool
    async def get_weather(self, location: str) -> str:
        """Holt aktuelle Wetterinformationen für einen bestimmten Ort"""
        try:
            logger.info(f"Suche Wetter für: {location}")
            
            # Verwende DuckDuckGo für Wettersuche
            query = f"weather {location} today current"
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=2))
            
            if not results:
                return f"Keine Wetterinformationen für '{location}' gefunden."
            
            # Extrahiere relevante Wetterinfo
            weather_info = []
            for result in results:
                if any(keyword in result['title'].lower() or keyword in result['body'].lower() 
                      for keyword in ['weather', 'temperature', 'forecast', 'wetter']):
                    weather_info.append(f"**{result['title']}**\n{result['body'][:300]}")
            
            if weather_info:
                return f"Wetter für {location}:\n\n" + "\n\n".join(weather_info)
            else:
                return f"Konnte keine spezifischen Wetterinformationen für '{location}' finden."
            
        except Exception as e:
            logger.error(f"Weather search error: {e}")
            return f"Fehler bei der Wettersuche: {str(e)}"
    
    async def close(self):
        """Schließe HTTP Client"""
        await self.client.aclose()
