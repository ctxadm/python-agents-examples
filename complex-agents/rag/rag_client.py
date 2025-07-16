import os
import logging
import httpx
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class RAGServiceClient:
    """Client für die Kommunikation mit dem zentralen RAG Service"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self._collections_cache = None
        logger.info(f"RAG Client initialisiert mit URL: {self.base_url}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Schließe HTTP Client"""
        await self.client.aclose()
    
    async def health_check(self) -> Dict:
        """Prüfe RAG Service Status"""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Health Check Fehler: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_collections(self) -> Optional[Dict]:
        """Hole verfügbare Collections (mit Cache)"""
        if self._collections_cache is None:
            try:
                response = await self.client.get("/collections")
                if response.status_code == 200:
                    self._collections_cache = response.json()
                    logger.info(f"Collections geladen: {len(self._collections_cache.get('collections', []))} verfügbar")
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der Collections: {e}")
        return self._collections_cache
    
    async def search(
        self, 
        query: str, 
        agent_type: str = "general",
        collection: Optional[str] = None,
        top_k: int = 3
    ) -> Dict:
        """
        Suche in der Wissensdatenbank
        
        Args:
            query: Suchanfrage
            agent_type: Typ des Agents (search, garage, medical, etc.)
            collection: Optionale spezifische Collection
            top_k: Anzahl der Ergebnisse
            
        Returns:
            Dict mit results und collection_used
        """
        try:
            request_data = {
                "query": query,
                "agent_type": agent_type,
                "top_k": top_k
            }
            
            if collection:
                request_data["collection"] = collection
            
            logger.info(f"RAG Suche: '{query}' (agent_type={agent_type}, collection={collection})")
            
            response = await self.client.post("/search", json=request_data)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"RAG Suche erfolgreich: {len(data.get('results', []))} Ergebnisse aus {data.get('collection_used')}")
                return data
            else:
                error_msg = f"RAG Service Fehler: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "results": [],
                    "collection_used": "none",
                    "error": error_msg
                }
                
        except httpx.TimeoutException:
            logger.error("RAG Service Timeout")
            return {
                "results": [],
                "collection_used": "none",
                "error": "Timeout bei der Suche"
            }
        except Exception as e:
            logger.error(f"RAG Anfrage-Fehler: {e}")
            return {
                "results": [],
                "collection_used": "none",
                "error": str(e)
            }
    
    def format_results(self, search_response: Dict, max_length: int = 300) -> str:
        """
        Formatiere Suchergebnisse für die Ausgabe
        
        Args:
            search_response: Response vom search() Call
            max_length: Maximale Länge pro Ergebnis
            
        Returns:
            Formatierter String
        """
        results = search_response.get("results", [])
        collection_used = search_response.get("collection_used", "unbekannt")
        error = search_response.get("error")
        
        if error:
            return f"Fehler bei der Suche: {error}"
        
        if not results:
            return "Keine relevanten Informationen in der Wissensdatenbank gefunden."
        
        # Formatiere die Ergebnisse
        formatted = f"Aus der Wissensdatenbank ({collection_used}):\n\n"
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            
            # Kürze lange Inhalte
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            formatted += f"{i}. {content}\n"
            
            # Metadata hinzufügen falls vorhanden
            source = metadata.get("source", metadata.get("title", ""))
            if source:
                formatted += f"   📄 Quelle: {source}\n"
            
            # Relevanz-Score
            formatted += f"   📊 Relevanz: {score:.2%}\n\n"
        
        return formatted.strip()
