# File: data_travel_tools.py
# Data Travel Tools für LiveKit Mobile Support Agent
# Lädt Roaming-Daten aus JSON und stellt Tools für Function Calling bereit

import json
import logging
import os
from typing import Optional
from livekit.agents import llm

logger = logging.getLogger("data-travel-tools")

# =============================================================================
# KONFIGURATION - Pfad zu den JSON-Dateien
# =============================================================================

DATA_PATH = os.getenv(
    "DATA_TRAVEL_PATH", 
    "/home/ctxusr/live.fastlane-ai.ch/agent-dateien/mobile-agent"
)

# =============================================================================
# DATA TRAVEL SERVICE - Lädt und verwaltet die Daten
# =============================================================================

class DataTravelService:
    """
    Service-Klasse zum Laden und Abfragen der Data Travel Roaming-Daten.
    Singleton-Pattern: Daten werden nur einmal geladen.
    """
    
    _instance: Optional['DataTravelService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.laender: dict = {}
        self.aliase: dict = {}
        self.zonen: dict = {}
        self._load_data()
        self._initialized = True
        logger.info(f"✅ DataTravelService initialisiert: {len(self.laender)} Länder, {len(self.aliase)} Aliase")
    
    def _load_data(self):
        """Lädt alle JSON-Dateien beim Start"""
        try:
            # Länder-Daten laden
            laender_path = os.path.join(DATA_PATH, "data-travel-laender.json")
            with open(laender_path, 'r', encoding='utf-8') as f:
                self.laender = json.load(f)
            logger.info(f"📁 Länder geladen: {laender_path}")
            
            # Aliase laden
            aliase_path = os.path.join(DATA_PATH, "data-travel-aliase.json")
            with open(aliase_path, 'r', encoding='utf-8') as f:
                self.aliase = json.load(f)
            # _info Feld entfernen falls vorhanden
            self.aliase.pop('_info', None)
            logger.info(f"📁 Aliase geladen: {aliase_path}")
            
            # Zonen-Übersicht laden
            zonen_path = os.path.join(DATA_PATH, "data-travel-zonen.json")
            with open(zonen_path, 'r', encoding='utf-8') as f:
                self.zonen = json.load(f)
            logger.info(f"📁 Zonen geladen: {zonen_path}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ JSON-Datei nicht gefunden: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON-Parse-Fehler: {e}")
            raise
    
    def resolve_alias(self, name: str) -> str:
        """
        Löst alternative Ländernamen auf.
        z.B. "England" -> "Grossbritannien", "USA" -> "USA"
        """
        # Erst direkt in Ländern suchen (case-insensitive)
        for land in self.laender.keys():
            if land.lower() == name.lower():
                return land
        
        # Dann in Aliasen suchen (case-insensitive)
        for alias, ziel in self.aliase.items():
            if alias.lower() == name.lower():
                return ziel
        
        # Nicht gefunden - Original zurückgeben
        return name
    
    def get_country_info(self, country: str) -> Optional[dict]:
        """
        Gibt alle Informationen zu einem Land zurück.
        Löst automatisch Aliase auf.
        """
        resolved = self.resolve_alias(country)
        return self.laender.get(resolved)
    
    def get_country_name(self, country: str) -> str:
        """Gibt den offiziellen Ländernamen zurück"""
        return self.resolve_alias(country)
    
    def format_price(self, price: Optional[float]) -> str:
        """Formatiert Preise für die Ausgabe"""
        if price is None:
            return "nicht verfügbar"
        return f"CHF {price:.2f}".replace('.', ',')
    
    def get_available_packages(self, country: str) -> list[tuple[str, float]]:
        """Gibt alle verfügbaren Pakete für ein Land zurück"""
        info = self.get_country_info(country)
        if not info:
            return []
        
        available = []
        for paket, preis in info.get("pakete", {}).items():
            if preis is not None:
                available.append((paket, preis))
        return available


# =============================================================================
# GLOBALE SERVICE-INSTANZ
# =============================================================================

_service: Optional[DataTravelService] = None

def get_service() -> DataTravelService:
    """Gibt die Singleton-Instanz des DataTravelService zurück"""
    global _service
    if _service is None:
        _service = DataTravelService()
    return _service


# =============================================================================
# LIVEKIT FUNCTION TOOLS
# =============================================================================

@llm.function_tool
def get_data_travel_info(country: str) -> str:
    """
    Gibt alle verfügbaren Data Travel Roaming-Pakete und Preise für ein bestimmtes Land zurück.
    Nutze diese Funktion wenn ein Kunde nach Roaming-Preisen, Datenpaketen oder 
    Data Travel für ein bestimmtes Reiseland fragt.
    
    Args:
        country: Name des Landes (z.B. "Thailand", "USA", "England", "Schweiz")
    
    Returns:
        Formatierte Übersicht aller verfügbaren Pakete mit Preisen
    """
    service = get_service()
    
    # Alias auflösen
    resolved_name = service.get_country_name(country)
    info = service.get_country_info(country)
    
    if not info:
        return f"Das Land '{country}' wurde leider nicht in unseren Data Travel Paketen gefunden. Bitte überprüfen Sie die Schreibweise oder fragen Sie nach einem anderen Land."
    
    zone = info.get("zone", "Unbekannt")
    pakete = info.get("pakete", {})
    
    # Verfügbare Pakete sammeln
    available = []
    not_available = []
    
    for paket, preis in pakete.items():
        if preis is not None:
            available.append(f"- {paket}: {service.format_price(preis)}")
        else:
            # Paketgröße extrahieren für die Nicht-Verfügbar-Liste
            not_available.append(paket.replace("Data Travel ", ""))
    
    # Antwort formatieren
    response_parts = [
        f"Data Travel Informationen für {resolved_name} (Zone: {zone}):",
        ""
    ]
    
    if available:
        response_parts.append("Verfügbare Pakete:")
        response_parts.extend(available)
    else:
        response_parts.append("Leider sind für dieses Land keine Data Travel Pakete verfügbar.")
    
    if not_available and available:
        response_parts.append("")
        response_parts.append(f"Nicht verfügbar: {', '.join(not_available)}")
    
    return "\n".join(response_parts)


@llm.function_tool
def get_package_price(country: str, package_size: str) -> str:
    """
    Gibt den Preis für ein spezifisches Data Travel Paket in einem bestimmten Land zurück.
    Nutze diese Funktion wenn ein Kunde nach dem Preis eines bestimmten Datenpakets fragt.
    
    Args:
        country: Name des Landes (z.B. "Thailand", "Mexiko", "England")
        package_size: Größe des Pakets (z.B. "100 MB", "500 MB", "1 GB", "5 GB", "10 GB")
    
    Returns:
        Preis des Pakets oder Information dass es nicht verfügbar ist
    """
    service = get_service()
    
    resolved_name = service.get_country_name(country)
    info = service.get_country_info(country)
    
    if not info:
        return f"Das Land '{country}' wurde nicht gefunden."
    
    # Paketname normalisieren
    size_normalized = package_size.upper().replace(" ", "")
    pakete = info.get("pakete", {})
    
    # Passendes Paket finden
    for paket_name, preis in pakete.items():
        paket_normalized = paket_name.upper().replace(" ", "").replace("DATATRAVEL", "")
        if size_normalized in paket_normalized or paket_normalized in size_normalized:
            if preis is not None:
                return f"Das Data Travel {package_size} Paket kostet für {resolved_name} {service.format_price(preis)}."
            else:
                # Alternative Pakete vorschlagen
                available = service.get_available_packages(country)
                if available:
                    alt_text = ", ".join([f"{p[0]}: {service.format_price(p[1])}" for p in available[:3]])
                    return f"Das Data Travel {package_size} Paket ist für {resolved_name} leider nicht verfügbar. Verfügbare Alternativen: {alt_text}"
                return f"Das Data Travel {package_size} Paket ist für {resolved_name} leider nicht verfügbar."
    
    return f"Das Paket '{package_size}' wurde nicht erkannt. Verfügbare Größen: 100 MB, 500 MB, 1 GB, 5 GB, 10 GB."


@llm.function_tool
def list_countries_in_zone(zone: str) -> str:
    """
    Listet alle Länder in einer bestimmten Tarifzone auf.
    Nutze diese Funktion wenn ein Kunde wissen möchte, welche Länder zu einer Zone gehören.
    
    Args:
        zone: Name der Zone (z.B. "EU/UK", "Welt 1", "Welt 2", "Rest der Welt")
    
    Returns:
        Liste aller Länder in dieser Zone
    """
    service = get_service()
    
    # Zone normalisieren
    zone_mapping = {
        "eu": "EU/UK",
        "uk": "EU/UK",
        "eu/uk": "EU/UK",
        "europa": "EU/UK",
        "welt1": "Welt 1",
        "welt 1": "Welt 1",
        "welt2": "Welt 2", 
        "welt 2": "Welt 2",
        "rest": "Rest der Welt",
        "rest der welt": "Rest der Welt",
        "restderwelt": "Rest der Welt"
    }
    
    zone_normalized = zone_mapping.get(zone.lower(), zone)
    
    # Länder in dieser Zone finden
    countries_in_zone = []
    for land, info in service.laender.items():
        if info.get("zone") == zone_normalized:
            countries_in_zone.append(land)
    
    if not countries_in_zone:
        return f"Die Zone '{zone}' wurde nicht gefunden. Verfügbare Zonen: EU/UK, Welt 1, Welt 2, Rest der Welt."
    
    countries_in_zone.sort()
    
    # Zonen-Info holen
    zone_info = service.zonen.get("zonen_uebersicht", {}).get(zone_normalized, {})
    beschreibung = zone_info.get("beschreibung", "")
    
    response = f"Länder in Zone '{zone_normalized}'"
    if beschreibung:
        response += f" ({beschreibung})"
    response += f":\n\n{', '.join(countries_in_zone)}"
    response += f"\n\nInsgesamt: {len(countries_in_zone)} Länder"
    
    return response


@llm.function_tool
def get_zone_prices(zone: str) -> str:
    """
    Gibt die Paketpreise für eine bestimmte Tarifzone zurück.
    Nutze diese Funktion wenn ein Kunde nach den generellen Preisen einer Zone fragt.
    
    Args:
        zone: Name der Zone (z.B. "EU/UK", "Welt 1", "Welt 2", "Rest der Welt")
    
    Returns:
        Preisübersicht für alle Pakete in dieser Zone
    """
    service = get_service()
    
    # Zone normalisieren (gleiche Logik wie oben)
    zone_mapping = {
        "eu": "EU/UK",
        "uk": "EU/UK",
        "eu/uk": "EU/UK",
        "europa": "EU/UK",
        "welt1": "Welt 1",
        "welt 1": "Welt 1",
        "welt2": "Welt 2",
        "welt 2": "Welt 2",
        "rest": "Rest der Welt",
        "rest der welt": "Rest der Welt",
        "restderwelt": "Rest der Welt"
    }
    
    zone_normalized = zone_mapping.get(zone.lower(), zone)
    
    zone_info = service.zonen.get("zonen_uebersicht", {}).get(zone_normalized)
    
    if not zone_info:
        return f"Die Zone '{zone}' wurde nicht gefunden. Verfügbare Zonen: EU/UK, Welt 1, Welt 2, Rest der Welt."
    
    beschreibung = zone_info.get("beschreibung", "")
    pakete = zone_info.get("verfuegbare_pakete", {})
    
    response_parts = [f"Preise für Zone '{zone_normalized}'"]
    if beschreibung:
        response_parts[0] += f" ({beschreibung})"
    response_parts.append("")
    
    for paket, preis in pakete.items():
        response_parts.append(f"- {paket}: {service.format_price(preis)}")
    
    return "\n".join(response_parts)


# =============================================================================
# TOOL-LISTE FÜR AGENT-REGISTRIERUNG
# =============================================================================

def get_data_travel_tools() -> list:
    """
    Gibt alle Data Travel Tools als Liste zurück.
    Wird vom Agent verwendet um die Tools zu registrieren.
    """
    return [
        get_data_travel_info,
        get_package_price,
        list_countries_in_zone,
        get_zone_prices
    ]


# =============================================================================
# TEST-FUNKTION
# =============================================================================

if __name__ == "__main__":
    # Für lokales Testen
    logging.basicConfig(level=logging.INFO)
    
    print("=== Data Travel Tools Test ===\n")
    
    # Service initialisieren
    service = get_service()
    
    # Test 1: Land mit Alias
    print("Test 1: England (Alias für Grossbritannien)")
    print(get_data_travel_info("England"))
    print()
    
    # Test 2: Spezifischer Preis
    print("Test 2: 1 GB Paket für Thailand")
    print(get_package_price("Thailand", "1 GB"))
    print()
    
    # Test 3: Nicht verfügbares Paket
    print("Test 3: 5 GB für Kuba (nicht verfügbar)")
    print(get_package_price("Kuba", "5 GB"))
    print()
    
    # Test 4: Länder in Zone
    print("Test 4: Länder in EU/UK")
    print(list_countries_in_zone("EU/UK"))
