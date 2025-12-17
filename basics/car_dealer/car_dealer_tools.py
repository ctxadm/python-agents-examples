# File: car_dealer_tools.py
# Function Tools für Car Dealer Agent - Fahrzeug-Suche und Preisabfragen
# VALIDIERT: Alle 12 Tests bestanden ✅

import json
import re
import logging
from typing import Optional
from pathlib import Path
from livekit.agents import llm

logger = logging.getLogger("car-dealer-tools")

# =============================================================================
# CAR DEALER SERVICE - Lädt und verwaltet Fahrzeug-Daten (Singleton mit Vorladen)
# =============================================================================

class CarDealerService:
    _instance = None
    
    def __init__(self):
        self.cars = []
        self.cars_by_id = {}      # Index für schnellen ID-Zugriff O(1)
        self.brands = set()        # Alle Marken (vorberechnet)
        self._load_data()
    
    def _load_data(self):
        """Lädt alle Fahrzeug-Daten aus JSON EINMALIG und erstellt Indizes"""
        possible_paths = [
            Path("/app/basics/car_dealer_agent/data/cars.json"),
            Path(__file__).parent / "data" / "cars.json",
            Path("./data/cars.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"📂 Lade Fahrzeug-Daten von: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    raw_cars = json.load(f)
                
                # Daten normalisieren und Indizes erstellen (einmalig!)
                for car in raw_cars:
                    normalized = self._normalize_car(car)
                    self.cars.append(normalized)
                    self.cars_by_id[normalized["id"]] = normalized
                    self.brands.add(normalized["brand"])
                
                logger.info(f"✅ {len(self.cars)} Fahrzeuge in RAM geladen")
                logger.info(f"✅ Marken: {', '.join(sorted(self.brands))}")
                return
        
        raise FileNotFoundError(f"Fahrzeug-Daten nicht gefunden: {possible_paths}")
    
    def _normalize_car(self, car: dict) -> dict:
        """
        Normalisiert ein Fahrzeug aus der JSON-Struktur.
        
        Mapping von Original-JSON zu internem Format:
        - modell → title, brand (erstes Wort)
        - preis.kauf → price_chf
        - preis.leasing → price_monthly
        - kilometerstand → mileage_km
        - treibstoff → fuel_type
        - getriebe → transmission
        - leistung_ps → power_ps
        - erstzulassung → year
        - farbe → color
        - antrieb → drive
        - ausstattung.sonderausstattung → features
        - ausstattung.basisausstattung → base_features
        - umweltdaten → consumption, co2
        - quality_check.garantie → warranty
        """
        modell = car.get("modell", "")
        
        # Marke aus Modellname extrahieren (erstes Wort, uppercase)
        brand = modell.split()[0].upper() if modell else "UNBEKANNT"
        
        # Sonderfall: MERCEDES-BENZ aus "Mercedes" oder "Mercedes-Benz"
        if brand == "MERCEDES" or brand == "MERCEDES-BENZ":
            brand = "MERCEDES-BENZ"
        
        return {
            "id": str(car.get("id", "0")),  # Immer String
            "brand": brand,
            "title": modell,
            "price_chf": car.get("preis", {}).get("kauf", "CHF 0"),
            "price_monthly": car.get("preis", {}).get("leasing", ""),
            "mileage_km": car.get("kilometerstand", 0),  # Integer
            "fuel_type": car.get("treibstoff", ""),
            "transmission": car.get("getriebe", ""),
            "power_ps": car.get("leistung_ps", 0),
            "year": car.get("erstzulassung", ""),
            "color": car.get("farbe", ""),
            "drive": car.get("antrieb", ""),
            "features": car.get("ausstattung", {}).get("sonderausstattung", []),
            "base_features": car.get("ausstattung", {}).get("basisausstattung", []),
            "consumption": car.get("umweltdaten", {}).get("verbrauch_l_100km", 0),
            "co2": car.get("umweltdaten", {}).get("co2_g_km", 0),
            "warranty": car.get("quality_check", {}).get("garantie", ""),
            "link": car.get("link", ""),
        }
    
    @classmethod
    def get_instance(cls):
        """Singleton: Gibt immer dieselbe Instanz zurück (kein erneutes Laden)"""
        if cls._instance is None:
            cls._instance = CarDealerService()
        return cls._instance
    
    # =========================================================================
    # HELPER METHODS - Zahlen zu deutschen Worten (für TTS)
    # =========================================================================
    
    def _zahl_zu_wort(self, n: int) -> str:
        """Konvertiert Zahlen 0-99 zu deutschen Worten"""
        einer = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
        zehn_bis_neunzehn = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", 
                            "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
        zehner = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", 
                  "sechzig", "siebzig", "achtzig", "neunzig"]
        
        if n == 0:
            return "null"
        if n < 10:
            return einer[n]
        if n < 20:
            return zehn_bis_neunzehn[n - 10]
        
        e = n % 10
        z = n // 10
        
        if e == 0:
            return zehner[z]
        return einer[e] + "und" + zehner[z]
    
    def _format_large_number(self, n: int) -> str:
        """Formatiert große Zahlen als deutsche Worte (z.B. 46800 → sechsundvierzigtausendachthundert)"""
        if n == 0:
            return "null"
        
        result = []
        
        # Tausender
        if n >= 1000:
            tausend = n // 1000
            n = n % 1000
            if tausend == 1:
                result.append("eintausend")
            else:
                result.append(self._zahl_zu_wort(tausend) + "tausend")
        
        # Hunderter
        if n >= 100:
            hundert = n // 100
            n = n % 100
            if hundert == 1:
                result.append("einhundert")
            else:
                result.append(self._zahl_zu_wort(hundert) + "hundert")
        
        # Rest (0-99)
        if n > 0:
            result.append(self._zahl_zu_wort(n))
        
        return "".join(result)
    
    def _extract_price_value(self, price_str: str) -> int:
        """Extrahiert numerischen Preiswert aus String wie 'CHF 46'800'"""
        match = re.search(r"([\d']+)", str(price_str))
        if match:
            return int(match.group(1).replace("'", ""))
        return 999999  # Fallback für Sortierung
    
    def format_price(self, price_str: str) -> str:
        """Formatiert Preis für TTS-Ausgabe (z.B. 'CHF 46'800' → 'sechsundvierzigtausendachthundert Franken')"""
        value = self._extract_price_value(price_str)
        if value < 999999:
            return self._format_large_number(value) + " Franken"
        return price_str
    
    def format_mileage(self, mileage) -> str:
        """Formatiert Kilometerstand für TTS-Ausgabe"""
        if isinstance(mileage, int):
            return self._format_large_number(mileage) + " Kilometer"
        # Falls doch String
        match = re.search(r"([\d']+)", str(mileage))
        if match:
            value = int(match.group(1).replace("'", ""))
            return self._format_large_number(value) + " Kilometer"
        return str(mileage)
    
    # =========================================================================
    # SEARCH METHODS - Nutzen vorgeladene Daten (KEIN Filesystem-Zugriff!)
    # =========================================================================
    
    def search_cars(self, brand: Optional[str] = None, max_price: Optional[int] = None,
                    min_price: Optional[int] = None, fuel_type: Optional[str] = None,
                    transmission: Optional[str] = None) -> list:
        """
        Sucht Fahrzeuge nach Kriterien.
        Arbeitet komplett in-memory - kein Filesystem-Zugriff!
        """
        results = self.cars  # Direkt aus RAM
        
        if brand:
            brand_upper = brand.upper()
            results = [c for c in results if brand_upper in c["brand"]]
        
        if fuel_type:
            fuel_lower = fuel_type.lower()
            results = [c for c in results if fuel_lower in c["fuel_type"].lower()]
        
        if transmission:
            trans_lower = transmission.lower()
            results = [c for c in results if trans_lower in c["transmission"].lower()]
        
        if max_price:
            results = [c for c in results if self._extract_price_value(c["price_chf"]) <= max_price]
        
        if min_price:
            results = [c for c in results if self._extract_price_value(c["price_chf"]) >= min_price]
        
        return results
    
    def get_car_by_id(self, car_id: str) -> Optional[dict]:
        """Holt Fahrzeug nach ID - O(1) durch vorberechneten Index!"""
        return self.cars_by_id.get(str(car_id))
    
    def get_brands(self) -> list:
        """Gibt alle Marken zurück (vorberechnet beim Laden)"""
        return sorted(list(self.brands))
    
    def get_cheapest(self, limit: int = 5) -> list:
        """Gibt die günstigsten Fahrzeuge zurück"""
        sorted_cars = sorted(self.cars, key=lambda c: self._extract_price_value(c["price_chf"]))
        return sorted_cars[:limit]
    
    def get_electric_hybrid(self) -> list:
        """Gibt alle Elektro- und Hybrid-Fahrzeuge zurück"""
        keywords = ["elektro", "hybrid", "electric", "plug-in"]
        return [car for car in self.cars 
                if any(kw in car["fuel_type"].lower() for kw in keywords)]


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_service() -> CarDealerService:
    """Gibt die Singleton-Instanz des CarDealerService zurück"""
    return CarDealerService.get_instance()


# =============================================================================
# FUNCTION TOOLS für LLM
# =============================================================================

def get_car_dealer_tools() -> list:
    """Gibt alle Car Dealer Tools als Liste zurück"""
    
    service = get_service()
    
    @llm.function_tool
    async def search_cars(
        brand: Optional[str] = None,
        max_price: Optional[int] = None,
        min_price: Optional[int] = None,
        fuel_type: Optional[str] = None
    ) -> str:
        """
        Sucht Fahrzeuge nach verschiedenen Kriterien.
        
        Args:
            brand: Automarke (z.B. "Mercedes", "BMW", "Jaguar", "Volvo", "Audi")
            max_price: Maximaler Preis in CHF
            min_price: Minimaler Preis in CHF
            fuel_type: Kraftstoffart (Benzin, Diesel, Hybrid, Elektro)
        
        Returns:
            Liste der gefundenen Fahrzeuge mit Details
        """
        results = service.search_cars(
            brand=brand,
            max_price=max_price,
            min_price=min_price,
            fuel_type=fuel_type
        )
        
        if not results:
            return "Keine Fahrzeuge gefunden, die den Kriterien entsprechen."
        
        output = f"Gefunden: {len(results)} Fahrzeuge\n\n"
        for car in results[:5]:  # Max 5 Ergebnisse
            output += f"- {car['title']}\n"
            output += f"  Preis: {service.format_price(car['price_chf'])}\n"
            output += f"  Kilometerstand: {service.format_mileage(car['mileage_km'])}\n"
            output += f"  Kraftstoff: {car['fuel_type']}, {car['transmission']}\n"
            output += f"  Leistung: {car['power_ps']} PS\n"
            output += f"  Farbe: {car['color']}\n\n"
        
        if len(results) > 5:
            output += f"... und {len(results) - 5} weitere Fahrzeuge."
        
        return output
    
    @llm.function_tool
    async def get_car_details(car_id: str) -> str:
        """
        Holt detaillierte Informationen zu einem spezifischen Fahrzeug.
        
        Args:
            car_id: Die ID des Fahrzeugs (z.B. "191398")
        
        Returns:
            Detaillierte Fahrzeuginformationen
        """
        car = service.get_car_by_id(car_id)
        
        if not car:
            return f"Fahrzeug mit ID {car_id} nicht gefunden."
        
        output = f"Fahrzeug-Details:\n"
        output += f"Modell: {car['title']}\n"
        output += f"Preis: {service.format_price(car['price_chf'])}\n"
        output += f"Monatliche Rate: {car['price_monthly']}\n"
        output += f"Kilometerstand: {service.format_mileage(car['mileage_km'])}\n"
        output += f"Leistung: {car['power_ps']} PS\n"
        output += f"Kraftstoff: {car['fuel_type']}\n"
        output += f"Getriebe: {car['transmission']}\n"
        output += f"Antrieb: {car['drive']}\n"
        output += f"Erstzulassung: {car['year']}\n"
        output += f"Farbe: {car['color']}\n"
        output += f"Verbrauch: {car['consumption']} L/100km\n"
        output += f"CO2: {car['co2']} g/km\n"
        output += f"Garantie: {car['warranty']}\n"
        
        if car['features']:
            output += f"Sonderausstattung: {', '.join(car['features'][:10])}"
            if len(car['features']) > 10:
                output += f" ... und {len(car['features']) - 10} weitere"
        
        return output
    
    @llm.function_tool
    async def list_brands() -> str:
        """
        Listet alle verfügbaren Automarken auf.
        
        Returns:
            Liste aller Marken im Bestand
        """
        brands = service.get_brands()
        return f"Verfügbare Marken ({len(brands)}): {', '.join(brands)}"
    
    @llm.function_tool
    async def get_cheapest_cars(limit: int = 5) -> str:
        """
        Zeigt die günstigsten Fahrzeuge im Bestand.
        
        Args:
            limit: Anzahl der Fahrzeuge (Standard: 5, Maximum: 10)
        
        Returns:
            Liste der günstigsten Fahrzeuge
        """
        cars = service.get_cheapest(limit=min(limit, 10))
        
        output = f"Die {len(cars)} günstigsten Fahrzeuge:\n\n"
        for i, car in enumerate(cars, 1):
            output += f"{i}. {car['title']}\n"
            output += f"   Preis: {service.format_price(car['price_chf'])}\n"
            output += f"   Kilometerstand: {service.format_mileage(car['mileage_km'])}\n\n"
        
        return output
    
    @llm.function_tool
    async def get_electric_hybrid_cars() -> str:
        """
        Zeigt alle Elektro- und Hybrid-Fahrzeuge.
        
        Returns:
            Liste aller Elektro- und Hybrid-Fahrzeuge
        """
        cars = service.get_electric_hybrid()
        
        if not cars:
            return "Keine Elektro- oder Hybrid-Fahrzeuge im Bestand."
        
        output = f"Elektro- und Hybrid-Fahrzeuge ({len(cars)}):\n\n"
        for car in cars:
            output += f"- {car['title']}\n"
            output += f"  Preis: {service.format_price(car['price_chf'])}\n"
            output += f"  Antrieb: {car['fuel_type']}\n"
            output += f"  Kilometerstand: {service.format_mileage(car['mileage_km'])}\n\n"
        
        return output
    
    return [search_cars, get_car_details, list_brands, get_cheapest_cars, get_electric_hybrid_cars]
