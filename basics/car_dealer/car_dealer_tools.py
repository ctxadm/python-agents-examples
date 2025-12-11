# File: car_dealer_tools.py
# Function Tools für Car Dealer Agent - Fahrzeug-Suche und Preisabfragen

import json
import os
import logging
from typing import Optional
from pathlib import Path
from livekit.agents import llm

logger = logging.getLogger("car-dealer-tools")

# =============================================================================
# CAR DEALER SERVICE - Lädt und verwaltet Fahrzeug-Daten
# =============================================================================

class CarDealerService:
    _instance = None
    
    def __init__(self):
        self.cars = []
        self._load_data()
    
    def _load_data(self):
        """Lädt alle Fahrzeug-Daten aus JSON"""
        # Pfade für Container und lokale Entwicklung
        possible_paths = [
            Path("/app/basics/car_dealer_agent/data/cars.json"),
            Path(__file__).parent / "data" / "cars.json",
            Path("./data/cars.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"📂 Lade Fahrzeug-Daten von: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    self.cars = json.load(f)
                logger.info(f"✅ {len(self.cars)} Fahrzeuge geladen")
                return
        
        raise FileNotFoundError(f"Fahrzeug-Daten nicht gefunden. Geprüfte Pfade: {possible_paths}")
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CarDealerService()
        return cls._instance
    
    # =========================================================================
    # HELPER METHODS
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
        else:
            return einer[e] + "und" + zehner[z]
    
    def _format_large_number(self, n: int) -> str:
        """Formatiert große Zahlen (Preise, Kilometer) als deutsche Worte"""
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
    
    def format_price(self, price_str: str) -> str:
        """Formatiert Preis für TTS-Ausgabe"""
        # Extrahiere Zahl aus String wie "CHF 43'990.-"
        import re
        match = re.search(r"([\d']+)", price_str)
        if match:
            number_str = match.group(1).replace("'", "")
            try:
                number = int(number_str)
                return self._format_large_number(number) + " Franken"
            except ValueError:
                return price_str
        return price_str
    
    def format_mileage(self, mileage_str: str) -> str:
        """Formatiert Kilometerstand für TTS-Ausgabe"""
        import re
        match = re.search(r"([\d']+)", mileage_str)
        if match:
            number_str = match.group(1).replace("'", "")
            try:
                number = int(number_str)
                return self._format_large_number(number) + " Kilometer"
            except ValueError:
                return mileage_str
        return mileage_str
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search_cars(self, brand: Optional[str] = None, max_price: Optional[int] = None,
                    min_price: Optional[int] = None, fuel_type: Optional[str] = None,
                    transmission: Optional[str] = None) -> list:
        """Sucht Fahrzeuge nach verschiedenen Kriterien"""
        results = self.cars.copy()
        
        if brand:
            brand_upper = brand.upper()
            # Auch Teilübereinstimmungen erlauben
            results = [c for c in results if brand_upper in c["brand"].upper()]
        
        if fuel_type:
            fuel_lower = fuel_type.lower()
            results = [c for c in results if fuel_lower in c["fuel_type"].lower()]
        
        if transmission:
            trans_lower = transmission.lower()
            results = [c for c in results if trans_lower in c["transmission"].lower()]
        
        if max_price:
            def extract_price(price_str):
                import re
                match = re.search(r"([\d']+)", price_str)
                if match:
                    return int(match.group(1).replace("'", ""))
                return 999999
            results = [c for c in results if extract_price(c["price_chf"]) <= max_price]
        
        if min_price:
            def extract_price(price_str):
                import re
                match = re.search(r"([\d']+)", price_str)
                if match:
                    return int(match.group(1).replace("'", ""))
                return 0
            results = [c for c in results if extract_price(c["price_chf"]) >= min_price]
        
        return results
    
    def get_car_by_id(self, car_id: int) -> Optional[dict]:
        """Holt ein spezifisches Fahrzeug nach ID"""
        for car in self.cars:
            if car["id"] == car_id:
                return car
        return None
    
    def get_brands(self) -> list:
        """Gibt alle verfügbaren Marken zurück"""
        brands = set(car["brand"] for car in self.cars)
        return sorted(list(brands))
    
    def get_cheapest(self, limit: int = 5) -> list:
        """Gibt die günstigsten Fahrzeuge zurück"""
        import re
        def extract_price(car):
            match = re.search(r"([\d']+)", car["price_chf"])
            if match:
                return int(match.group(1).replace("'", ""))
            return 999999
        
        sorted_cars = sorted(self.cars, key=extract_price)
        return sorted_cars[:limit]
    
    def get_electric_hybrid(self) -> list:
        """Gibt alle Elektro- und Hybrid-Fahrzeuge zurück"""
        keywords = ["elektro", "hybrid", "electric"]
        results = []
        for car in self.cars:
            fuel_lower = car["fuel_type"].lower()
            if any(kw in fuel_lower for kw in keywords):
                results.append(car)
        return results


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_service() -> CarDealerService:
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
            brand: Automarke (z.B. "Mercedes", "BMW", "Audi")
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
            output += f"  Leistung: {car['power_ps']}\n"
            output += f"  Standort: {car['location']}\n\n"
        
        if len(results) > 5:
            output += f"... und {len(results) - 5} weitere Fahrzeuge."
        
        return output
    
    @llm.function_tool
    async def get_car_details(car_id: int) -> str:
        """
        Holt detaillierte Informationen zu einem spezifischen Fahrzeug.
        
        Args:
            car_id: Die ID des Fahrzeugs (1-24)
        
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
        output += f"Leistung: {car['power_ps']}\n"
        output += f"Kraftstoff: {car['fuel_type']}\n"
        output += f"Getriebe: {car['transmission']}\n"
        output += f"Erstzulassung: {car['year']}\n"
        output += f"Standort: {car['location']}\n"
        output += f"Ausstattung: {', '.join(car['features'])}\n"
        output += f"Beschreibung: {car['description']}"
        
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
            limit: Anzahl der Fahrzeuge (Standard: 5)
        
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
