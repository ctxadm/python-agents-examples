# Internet Search Agent

Ein LiveKit Voice Agent mit Internetzugriff für aktuelle Informationen, Nachrichten und Web-Inhalte.

## Features

- 🔍 **Web-Suche**: Aktuelle Informationen via DuckDuckGo
- 📰 **Nachrichten**: Neueste News zu beliebigen Themen  
- 🌐 **Webseiten**: Vollständige Inhalte von URLs laden
- 🌤️ **Wetter**: Aktuelle Wetterinformationen für jeden Ort
- 🎯 **Kostenlos**: Keine API-Keys erforderlich

## Verfügbare Funktionen

### `search_web(query, max_results=3)`
Sucht im Internet nach aktuellen Informationen zu einem Thema.

**Beispiele:**
- "Suche nach Python 3.12 Features"
- "Aktuelle Informationen über ChatGPT"
- "Was ist Kubernetes?"

### `search_news(topic, max_results=3)`
Findet aktuelle Nachrichten zu einem bestimmten Thema.

**Beispiele:**
- "Neueste KI-Nachrichten"
- "Tesla Aktie News"
- "Klimawandel Nachrichten"

### `fetch_webpage(url)`
Lädt den vollständigen Inhalt einer Webseite.

**Beispiel:**
- "Lade den Inhalt von https://example.com"

### `get_weather(location)`
Holt aktuelle Wetterinformationen.

**Beispiele:**
- "Wetter in Berlin"
- "Wie ist das Wetter in Tokyo?"

## Installation

1. **Repository Fork aktualisieren:**
```bash
cd /path/to/python-agents-examples
git pull origin main
