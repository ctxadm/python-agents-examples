# Agent Selector für LiveKit

Dieser Agent wählt automatisch zwischen verschiedenen AI Agents basierend auf dem Room-Namen.

## Unterstützte Agents

### 1. Vision-Ollama Agent
**Aktiviert durch Room-Namen mit:**
- `vision`
- `ollama`
- `bild`
- `image`
- `visual`  
- `camera`

**Beispiele:** 
- `vision-demo`
- `image-analysis-room`
- `bildverarbeitung-test`

### 2. RAG Agent (mit Qdrant)
**Aktiviert durch Room-Namen mit:**
- `rag`
- `knowledge`
- `wissen`
- `datenbank`
- `qdrant`
- `search`

**Beispiele:**
- `rag-demo`
- `knowledge-base`
- `wissensdatenbank-test`

## Standard-Verhalten

Wenn kein Keyword im Room-Namen erkannt wird, wird automatisch der **Vision-Ollama Agent** gestartet.

## Verwendung

1. LiveKit Playground öffnen
2. Mit Server verbinden
3. Room-Namen eingeben (z.B. `vision-test` oder `rag-demo`)
4. Connect klicken

Der Agent-Selector wählt automatisch den passenden Agent aus.
