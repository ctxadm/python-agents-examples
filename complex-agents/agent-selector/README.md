# complex-agents/agent-selector/README.md

# LiveKit Agent Selector

Dieser Agent wählt automatisch zwischen verschiedenen Agent-Typen basierend auf dem Room-Namen.

## Unterstützte Agents

### Vision-Ollama Agent
Wird aktiviert bei Room-Namen mit:
- `vision`
- `ollama`
- `bild`
- `image`
- `visual`

Beispiele: `vision-demo`, `image-analysis`, `bildverarbeitung`

### RAG Agent (Qdrant)
Wird aktiviert bei Room-Namen mit:
- `rag`
- `knowledge`
- `wissen`
- `datenbank`
- `qdrant`
- `search`

Beispiele: `rag-demo`, `wissensdatenbank`, `knowledge-base`

## Standard-Verhalten
Wenn kein Keyword erkannt wird, startet der Vision-Ollama Agent.

## Umgebungsvariablen

```bash
# Ollama Konfiguration (für Vision Agent)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llava-llama3:latest

# RAG Service
RAG_SERVICE_URL=http://localhost:8000

# OpenAI (für TTS und RAG LLM)
OPENAI_API_KEY=sk-...

# Deepgram (für STT)
DEEPGRAM_API_KEY=...
