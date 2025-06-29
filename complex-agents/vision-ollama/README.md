# Ollama Vision Agent

This is a modified version of the Vision Agent that uses Ollama instead of X.AI Grok-2-Vision.

## Features

- Uses local Ollama server for vision processing
- Compatible with vision models like llava
- Same functionality as original vision agent but with local processing

## Configuration

Set these environment variables:

- `OLLAMA_HOST`: Your Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: The vision model to use (default: `llava`)

## Requirements

1. Ollama server running with a vision-capable model installed:
   ```bash
   ollama pull llava
   ollama serve
   ```

2. Standard LiveKit agent requirements:
   - Deepgram API key for STT
   - Cartesia API key for TTS
   - LiveKit server connection

## Usage

In your Dockerfile, use:
```dockerfile
CMD ["python", "complex-agents/vision-ollama/agent.py", "start"]
```

## Differences from Original

- Uses Ollama instead of X.AI Grok-2-Vision
- Uses Cartesia TTS instead of Rime TTS
- Processes vision locally instead of via cloud API
