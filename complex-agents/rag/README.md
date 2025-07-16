# RAG Agent for LiveKit

A Retrieval-Augmented Generation (RAG) agent that integrates with LiveKit for voice-based interactions.

## Features

- Multi-agent support (search, garage, medical)
- Integration with external RAG service
- German language support
- Voice interaction via LiveKit
- Automatic knowledge base search

## Configuration

Environment variables:
- `AGENT_TYPE`: Type of agent (search, garage, medical)
- `RAG_SERVICE_URL`: URL of the RAG service (default: http://localhost:8000)
- `LIVEKIT_URL`: LiveKit server URL
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret
- `OPENAI_API_KEY`: OpenAI API key for LLM
- `DEEPGRAM_API_KEY`: Deepgram API key for STT

## Usage

```python
from rag.main import entrypoint
