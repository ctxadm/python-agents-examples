import logging
from livekit import rtc
from livekit.agents import (
    JobContext, 
    Agent, 
    AgentSession, 
    RoomInputOptions,
    AutoSubscribe
)
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("medical-assistant")

class MedicalAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Du bist ein hilfreicher Assistent für medizinische Informationen.

WICHTIGER HINWEIS: Du stellst nur allgemeine Gesundheitsinformationen zur Verfügung.
Erinnere die Nutzer immer daran, dass:
- Dies KEIN Ersatz für professionellen medizinischen Rat ist
- Sie bei medizinischen Anliegen qualifizierte Gesundheitsfachkräfte konsultieren sollten
- Sie in Notfällen sofort den Notdienst anrufen sollten

Du kannst helfen bei:
- Allgemeinen Gesundheitsinformationen und Wellness-Tipps
- Erklärung gängiger medizinischer Begriffe in einfacher Sprache
- Grundlegenden Erste-Hilfe-Informationen
- Empfehlungen für einen gesunden Lebensstil
- Verständnis häufiger Symptome (mit Haftungsausschluss)
- Medikamenten-Erinnerungen und allgemeinen Informationen

Sei präzise, einfühlsam und klar in deinen Antworten.
Im Zweifelsfall solltest du immer zur Vorsicht raten und eine professionelle Beratung empfehlen.""",
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
                timeout=120.0,
                temperature=0.7,
            ),
        )

async def entrypoint(ctx: JobContext):
    """Main entry point for the medical assistant agent"""
    logger.info("Medical assistant starting")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Create the agent
        agent = MedicalAssistant()
        
        # Create the session with all components
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1",
                api_key="ollama",
                timeout=120.0,
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="shimmer"  # Professional, clear voice for medical info
            ),
        )
        
        # Start the session
        await session.start(
            agent=agent, 
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # Optional: Add noise cancellation if needed
                # noise_cancellation=noise_cancellation.BVC()
            )
        )
        
        # Generate initial greeting with medical disclaimer
        await session.generate_reply(
            instructions="""Greet the user professionally. Say: 
            'Hello! I'm your medical information assistant. I can help you understand general health topics and medical terms. 
            Please remember that I provide general information only and cannot replace professional medical advice. 
            For any medical concerns, please consult with a qualified healthcare provider. 
            How can I help you today?'"""
        )
        
        logger.info("Medical assistant ready and listening")
        
    except Exception as e:
        logger.error(f"Error in medical assistant: {e}", exc_info=True)
        raise

# Make sure the entrypoint is available at module level
__all__ = ['entrypoint']
