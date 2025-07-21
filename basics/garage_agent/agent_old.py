import logging
from livekit import rtc
from livekit.agents import (
    JobContext, 
    Agent, 
    AgentSession, 
    RoomInputOptions,
    AutoSubscribe
)
from livekit.plugins import deepgram, openai, silero, noise_cancellation

logger = logging.getLogger("garage-assistant")

class GarageAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Du bist ein hilfreicher Kundenservice-Assistent für eine professionelle Autowerkstatt.
            Du kannst Kunden helfen bei:
            - Terminvereinbarungen für Service und Wartung
            - Erklärung von Autoreparaturen und technischen Problemen in einfachen Worten
            - Kostenvoranschlägen für gängige Dienstleistungen
            - Fragen zu Wartungsintervallen und Wartungsplänen
            - Überprüfung von Service-Historie und Garantieinformationen
            - Empfehlungen für vorbeugende Wartung
            
            Sei professionell, sachkundig und ehrlich bezüglich Reparaturbedarf.
            Betone immer Sicherheit und ordnungsgemäße Wartung.
            Wenn du dir bei spezifischen technischen Details unsicher bist, empfehle die Rücksprache mit unseren Mechanikern.""",
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1"
            ),
        )

async def entrypoint(ctx: JobContext):
    """Main entry point for the garage assistant agent"""
    logger.info("Garage assistant starting")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        # Create the agent
        agent = GarageAssistant()
        
        # Create the session with all components
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="llama3.2:latest",
                base_url="http://172.16.0.146:11434/v1"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova"
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
        
        # Generate initial greeting
        await session.generate_reply(
            instructions="Greet the customer warmly. Say: 'Hello! Welcome to our automotive service center. I'm here to help with your vehicle needs. How can I assist you today?'"
        )
        
        logger.info("Garage assistant ready and listening")
        
    except Exception as e:
        logger.error(f"Error in garage assistant: {e}", exc_info=True)
        raise

# Make sure the entrypoint is available at module level
__all__ = ['entrypoint']
