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
            instructions="""You are a helpful automotive service assistant for a professional garage.
            You can help customers with:
            - Scheduling service appointments and maintenance
            - Explaining car repairs and technical issues in simple terms
            - Providing cost estimates for common services
            - Answering questions about vehicle maintenance schedules
            - Checking service history and warranty information
            - Recommending preventive maintenance
            
            Be professional, knowledgeable, and honest about repair needs.
            Always emphasize safety and proper maintenance.
            If you're unsure about specific technical details, recommend consulting with our mechanics.""",
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
