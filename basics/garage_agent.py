import logging
from typing import Annotated
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe, llm, stt, tts, VoiceAssistant
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("garage-assistant")

async def entrypoint(ctx: JobContext):
    logger.info("Garage assistant starting")
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""You are a helpful automotive service assistant for a professional garage.
            You can help customers with:
            - Scheduling service appointments and maintenance
            - Explaining car repairs and technical issues in simple terms
            - Providing cost estimates for common services
            - Answering questions about vehicle maintenance schedules
            - Checking service history and warranty information
            - Recommending preventive maintenance
            
            Be professional, knowledgeable, and honest about repair needs.
            Always emphasize safety and proper maintenance.
            If you're unsure about specific technical details, recommend consulting with our mechanics."""
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(
            model="llama3.2:latest",
            base_url="http://172.16.0.146:11434/v1"
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="nova"  # oder "alloy" für eine andere Stimme
        ),
        chat_ctx=initial_ctx,
    )
    
    assistant.start(ctx.room)
    
    await assistant.say("Hello! Welcome to our automotive service center. I'm here to help with your vehicle needs. How can I assist you today?", allow_interruptions=True)
    
    logger.info("Garage assistant ready")
