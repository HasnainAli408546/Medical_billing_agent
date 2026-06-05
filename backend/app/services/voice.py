import os
import asyncio
import edge_tts
import logging

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        """
        Production-grade Voice Service using Groq's extremely fast Whisper API.
        Zero memory footprint locally! Essential for 512MB RAM constraints.
        """
        self.groq_key = os.getenv("GROQ_API_KEY")
        if self.groq_key:
            from groq import Groq
            self.client = Groq(api_key=self.groq_key)
            logger.info("VoiceService initialized with Groq Whisper API.")
        else:
            logger.warning("GROQ_API_KEY not found. Transcriptions will fail.")
            self.client = None
        
    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribes speech to text using Groq's whisper-large-v3.
        """
        if not self.client:
            raise Exception("GROQ_API_KEY is missing. Cannot transcribe audio.")
            
        try:
            with open(audio_file_path, "rb") as file:
                # Groq API expects a tuple (filename, bytes)
                transcription = self.client.audio.transcriptions.create(
                  file=(os.path.basename(audio_file_path), file.read()),
                  model="whisper-large-v3",
                  response_format="json",
                  language="en",
                )
            
            text = transcription.text
            return text.strip()
        except Exception as e:
            logger.error(f"Transcription failed via Groq API: {str(e)}")
            raise e

    async def generate_speech(self, text: str, output_path: str, voice: str = "en-US-AriaNeural") -> str:
        """
        Converts text back to speech using edge-tts.
        Common voices: en-US-AriaNeural (Female), en-US-GuyNeural (Male), en-US-ChristopherNeural (Male).
        """
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            logger.info(f"Generated TTS audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise e

# Instantiate as a singleton so FastAPI can import this directly 
# without reloading the heavy ML weights for every request.
voice_service = VoiceService()
