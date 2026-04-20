import os
import asyncio
from faster_whisper import WhisperModel
import edge_tts
import logging

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Production-grade Voice Service initializing the Whisper model in memory.
        Using 'base' or 'small' is good for fast CPU inference. For GPU, use device='cuda' and compute_type='float16'
        """
        logger.info(f"Loading WhisperModel: size={model_size}, device={device}")
        # Model is loaded once at startup and lives in memory for rapid API requests
        self.stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("WhisperModel loaded successfully.")
        
    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribes speech to text. Uses beam search for accuracy.
        """
        try:
            # We enforce English for the copilot right now
            segments, info = self.stt_model.transcribe(audio_file_path, beam_size=5, language="en")
            
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
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
