"""
Audio Transcription Service using OpenAI's Whisper
===================================================

This module provides a service to transcribe audio files into text using
the OpenAI Whisper model.
"""

import os
import whisper
import logging
from tempfile import NamedTemporaryFile
import numpy as np
from scipy.io import wavfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperAudioTranscriber:
    """
    A class to handle audio transcription using the Whisper model.
    """

    def __init__(self, model_name: str = "base"):
        """
        Initializes the transcriber and loads the Whisper model.
        
        Args:
            model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium").
        """
        self.model_name = model_name
        self.model = None
        try:
            logger.info(f"Loading Whisper model: '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            logger.info("✓ Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe_audio_file(self, audio_data: bytes) -> str:
        """
        Transcribes an audio file to text.

        Args:
            audio_data (bytes): The audio data in bytes.

        Returns:
            str: The transcribed text.
        """
        if not self.model:
            logger.error("Whisper model is not loaded. Cannot transcribe.")
            return ""

        try:
            # The audio_data is in WAV format. We need to convert it to a NumPy array.
            # Read the WAV file from the bytes buffer
            samplerate, data = wavfile.read(io.BytesIO(audio_data))

            # Convert to float32, which is what Whisper expects
            audio_np = data.astype(np.float32) / 32768.0

            logger.info("Transcribing audio from in-memory NumPy array.")
            
            # Transcribe the audio data
            result = self.model.transcribe(audio_np, fp16=False, language='en')
            transcribed_text = result.get("text", "")

            
            logger.info(f"✓ Transcription successful. Text length: {len(transcribed_text)}")
            logger.info("Printable transcription result: %d ", len(transcribed_text.strip()));
            logger.info(f"\nTranscription result: {transcribed_text}")
            
            return transcribed_text

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            return ""


