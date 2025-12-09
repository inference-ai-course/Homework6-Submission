"""
Whisper Service
Handles audio transcription using OpenAI Whisper
"""
import whisper
import torch
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperService:
    """Service for audio transcription using Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper service
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper will use device: {self.device}")
        
    def load_model(self):
        """Load the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.load_model()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Transcribe with optional language parameter
            options = {}
            if language:
                options['language'] = language
            
            result = self.model.transcribe(audio_path, **options)
            
            logger.info(f"Transcription completed: {result['text'][:50]}...")
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        """
        Transcribe audio with word-level timestamps
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription and timestamps
        """
        if self.model is None:
            self.load_model()
        
        try:
            result = self.model.transcribe(audio_path, word_timestamps=True)
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "words": [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    }
                    for segment in result.get("segments", [])
                    for word in segment.get("words", [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Transcription with timestamps failed: {e}")
            raise
