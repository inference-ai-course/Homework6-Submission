"""
SpeechT5 Service
Handles text-to-speech using Microsoft's SpeechT5
"""
import logging
import numpy as np
import soundfile as sf
import torch
from typing import Optional
import io
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

logger = logging.getLogger(__name__)


class SpeechT5Service:
    """Service for text-to-speech using SpeechT5"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SpeechT5 service
        
        Args:
            model_path: Path to SpeechT5 model (defaults to microsoft/speecht5_tts)
        """
        self.model_path = model_path or "microsoft/speecht5_tts"
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.sample_rate = 16000  # SpeechT5 uses 16kHz
  
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Initializing SpeechT5 service on device: {self.device}")
        
    def load_model(self):
        """Load the SpeechT5 model, processor, and vocoder"""
        try:
            logger.info(f"Loading SpeechT5 model from: {self.model_path}")
            
            # Load processor
            self.processor = SpeechT5Processor.from_pretrained(self.model_path)
            
            # Load model
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_path)
            self.model.to(self.device)
            
            # Load vocoder (HiFi-GAN)
            logger.info("Loading HiFi-GAN vocoder...")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.vocoder.to(self.device)
            
            # Load default speaker embeddings
            logger.info("Loading speaker embeddings...")
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", 
                split="validation"
            )
            # Use the first speaker embedding as default
            self.speaker_embeddings = torch.tensor(
                embeddings_dataset[7306]["xvector"]
            ).unsqueeze(0).to(self.device)
            
            logger.info("SpeechT5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SpeechT5 model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_id: Optional[int] = None,
        speed: float = 1.0
    ) -> bytes:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            speaker_id: Optional speaker embedding index (0-7305)
            speed: Speech speed multiplier (note: SpeechT5 doesn't natively support speed control)
            
        Returns:
            Audio data as bytes
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Synthesizing speech for: {text[:50]}...")
            
            # Process input text
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use custom speaker embedding if provided
            speaker_emb = self.speaker_embeddings
            if speaker_id is not None:
                try:
                    embeddings_dataset = load_dataset(
                        "Matthijs/cmu-arctic-xvectors", 
                        split="validation"
                    )
                    speaker_emb = torch.tensor(
                        embeddings_dataset[speaker_id]["xvector"]
                    ).unsqueeze(0).to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to load speaker {speaker_id}, using default: {e}")
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_emb,
                    vocoder=self.vocoder
                )
            
            # Convert to numpy
            audio_data = speech.cpu().numpy()
            
            # Apply speed adjustment if needed (crude implementation via resampling)
            if speed != 1.0:
                logger.info(f"Applying speed adjustment: {speed}x")
                # Simple speed adjustment by resampling
                import scipy.signal as signal
                num_samples = int(len(audio_data) / speed)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Convert to bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, self.sample_rate, format='WAV')
            audio_bytes = audio_buffer.getvalue()
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"Audio saved to: {output_path}")
            
            logger.info("Speech synthesis completed")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def synthesize_streaming(self, text: str, chunk_size: int = 1024):
        """
        Synthesize speech with streaming output
        
        Args:
            text: Text to synthesize
            chunk_size: Size of audio chunks to yield
            
        Yields:
            Audio data chunks as bytes
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Streaming synthesis for: {text[:50]}...")
            
            # Generate full audio
            audio_bytes = self.synthesize(text)
            
            # Yield in chunks
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise
    
    def get_available_speakers(self) -> list:
        """
        Get list of available speaker voices
        
        Returns:
            List of speaker IDs (indices into the xvectors dataset)
        """
        # The CMU Arctic xvectors dataset has 7306 speakers (0-7305)
        # Return a subset of common/good quality speakers
        return list(range(0, 100))  # First 100 speakers as examples
    
    def set_sample_rate(self, sample_rate: int):
        """
        Set the audio sample rate
        Note: SpeechT5 natively outputs at 16kHz, setting this will require resampling
        
        Args:
            sample_rate: Sample rate in Hz
        """
        if sample_rate != 16000:
            logger.warning(
                f"SpeechT5 natively outputs at 16kHz. "
                f"Setting to {sample_rate}Hz will require resampling and may affect quality."
            )
        self.sample_rate = sample_rate
        logger.info(f"Sample rate set to: {sample_rate}")
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            del self.processor
            del self.vocoder
            del self.speaker_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SpeechT5 service cleaned up")
