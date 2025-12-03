# app/tts.py
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TTS] Using device: {device}")

# Global model loading (load once for efficiency)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model = model.to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
vocoder = vocoder.to(device)

print("[TTS] Models loaded successfully")


def synthesize_speech(text: str, output_file: str) -> str:
    """
    Generate speech using SpeechT5 with female voice and moderate speed
    
    Args:
        text: Text to convert to speech
        output_file: Output WAV file path
    
    Returns:
        Output file path
    """
    try:
        # Limit text length
        text = text[:500] if len(text) > 500 else text
        
        # Process input text
        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Create speaker embeddings on the correct device
        # Using a fixed female speaker embedding (512 dims)
        speaker_embeddings = torch.FloatTensor([
            0.1, -0.2, 0.3, -0.15, 0.25, -0.1, 0.2, -0.25, 0.15, -0.3,
            -0.1, 0.2, -0.25, 0.15, -0.3, 0.1, -0.2, 0.3, -0.15, 0.25,
            0.05, -0.15, 0.2, -0.1, 0.25, -0.2, 0.15, -0.25, 0.1, -0.3,
            -0.15, 0.1, -0.2, 0.15, -0.25, 0.2, -0.1, 0.3, -0.05, 0.2,
        ] * 13)[:512]  # Ensure exactly 512 dimensions
        speaker_embeddings = speaker_embeddings.unsqueeze(0).to(device)
        
        # Generate speech
        with torch.no_grad():
            speech = model.generate_speech(
                input_ids, 
                speaker_embeddings,
                vocoder=vocoder
            )
        
        # Convert to numpy array
        speech_np = speech.cpu().numpy()
        
        # Ensure audio is in correct format
        if speech_np.ndim > 1:
            speech_np = speech_np.squeeze()
        
        # Normalize audio
        max_val = np.max(np.abs(speech_np))
        if max_val > 1.0:
            speech_np = speech_np / max_val
        
        # Slow down speech by 15%
        speech_slow = np.interp(
            np.linspace(0, len(speech_np) - 1, int(len(speech_np) * 1.15)),
            np.arange(len(speech_np)),
            speech_np
        )
        
        # Save as WAV file
        sf.write(output_file, speech_slow, samplerate=16000)
        print(f"[TTS] Audio saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"[TTS Error] {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create silent audio
        silent_audio = np.zeros(16000)
        sf.write(output_file, silent_audio, samplerate=16000)
        raise Exception(f"TTS generation failed: {str(e)}")