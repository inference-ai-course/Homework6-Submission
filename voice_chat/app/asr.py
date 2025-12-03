# app/asr.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from io import BytesIO
from typing import Union

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device set to use {device}")

def transcribe_audio(audio_input: Union[str, bytes]) -> str:
    """
    Transcribe audio to text
    
    Args:
        audio_input: Audio file path (str) or audio bytes data (bytes)
    
    Returns:
        Transcribed text
    """
    # Determine input type
    if isinstance(audio_input, bytes):
        # Read from bytes
        audio_np, samplerate = sf.read(BytesIO(audio_input))
    elif isinstance(audio_input, str):
        # Read from file path
        audio_np, samplerate = sf.read(audio_input)
    else:
        raise TypeError(f"Expected str or bytes, got {type(audio_input)}")
    
    # Resample to 16kHz
    if samplerate != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=samplerate, target_sr=16000)
    
    # Process audio
    input_features = processor(
        audio_np, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription