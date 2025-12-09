# Migration from CozyVoice to SpeechT5

## Summary of Changes

This document outlines all the changes made to switch the voice assistant from CozyVoice to Microsoft's SpeechT5 for text-to-speech functionality.

## Files Created

1. **services/speecht5_service.py**
   - New TTS service using Microsoft SpeechT5
   - Includes HiFi-GAN vocoder for high-quality audio generation
   - Supports configurable speaker embeddings from CMU Arctic dataset
   - Native 16kHz audio output
   - Methods: `load_model()`, `synthesize()`, `synthesize_streaming()`, `get_available_speakers()`, `set_sample_rate()`, `cleanup()`

## Files Modified

### Core Application Files

1. **main.py**
   - Changed import: `CozyVoiceService` → `SpeechT5Service`
   - Updated service initialization: `cozyvoice_service` → `speecht5_service`
   - Changed environment variable: `COZYVOICE_MODEL_PATH` → `SPEECHT5_MODEL_PATH` (default: `microsoft/speecht5_tts`)
   - Updated API description: "CozyVoice" → "SpeechT5"
   - Updated health check endpoint to report "speecht5" status
   - All service calls updated to use `speecht5_service`

2. **requirements.txt**
   - Added: `datasets==2.15.0` (for speaker embeddings)
   - Added: `sentencepiece==0.1.99` (for tokenization support)

3. **test_services.py**
   - Changed import: `CozyVoiceService` → `SpeechT5Service`
   - Updated test function: `test_cozyvoice_service()` → `test_speecht5_service()`

### Configuration Files

4. **.env.example**
   - Changed: `COZYVOICE_MODEL_PATH=path/to/cozyvoice/model` → `SPEECHT5_MODEL_PATH=microsoft/speecht5_tts`

5. **docker-compose.yml**
   - Added environment variable: `SPEECHT5_MODEL_PATH=microsoft/speecht5_tts`

### Documentation Files

6. **README.md**
   - Updated architecture diagram: CozyVoice → SpeechT5
   - Updated features list: "CozyVoice" → "Microsoft SpeechT5"
   - Updated project structure: `cozyvoice_service.py` → `speecht5_service.py`
   - Removed step about downloading CozyVoice model (SpeechT5 auto-downloads)

7. **ARCHITECTURE.md**
   - Updated service layer diagram: CozyVoice Service → SpeechT5 Service
   - Updated service description with SpeechT5 features
   - Updated data flow diagram
   - Updated file structure listing

8. **PROJECT_SUMMARY.md**
   - Updated main architecture description
   - Updated project structure
   - Changed TTS description: "CozyVoice" → "Microsoft SpeechT5 with HiFi-GAN vocoder"
   - Updated data flow (step 9)
   - Changed environment variable documentation
   - Updated implementation notes section
   - Added model download information for SpeechT5 components

9. **DEVELOPMENT.md**
   - Replaced CozyVoice setup section with SpeechT5 section
   - Updated environment variable example
   - Added information about automatic model downloads

10. **QUICKSTART.md**
    - Updated audio configuration section
    - Changed sample rate reference and added speaker selection example

## Key Technical Differences

### CozyVoice → SpeechT5

| Aspect | CozyVoice | SpeechT5 |
|--------|-----------|----------|
| Provider | Unknown/Custom | Microsoft |
| Implementation | Placeholder | Fully functional |
| Model Loading | Manual setup required | Auto-downloads from Hugging Face |
| Audio Quality | Unknown | High quality with HiFi-GAN vocoder |
| Sample Rate | 22050 Hz | 16000 Hz (native) |
| Speaker Voices | Custom | 7306 speaker embeddings available |
| Dependencies | Unknown | transformers, datasets, sentencepiece |
| Model Size | Unknown | ~200MB (model) + ~100MB (vocoder) + ~50MB (embeddings) |

## Benefits of SpeechT5

1. **Production Ready**: Fully implemented, not a placeholder
2. **Open Source**: Microsoft's open-source model on Hugging Face
3. **High Quality**: Uses HiFi-GAN vocoder for natural-sounding speech
4. **Flexible**: 7306+ different speaker voices available
5. **Easy Setup**: Automatic model downloads, no manual configuration needed
6. **Well Maintained**: Part of Hugging Face transformers library
7. **Good Performance**: Efficient inference on CPU and GPU

## Migration Checklist

- [x] Create new SpeechT5 service implementation
- [x] Update requirements.txt with new dependencies
- [x] Update main.py imports and service initialization
- [x] Update all environment variable references
- [x] Update test files
- [x] Update configuration files (.env.example, docker-compose.yml)
- [x] Update all documentation files
- [x] Remove/replace all references to CozyVoice

## Testing

To test the new SpeechT5 integration:

```bash
# Install new dependencies
pip install -r requirements.txt

# Run tests
python test_services.py

# Start the application
python main.py
```

The first run will download the SpeechT5 models automatically (~350MB total).

## Notes

- The old `services/cozyvoice_service.py` file is still present but no longer used
- You may want to delete it: `rm services/cozyvoice_service.py`
- All models are cached in `~/.cache/huggingface/` after first download
- SpeechT5 requires less setup than CozyVoice (no manual model downloads)
