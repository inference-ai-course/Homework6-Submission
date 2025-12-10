# Development notes and setup instructions

## Model Setup

### 1. Whisper
The Whisper model will be automatically downloaded on first use. Available models:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

### 2. LLaMA3
You need to set up LLaMA3:

#### Option A: Use Hugging Face (Recommended)
1. Get access to LLaMA3 on Hugging Face: https://huggingface.co/meta-llama
2. Install huggingface-cli: `pip install huggingface-hub`
3. Login: `huggingface-cli login`
4. Set in .env: `LLAMA_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct`

#### Option B: Local Model
1. Download LLaMA3 model files
2. Set in .env: `LLAMA_MODEL_PATH=/path/to/local/llama3/model`

### 3. SpeechT5
SpeechT5 will be automatically downloaded from Hugging Face on first use:

1. The model downloads from `microsoft/speecht5_tts` by default
2. HiFi-GAN vocoder downloads from `microsoft/speecht5_hifigan`
3. Speaker embeddings download from the CMU Arctic dataset
4. Set model path in .env if using a custom model

All models are cached locally after the first download.

## Environment Variables

Create a `.env` file with:

```bash
# Application
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Models
WHISPER_MODEL=base
LLAMA_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
SPEECHT5_MODEL_PATH=microsoft/speecht5_tts

# Audio
AUDIO_UPLOAD_DIR=./uploads
MAX_AUDIO_SIZE_MB=10
SUPPORTED_AUDIO_FORMATS=wav,mp3,m4a,ogg

# Conversation
MAX_CONVERSATION_HISTORY=10
SESSION_TIMEOUT_MINUTES=30
```

## Running the Application

### Development Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Run with auto-reload
python main.py
```

### Production Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Endpoints

### POST /api/voice-chat
Upload audio and get voice response
- Request: multipart/form-data with `audio` file and optional `session_id`
- Response: JSON with transcription, response text, and audio URL

### POST /api/voice-chat-stream
Streaming version of voice chat
- Request: Same as /api/voice-chat
- Response: Streaming audio with headers containing metadata

### GET /api/audio/{filename}
Serve generated audio files
- Response: WAV audio file

### GET /api/session/{session_id}
Get conversation history
- Response: JSON with conversation messages

### DELETE /api/session/{session_id}
Clear conversation history
- Response: Success message

### GET /health
Health check endpoint
- Response: Service status

## Testing

Run component tests:
```bash
python test_services.py
```

Test individual services:
```python
from services.whisper_service import WhisperService
service = WhisperService()
service.load_model()
result = service.transcribe("audio.wav")
```

## Troubleshooting

### Out of Memory
- Use smaller Whisper model: `WHISPER_MODEL=tiny`
- Use quantized LLaMA3 model
- Reduce conversation history: `MAX_CONVERSATION_HISTORY=5`

### Slow Response
- Use GPU if available
- Reduce `max_new_tokens` in LLaMA service
- Use smaller models

### Audio Recording Issues
- Check browser permissions for microphone
- Use HTTPS (required for some browsers)
- Test with different audio formats

## Performance Optimization

1. **Model Loading**: Models are loaded on startup (can be slow)
2. **GPU Acceleration**: Use CUDA if available
3. **Caching**: Consider caching common responses
4. **Quantization**: Use quantized models to reduce memory

## Security Considerations

1. Add authentication for production
2. Implement rate limiting
3. Validate audio file types and sizes
4. Sanitize user inputs
5. Use HTTPS in production
6. Add CORS restrictions for production

## Future Enhancements

- [ ] WebSocket support for real-time streaming
- [ ] Multiple language support
- [ ] Voice selection for TTS
- [ ] Conversation export
- [ ] User authentication
- [ ] Analytics and monitoring
- [ ] Model fine-tuning interface
- [ ] Mobile app support
