# Voice Assistant Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                    (Browser - static/index.html)                 │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Microphone  │───▶│  Recording   │───▶│   Upload     │      │
│  │   Access     │    │   Control    │    │   Handler    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Audio     │◀───│  Response    │◀───│  Conversation│      │
│  │   Playback   │    │   Display    │    │   History    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 │ HTTP/REST API
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                        FASTAPI SERVER                             │
│                          (main.py)                                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              API Endpoints                               │    │
│  │                                                          │    │
│  │  POST /api/voice-chat          - Main chat endpoint     │    │
│  │  POST /api/voice-chat-stream   - Streaming chat         │    │
│  │  GET  /api/audio/{filename}    - Serve audio files      │    │
│  │  GET  /api/session/{id}        - Get conversation       │    │
│  │  DELETE /api/session/{id}      - Clear conversation     │    │
│  │  GET  /health                  - Health check           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│                           ┌─────────┐                            │
│                           │ Request │                            │
│                           │ Handler │                            │
│                           └────┬────┘                            │
│                                │                                 │
└────────────────────────────────┼─────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
        ┌───────────────┐  ┌──────────┐  ┌──────────────┐
        │   SERVICES    │  │ SERVICES │  │   SERVICES   │
        └───────────────┘  └──────────┘  └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                        SERVICE LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │         1. AUDIO INPUT PROCESSING                       │     │
│  │                                                          │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │    Whisper Service (whisper_service.py)         │   │     │
│  │  │                                                  │   │     │
│  │  │  • Load OpenAI Whisper model                    │   │     │
│  │  │  • Transcribe audio to text                     │   │     │
│  │  │  • Support multiple languages                   │   │     │
│  │  │  • Extract timestamps (optional)                │   │     │
│  │  │                                                  │   │     │
│  │  │  Input: Audio file (.wav, .mp3, etc.)          │   │     │
│  │  │  Output: Text transcription + metadata          │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────┘     │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │         2. CONVERSATION MANAGEMENT                      │     │
│  │                                                          │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │   State Manager (state_manager.py)              │   │     │
│  │  │                                                  │   │     │
│  │  │  • Manage session lifecycle                     │   │     │
│  │  │  • Store conversation history (in-memory)       │   │     │
│  │  │  • Thread-safe operations                       │   │     │
│  │  │  • Session timeout handling                     │   │     │
│  │  │  • Message history trimming                     │   │     │
│  │  │                                                  │   │     │
│  │  │  Data Structure:                                │   │     │
│  │  │    sessions[session_id] = {                     │   │     │
│  │  │      conversation_history: [...],               │   │     │
│  │  │      created_at: datetime,                      │   │     │
│  │  │      last_activity: datetime                    │   │     │
│  │  │    }                                            │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────┘     │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │         3. RESPONSE GENERATION                          │     │
│  │                                                          │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │    LLaMA Service (llama_service.py)             │   │     │
│  │  │                                                  │   │     │
│  │  │  • Load LLaMA3 model from HuggingFace          │   │     │
│  │  │  • Format conversation context                  │   │     │
│  │  │  • Generate contextual responses                │   │     │
│  │  │  • Support streaming (future)                   │   │     │
│  │  │  • Customizable parameters:                     │   │     │
│  │  │    - temperature                                │   │     │
│  │  │    - max_tokens                                 │   │     │
│  │  │    - top_p                                      │   │     │
│  │  │                                                  │   │     │
│  │  │  Input: User message + conversation history     │   │     │
│  │  │  Output: Generated text response                │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────┘     │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │         4. AUDIO OUTPUT GENERATION                      │     │
│  │                                                          │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │   SpeechT5 Service (speecht5_service.py)        │   │     │
│  │  │                                                  │   │     │
│  │  │  • Load Microsoft SpeechT5 TTS model            │   │     │
│  │  │  • Synthesize speech from text                  │   │     │
│  │  │  • Support streaming output                     │   │     │
│  │  │  • Configurable speaker embeddings              │   │     │
│  │  │  • HiFi-GAN vocoder for high quality            │   │     │
│  │  │                                                  │   │     │
│  │  │  Input: Text response                           │   │     │
│  │  │  Output: Audio data (WAV format, 16kHz)         │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘


## Data Flow Diagram

```
   User Speaks
       │
       ▼
┌──────────────┐
│  JavaScript  │  MediaRecorder API captures audio
│   Frontend   │  
└──────┬───────┘
       │
       │ HTTP POST (multipart/form-data)
       │ audio: Blob (WAV)
       │ session_id: string (optional)
       │
       ▼
┌──────────────────┐
│   FastAPI        │  1. Save audio file to uploads/
│   /api/voice-chat│  2. Generate/retrieve session_id
└──────┬───────────┘
       │
       │ audio_path
       │
       ▼
┌──────────────────┐
│ Whisper Service  │  transcribe(audio_path)
│                  │  → {"text": "...", "language": "en"}
└──────┬───────────┘
       │
       │ transcription_text
       │
       ▼
┌──────────────────┐
│  State Manager   │  add_message(session_id, "user", text)
│                  │  → Update conversation history
└──────┬───────────┘
       │
       │ conversation_history
       │
       ▼
┌──────────────────┐
│  LLaMA Service   │  generate_response(text, history)
│                  │  → Generated response text
└──────┬───────────┘
       │
       │ response_text
       │
       ▼
┌──────────────────┐
│  State Manager   │  add_message(session_id, "assistant", response)
│                  │  → Update conversation history
└──────┬───────────┘
       │
       │ response_text
       │
       ▼
┌──────────────────┐
│ SpeechT5 Service │  synthesize(response_text)
│                  │  → Audio bytes (WAV, 16kHz)
└──────┬───────────┘
       │
       │ audio_bytes
       │
       ▼
┌──────────────────┐
│   FastAPI        │  Save audio to uploads/
│   Response       │  Return JSON with:
│                  │  - session_id
│                  │  - transcription
│                  │  - response_text
│                  │  - audio_url
└──────┬───────────┘
       │
       │ JSON Response
       │
       ▼
┌──────────────────┐
│  JavaScript      │  1. Display transcription
│   Frontend       │  2. Display response text
│                  │  3. Create <audio> element
│                  │  4. Auto-play response
└──────────────────┘
       │
       ▼
   User Hears Response
```

## Component Interactions

### Session Management Flow

```
First Request:
  User → API (no session_id)
         API generates UUID → session_id
         State Manager creates new session
         API returns session_id to user
         User stores session_id

Subsequent Requests:
  User → API (with session_id)
         API retrieves existing session
         State Manager updates history
         API returns updated state
```

### Conversation History Structure

```python
sessions = {
    "uuid-1234": {
        "conversation_history": [
            {
                "role": "user",
                "content": "Hello!",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "role": "assistant",
                "content": "Hi! How can I help?",
                "timestamp": "2024-01-01T10:00:05"
            }
        ],
        "created_at": datetime,
        "last_activity": datetime
    }
}
```

## Technology Stack

### Backend
- **FastAPI**: Web framework for API
- **Uvicorn**: ASGI server
- **OpenAI Whisper**: Speech recognition
- **Transformers**: LLaMA3 integration
- **PyTorch**: ML framework
- **Python-multipart**: File upload handling
- **Aiofiles**: Async file operations

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling
- **Vanilla JavaScript**: Logic
- **MediaRecorder API**: Audio capture
- **Fetch API**: HTTP requests
- **Audio API**: Playback

### Storage
- **In-Memory Dictionary**: Conversation state
- **File System**: Temporary audio files

## Security Considerations

1. **Input Validation**: File size limits, format checking
2. **Session Management**: UUID generation, timeout handling
3. **File Cleanup**: Remove temporary files after processing
4. **CORS**: Configured for cross-origin requests
5. **Error Handling**: Comprehensive try-catch blocks

## Performance Optimizations

1. **Model Loading**: Models loaded once at startup
2. **GPU Acceleration**: Automatic CUDA detection
3. **Async Operations**: File I/O and HTTP
4. **History Trimming**: Limit conversation history size
5. **Session Cleanup**: Remove expired sessions

## Scalability Considerations

### Current Implementation (Single Server)
- In-memory state management
- Local file storage
- Single process handling

### Future Enhancements
- Redis for distributed state
- S3/Object storage for audio files
- Load balancer for multiple instances
- WebSocket for real-time streaming
- Message queue for async processing

## Error Handling

Each service includes comprehensive error handling:

```python
try:
    # Service operation
    result = service.process()
except FileNotFoundError:
    # Handle missing files
except ModelError:
    # Handle model loading/inference errors
except Exception as e:
    # Generic error handling
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

## Monitoring and Logging

- Application logs: INFO level
- Service logs: Component-specific
- Error logs: ERROR level with stack traces
- Health check endpoint: `/health`

## File Structure

```
Assignment App/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── README.md                 # Project overview
├── QUICKSTART.md             # Quick start guide
├── DEVELOPMENT.md            # Development guide
├── ARCHITECTURE.md           # This file
├── setup.sh                  # Setup script
├── test_services.py          # Component tests
├── services/
│   ├── __init__.py
│   ├── whisper_service.py    # Speech-to-text
│   ├── llama_service.py      # Conversation AI
│   ├── speecht5_service.py   # Text-to-speech
│   └── state_manager.py      # Session management
├── static/
│   ├── index.html            # UI
│   └── app.js                # Frontend logic
└── uploads/                  # Temporary audio storage
```

## API Request/Response Examples

### POST /api/voice-chat

**Request:**
```http
POST /api/voice-chat HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="audio"; filename="recording.wav"
Content-Type: audio/wav

[binary audio data]
------WebKitFormBoundary
Content-Disposition: form-data; name="session_id"

uuid-1234-5678
------WebKitFormBoundary--
```

**Response:**
```json
{
  "session_id": "uuid-1234-5678",
  "transcription": "Hello, how are you?",
  "response_text": "I'm doing well, thank you! How can I help you today?",
  "audio_url": "/api/audio/response-uuid.wav",
  "conversation_length": 2
}
```

## Deployment Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ App 1 │ │ App 2 │  (Multiple instances)
└───┬───┘ └──┬────┘
    │        │
    └───┬────┘
        │
┌───────▼────────┐
│  Redis Cache   │  (Shared session state)
└───────┬────────┘
        │
┌───────▼────────┐
│  Object Store  │  (Audio file storage)
└────────────────┘
```

This architecture provides a complete, scalable solution for a voice assistant application with proper separation of concerns and comprehensive functionality.
