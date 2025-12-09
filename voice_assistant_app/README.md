# Voice Assistant App

A full-stack voice assistant application using Whisper for speech recognition, LLaMA3 for conversation, and SpeechT5 for text-to-speech.

## Architecture

```
User Audio Input → Whisper → LLaMA3 → SpeechT5 → Audio Output
```

## Features

- 🎤 Audio recording from browser
- 🗣️ Speech-to-text with OpenAI Whisper
- 🤖 Conversational AI with LLaMA3
- 🔊 Text-to-speech with Microsoft SpeechT5
- 💬 Conversation state management
- 🌊 Audio streaming to UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

3. Run the application:
```bash
python main.py
```

4. Open your browser to `http://localhost:8000`

## API Endpoints

- `POST /api/voice-chat` - Upload audio and get voice response
- `GET /api/session/{session_id}` - Get conversation history
- `DELETE /api/session/{session_id}` - Clear conversation history

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── services/
│   ├── whisper_service.py  # Whisper transcription
│   ├── llama_service.py    # LLaMA3 conversation
│   ├── speecht5_service.py # SpeechT5 TTS
│   └── state_manager.py    # Conversation state management
├── static/
│   ├── index.html          # Frontend UI
│   └── app.js              # Frontend JavaScript
├── uploads/                # Temporary audio files
└── requirements.txt        # Python dependencies
```

## Configuration

See `.env.example` for all available configuration options.
