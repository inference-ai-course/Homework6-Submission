# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment
```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file - IMPORTANT: Set your model paths
nano .env  # or use your favorite editor
```

**Minimum configuration needed:**
```bash
WHISPER_MODEL=base  # Will auto-download
LLAMA_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct  # Requires HuggingFace access
```

### Step 3: Run the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py
```

### Step 4: Open in Browser
Open your browser and navigate to:
```
http://localhost:8000
```

## 🎯 First Time Use

1. **Allow Microphone Access**: Your browser will ask for microphone permission
2. **Click the Microphone Button**: Start recording your message
3. **Click Again to Stop**: Your audio will be processed
4. **Wait for Response**: The assistant will respond with voice and text

## 📋 Prerequisites

### Required
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- Microphone access in browser

### Optional (for better performance)
- NVIDIA GPU with CUDA support
- 8GB+ VRAM for GPU acceleration

## 🔧 Configuration Options

### Model Sizes (in .env)

**Whisper Models:**
- `tiny`: 39M params, fastest, ~1GB memory
- `base`: 74M params, good balance (default)
- `small`: 244M params, better accuracy
- `medium`: 769M params, high accuracy
- `large`: 1550M params, best accuracy

**LLaMA3 Models:**
- `Meta-Llama-3-8B-Instruct`: 8B params, requires ~16GB RAM
- `Meta-Llama-3-70B-Instruct`: 70B params, requires ~140GB RAM (not recommended without GPU cluster)

### Performance Settings

For low-memory systems:
```bash
WHISPER_MODEL=tiny
MAX_CONVERSATION_HISTORY=3
```

For high-performance systems:
```bash
WHISPER_MODEL=large
MAX_CONVERSATION_HISTORY=20
```

## 🐛 Common Issues

### Issue: "No module named 'whisper'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:** Use smaller models or CPU
```bash
# In .env
WHISPER_MODEL=tiny
```

### Issue: "Microphone not working"
**Solution:** 
- Check browser permissions
- Use HTTPS (required by some browsers)
- Try a different browser

### Issue: "LLaMA model not found"
**Solution:** Set up Hugging Face access
```bash
pip install huggingface-hub
huggingface-cli login
# Enter your HF token
```

## 📱 Browser Compatibility

**Recommended:**
- Chrome/Edge 80+
- Firefox 75+
- Safari 14+

**Note:** HTTPS is required for microphone access in most browsers

## 🔐 HuggingFace Setup (for LLaMA3)

1. Create account at https://huggingface.co
2. Request access to LLaMA3 at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Get your access token from https://huggingface.co/settings/tokens
4. Login via CLI:
```bash
pip install huggingface-hub
huggingface-cli login
# Paste your token
```

## 🎨 Customization

### Change Voice Assistant Personality
Edit `services/llama_service.py`:
```python
messages.append({
    "role": "system",
    "content": "You are a friendly assistant..."  # Customize this
})
```

### Adjust Response Length
Edit `services/llama_service.py`:
```python
max_new_tokens=256,  # Increase for longer responses
```

### Change Audio Sample Rate or Speaker
Edit `services/speecht5_service.py`:
```python
self.sample_rate = 16000  # SpeechT5 native sample rate
# To use a different speaker, pass speaker_id parameter:
speecht5_service.synthesize(text, speaker_id=42)  # Use speaker 42
```

## 📊 Testing

Test individual components:
```bash
python test_services.py
```

Test API endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Upload audio (example)
curl -X POST -F "audio=@test.wav" http://localhost:8000/api/voice-chat
```

## 🚢 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker (optional)
```bash
# Create Dockerfile
docker build -t voice-assistant .
docker run -p 8000:8000 voice-assistant
```

## 📞 Support

- Check `DEVELOPMENT.md` for detailed documentation
- Review logs in the terminal for error messages
- Check `uploads/` directory permissions

## 🎉 Next Steps

Once running:
1. Try a simple greeting: "Hello, how are you?"
2. Test conversation context: Ask follow-up questions
3. Check the conversation history
4. Clear and start a new conversation

Enjoy your voice assistant! 🎙️
