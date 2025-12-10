"""
Voice Assistant FastAPI Application
Main application file with API endpoints
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import aiofiles

from services.whisper_service import WhisperService
from services.llama_service import LLaMAService
from services.speecht5_service import SpeechT5Service
from services.state_manager import ConversationStateManager
from services.function_call_handler import FunctionCallHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="Voice assistant with Whisper, LLaMA3, and SpeechT5",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AUDIO_UPLOAD_DIR = os.getenv("AUDIO_UPLOAD_DIR", "./uploads")
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", 10))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")
SPEECHT5_MODEL_PATH = os.getenv("SPEECHT5_MODEL_PATH", "microsoft/speecht5_tts")
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 10))
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 30))

# Create upload directory
Path(AUDIO_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Initialize services
whisper_service = WhisperService(model_name=WHISPER_MODEL)
llama_service = LLaMAService(model_path=LLAMA_MODEL_PATH)
speecht5_service = SpeechT5Service(model_path=SPEECHT5_MODEL_PATH)
state_manager = ConversationStateManager(
    max_history=MAX_CONVERSATION_HISTORY,
    session_timeout_minutes=SESSION_TIMEOUT_MINUTES
)
function_call_handler = FunctionCallHandler()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Voice Assistant API")
    try:
        logger.info("Loading Whisper model...")
        whisper_service.load_model()
        
        logger.info("Loading LLM model...")
        llama_service.load_model()
        
        logger.info("Loading SpeechT5 model...")
        speecht5_service.load_model()
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Voice Assistant API")


@app.get("/")
async def root():
    """Serve the main UI"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "whisper": whisper_service.model is not None,
            "llama": llama_service.model is not None,
            "speecht5": speecht5_service.model is not None
        },
        "active_sessions": state_manager.get_active_sessions_count()
    }


@app.post("/api/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Main voice chat endpoint
    
    Process flow:
    1. Save uploaded audio
    2. Transcribe with Whisper
    3. Generate response with LLaMA3
    4. Synthesize speech with CozyVoice
    5. Return audio response
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"Processing voice chat request for session: {session_id}")
    
    # Validate file size
    content = await audio.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {MAX_AUDIO_SIZE_MB}MB"
        )
    
    try:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, audio_filename)
        
        async with aiofiles.open(audio_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"Audio saved to: {audio_path}")
        
        # Step 1: Transcribe audio with Whisper
        logger.info("Transcribing audio...")
        transcription_result = whisper_service.transcribe(audio_path)
        user_text = transcription_result["text"]
        logger.info(f"Transcription: {user_text}")
        
        # Add user message to conversation history
        state_manager.add_message(session_id, "user", user_text)
        
        # Step 2: Generate response with LLaMA3
        logger.info("Generating response...")
        conversation_history = state_manager.get_conversation_history(session_id)
        llama_response = llama_service.generate_response(
            user_message=user_text,
            conversation_history=conversation_history[:-1]  # Exclude the just-added user message
        )
        
        # Step 3: Process function calls if present
        logger.info("Processing for function calls...")
        assistant_response, was_function_call = function_call_handler.process_llm_response(llama_response)
        
        if was_function_call:
            logger.info(f"Function call executed, result: {assistant_response}")
        else:
            logger.info(f"Normal LLM response: {assistant_response}")

        logger.info(f"Final response: {assistant_response}")
        
        # Add assistant response to conversation history
        # state_manager.add_message(session_id, "assistant", assistant_response)
        
        # Step 4: Synthesize speech with SpeechT5
        logger.info("Synthesizing speech...")
        output_audio_filename = f"{uuid.uuid4()}_response.wav"
        output_audio_path = os.path.join(AUDIO_UPLOAD_DIR, output_audio_filename)
        
        audio_bytes = speecht5_service.synthesize(
            text=assistant_response,
            output_path=output_audio_path
        )
        
        # Cleanup uploaded audio file
        try:
            os.remove(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove uploaded audio: {e}")
        
        # Return audio response with metadata
        return JSONResponse(content={
            "session_id": session_id,
            "transcription": user_text,
            "response_text": assistant_response,
            "audio_url": f"/api/audio/{output_audio_filename}"
        })
        
    except Exception as e:
        logger.error(f"Error processing voice chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice-chat-stream")
async def voice_chat_stream(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Streaming voice chat endpoint
    Returns audio in chunks for real-time playback
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"Processing streaming voice chat for session: {session_id}")
    
    content = await audio.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {MAX_AUDIO_SIZE_MB}MB"
        )
    
    try:
        # Save and transcribe audio
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, audio_filename)
        
        async with aiofiles.open(audio_path, 'wb') as f:
            await f.write(content)
        
        transcription_result = whisper_service.transcribe(audio_path)
        user_text = transcription_result["text"]
        
        state_manager.add_message(session_id, "user", user_text)
        
        # Generate response
        conversation_history = state_manager.get_conversation_history(session_id)
        llama_response = llama_service.generate_response(
            user_message=user_text,
            conversation_history=conversation_history[:-1]
        )
        
        # Process function calls if present
        assistant_response, was_function_call = function_call_handler.process_llm_response(llama_response)
        
        if was_function_call:
            logger.info(f"Function call executed in streaming, result: {assistant_response}")
        else:
            logger.info(f"Normal LLM response in streaming: {assistant_response}")
        
        state_manager.add_message(session_id, "assistant", assistant_response)
        
        # Stream audio synthesis
        def audio_stream():
            for chunk in speecht5_service.synthesize_streaming(assistant_response):
                yield chunk
        
        # Cleanup
        try:
            os.remove(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove uploaded audio: {e}")
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "X-Session-ID": session_id,
                "X-Transcription": user_text,
                "X-Response-Text": assistant_response
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming voice chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = os.path.join(AUDIO_UPLOAD_DIR, filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session"""
    conversation_history = state_manager.get_conversation_history(session_id)
    
    if not conversation_history:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "session_id": session_id,
        "conversation_history": conversation_history,
        "message_count": len(conversation_history)
    }


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    success = state_manager.clear_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session cleared successfully"}


@app.post("/api/cleanup")
async def cleanup_sessions():
    """Cleanup expired sessions"""
    removed_count = state_manager.cleanup_expired_sessions()
    return {"message": f"Removed {removed_count} expired sessions"}


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
