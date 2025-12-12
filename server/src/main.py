from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from server.src.audio.transcribe import transcribe_audio
from server.src.audio.tts import text_to_speech
from server.src.llm.generate import generate_response

origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000"   # sometimes React uses this instead
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    user_text = transcribe_audio(audio_bytes)
    print("User:", user_text)
    bot_text = generate_response(user_text)
    print("Output: ", bot_text)

    path = "response.wav"
    text_to_speech(bot_text, path)

    return FileResponse(path, media_type="audio/wav")
