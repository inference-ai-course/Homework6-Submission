from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import whisper
import sys
import os
import shutil
import logging
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import util.util as util

logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    print('app started...')
    audio_dir = './static/audio/'
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    os.mkdir(audio_dir)


@app.get("/") 
async def root():
    """Serve the main UI"""
    return FileResponse("static/index.html")

@app.post("/api/voice-query/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # ASR
    user_text = transcribe_audio(audio_bytes)
    logger.info(f'***user_text={user_text}')

    # LLM 
    bot_text = util.generate_response(user_text)
    logger.info(f'***bot_text={bot_text}')

    # Function calling
    is_function, function_text = util.is_function(bot_text)
    if is_function:
        logger.info(f'***Routing to call_function...')
        result = util.call_function(function_text)
    else:
        result = bot_text

    # TTS
    audio_path = synthesize_speech(result)

    return JSONResponse(content={
            "user_text": user_text,
            "response_text": result,
            "audio_path": audio_path
        })


asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    temp_audio_file = "temp.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe(temp_audio_file, fp16=False)
    try:
        os.remove(temp_audio_file)
    except Exception as e:
        logger.error(f"Failed to remove uploaded audio: {e}")
    return result["text"]


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

def text_generator(bot_text):
    if bot_text:
        for sentence in bot_text.split('.'):
            yield sentence
    else:
        yield bot_text


count = 0
MAX = 10 # keep 10 saved audio output
def synthesize_speech(bot_text):
    bot_text = ' '.join(bot_text.split()[:50])
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    shots = cosyvoice.inference_zero_shot(text_generator(bot_text), '', prompt_speech_16k, stream=False)
    global count
    n = count % MAX
    count += 1
    for i, shot in enumerate(shots):
        torchaudio.save('./static/audio/zero_shot_{}.wav'.format(n), shot['tts_speech'], cosyvoice.sample_rate)
    return f'static/audio/zero_shot_{n}.wav'