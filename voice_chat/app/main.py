# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import inflect
import re
from pathlib import Path
import traceback
from app.asr import transcribe_audio
from app.tts import synthesize_speech
from app.llm import generate_response, route_llm_output
from app.tools import save_interaction, FUNCTION_MAP

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Directories ----------------
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)  # 自动创建 static 目录
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------- Helper ----------------
def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[JSON Parse Error] {e}")
        return None

p = inflect.engine()

def normalize_numbers(text: str) -> str:
    def replacer(match):
        num = match.group()
        return p.number_to_words(num)
    return re.sub(r"\d+", replacer, text)

def tts_background(text: str, output_path: Path):
    """生成语音文件"""
    try:
        synthesize_speech(text, str(output_path))
        print(f"[TTS] Audio saved → {output_path}")
    except Exception as e:
        print(f"[TTS Error] {e}")
        # fallback: 生成静音文件
        import numpy as np
        import soundfile as sf
        silent_audio = np.zeros(16000)
        sf.write(output_path, silent_audio, samplerate=16000)

# ---------------- Routes ----------------
@app.get("/")
def read_root():
    return FileResponse("./index.html", media_type="text/html")

@app.post("/chat/")
async def chat_endpoint(audio: UploadFile = File(...)):
    input_path = None

    try:
        # ---------------- Save uploaded audio ----------------
        input_path = TEMP_DIR / f"input_{uuid.uuid4().hex}.wav"
        content = await audio.read()
        input_path.write_bytes(content)
        print(f"[CHAT] Saved audio → {input_path}")

        # ---------------- ASR ----------------
        print("[CHAT] Transcribing...")
        user_text = transcribe_audio(str(input_path)).strip()
        print(f"[CHAT] User text: {user_text}")

        if not user_text:
            return {"user_text": "", "bot_response": "Sorry, I could not understand the audio.", "audio_url": ""}

        # ---------------- LLM ----------------
        print("[CHAT] Generating LLM response...")
        raw_llm_output = generate_response(user_text)
        func_output = None

        # ---------------- 工具调用解析 ----------------
        parsed = try_parse_json(raw_llm_output)

        if parsed and isinstance(parsed, dict):
            func_name = parsed.get("function")
            args = parsed.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            if func_name and func_name in FUNCTION_MAP:
                try:
                    func_output = FUNCTION_MAP[func_name](**args)
                    print(f"[TOOL] Executed {func_name}, result: {func_output}")
                except Exception as e:
                    func_output = {"error": f"工具执行出错: {e}"}

        # ---------------- Route LLM output ----------------
        bot_text = route_llm_output(raw_llm_output)

        if func_output is not None:
            if isinstance(func_output, (int, float)):
                bot_text = f"{bot_text} The answer is {func_output}."
            elif isinstance(func_output, str):
                if "error" in func_output.lower():
                    bot_text = f"{bot_text} {func_output}"
                else:
                    bot_text = f"{bot_text} The result is {func_output}."
            elif isinstance(func_output, dict):
                if func_output.get("success"):
                    bot_text = f"{bot_text} The answer is {func_output['result']}."
                else:
                    bot_text = f"{bot_text} {func_output.get('error', 'Unknown error')}."

        print(f"[CHAT] Bot response (before normalization): {bot_text}")

        bot_text = normalize_numbers(bot_text)
        print(f"[CHAT] Bot response (after normalization): {bot_text}")

        # ---------------- Save JSON log ----------------
        save_interaction(user_text, raw_llm_output, bot_text, func_output)

        # ---------------- TTS ----------------
        output_filename = f"output_{uuid.uuid4().hex}.wav"
        output_path = STATIC_DIR / output_filename
        tts_background(bot_text, output_path)

        # ---------------- 返回 URL ----------------
        audio_url = f"/static/{output_filename}"
        return {"user_text": user_text, "bot_response": bot_text, "audio_url": audio_url}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Server error: {str(e)}"}

    finally:
        try:
            if input_path and input_path.exists():
                input_path.unlink()
        except:
            pass

@app.get("/health")
def health_check():
    return {"status": "healthy"}
