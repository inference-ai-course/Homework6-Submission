import pyttsx3
import tempfile
import shutil

def text_to_speech(text: str, path: str):
    engine = pyttsx3.init()

    temp_file = tempfile.mktemp(suffix=".wav")
    engine.save_to_file(text, temp_file)
    engine.runAndWait()

    shutil.move(temp_file, path)