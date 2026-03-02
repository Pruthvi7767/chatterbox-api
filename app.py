import io
import torch
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer

app = FastAPI()

device = "cpu"

print("Loading VITS model...")
model = VitsModel.from_pretrained("espnet/kan-bayashi_ljspeech_vits").to(device)
tokenizer = AutoTokenizer.from_pretrained("espnet/kan-bayashi_ljspeech_vits")
print("Model loaded.")

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/tts")
def generate_tts(request: TTSRequest):
    inputs = tokenizer(request.text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs).waveform

    buffer = io.BytesIO()
    sf.write(buffer, output.squeeze().cpu().numpy(), 22050, format="WAV")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")