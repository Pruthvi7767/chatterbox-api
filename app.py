import io
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from huggingface_hub import snapshot_download
from chatterbox.tts import ChatterboxTTS

app = FastAPI()

print("Starting app...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Downloading model...")
local_dir = snapshot_download(
    repo_id="ResembleAI/chatterbox-turbo"
)

print("Loading model...")
model = ChatterboxTTS.from_local(
    Path(local_dir),
    device
)

model.eval()
print("Model loaded successfully!")

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        wav = model.generate(request.text)

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.cpu(), 24000, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))