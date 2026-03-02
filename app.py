import io
import edge_tts
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    communicate = edge_tts.Communicate(
        request.text,
        voice="en-IN-NeerjaNeural"
    )

    audio_stream = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])

    audio_stream.seek(0)

    return StreamingResponse(audio_stream, media_type="audio/mpeg")