"""ASR backend: wraps mlx-whisper (Apple Silicon native, fast local
transcription) behind a tiny HTTP API that extension.mjs's lib/backends.mjs
talks to over loopback.

Run via backend/start.sh (not imported directly by the extension -- this is
a standalone process the user starts once and leaves running).
"""
import io
import os
import tempfile

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mlx_whisper

MODEL_REPO = os.environ.get("MONA_ASR_MODEL", "mlx-community/whisper-small-mlx")

app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_REPO}


@app.post("/asr/transcribe")
async def transcribe(request: Request):
    wav_bytes = await request.body()
    if not wav_bytes:
        return JSONResponse({"text": ""})

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        result = mlx_whisper.transcribe(tmp.name, path_or_hf_repo=MODEL_REPO)

    return {"text": result.get("text", "").strip()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MONA_ASR_PORT", 8765)))
