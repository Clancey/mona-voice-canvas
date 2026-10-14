"""TTS backend: wraps microsoft/VibeVoice-Realtime-0.5B behind a tiny HTTP
API. Requires the VibeVoice repo cloned into ./VibeVoice (see
backend/setup.sh) since it isn't published as a pip package.

HONEST LATENCY NOTE: this MVP uses VibeVoiceStreamingForConditionalGenerationInference's
documented *synchronous* generate() API (the same one used by VibeVoice's own
`demo/realtime_model_inference_from_file.py`), which produces the complete
reply's audio in one blocking call before we send anything back. VibeVoice's
~200ms "first audible speech" figure applies to their separate websocket demo
(web/app.py), which streams token-by-token with incremental decode hooks this
simpler script does not implement. Real per-reply latency on this machine has
NOT been benchmarked until this is actually run end-to-end -- do not assume
it is low. Because of this, true mid-generation barge-in isn't possible
either; /tts/stop only prevents an already-finished clip from starting
playback and signals clients to stop (actual interruption of in-flight audio
happens client-side by aborting playback immediately).
"""
import io
import json
import os
import sys
import threading
import wave

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

VIBEVOICE_DIR = os.path.join(os.path.dirname(__file__), "VibeVoice")
if os.path.isdir(VIBEVOICE_DIR):
    sys.path.insert(0, VIBEVOICE_DIR)

from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # noqa: E402
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (  # noqa: E402
    VibeVoiceStreamingProcessor,
)
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402

MODEL_PATH = os.environ.get("MONA_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
SPEAKER_NAME = os.environ.get("MONA_TTS_SPEAKER", "Wayne")
SAMPLE_RATE = 24000

app = FastAPI()
_lock = threading.Lock()
_generation_id = 0
_cancelled_ids = set()


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Model:
    """Lazily loaded singleton -- model load (weights download on first run,
    then disk load) is slow, so we don't block server startup on it."""

    processor = None
    model = None
    voice_sample = None
    device = None

    @classmethod
    def ensure_loaded(cls):
        if cls.model is not None:
            return
        cls.device = _pick_device()
        dtype = torch.float32 if cls.device in ("mps", "cpu") else torch.bfloat16
        attn_impl = "sdpa" if cls.device in ("mps", "cpu") else "flash_attention_2"

        cls.processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)
        cls.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=None if cls.device == "mps" else cls.device,
        )
        if cls.device == "mps":
            cls.model.to("mps")
        cls.model.eval()
        cls.model.set_ddpm_inference_steps(num_steps=5)

        voices_dir = os.path.join(VIBEVOICE_DIR, "demo", "voices", "streaming_model")
        voice_path = os.path.join(voices_dir, f"{SPEAKER_NAME}.pt")
        if not os.path.isfile(voice_path):
            raise RuntimeError(
                f"voice preset '{SPEAKER_NAME}' not found at {voice_path} -- "
                "run backend/setup.sh to clone VibeVoice with its bundled voices."
            )
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            cls.voice_sample = torch.load(voice_path, map_location=cls.device, weights_only=True)


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_PATH, "loaded": _Model.model is not None}


class SpeakRequest(BaseModel):
    text: str


def _generate_pcm(text: str) -> np.ndarray:
    _Model.ensure_loaded()
    inputs = _Model.processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=_Model.voice_sample,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(_Model.device)

    import copy

    outputs = _Model.model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=_Model.processor.tokenizer,
        generation_config={"do_sample": False},
        all_prefilled_outputs=copy.deepcopy(_Model.voice_sample),
    )
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        return np.zeros(0, dtype=np.float32)
    audio = outputs.speech_outputs[0]
    if torch.is_tensor(audio):
        audio = audio.detach().to("cpu").float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _chunked_response(pcm: np.ndarray, generation_id: int):
    header = json.dumps({"sampleRate": SAMPLE_RATE, "channels": 1}) + "\n"
    yield header.encode("utf-8")
    chunk_samples = SAMPLE_RATE // 5  # ~200ms per transport chunk
    for start in range(0, len(pcm), chunk_samples):
        if generation_id in _cancelled_ids:
            break
        yield pcm[start : start + chunk_samples].tobytes()


@app.post("/tts/speak")
async def speak(req: SpeakRequest):
    global _generation_id
    with _lock:
        _generation_id += 1
        gen_id = _generation_id
    pcm = _generate_pcm(req.text)
    return StreamingResponse(_chunked_response(pcm, gen_id), media_type="application/octet-stream")


@app.post("/tts/stop")
async def stop():
    _cancelled_ids.add(_generation_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MONA_TTS_PORT", 8766)))
