# Mona voice backends

Two standalone local Python services the `mona-voice` canvas extension talks
to over loopback HTTP. The extension never spawns these itself -- start them
yourself and leave them running while you use the canvas.

| Service | Port | Model | Purpose |
| --- | --- | --- | --- |
| ASR | 8765 | [mlx-community/whisper-small-mlx](https://huggingface.co/mlx-community/whisper-small-mlx) via `mlx-whisper` | Speech-to-text, Apple Silicon native |
| TTS | 8766 | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | Text-to-speech |

## Setup

Requires Python 3.11 (`~/.local/bin/python3.11` on this machine; override
with `MONA_PYTHON=/path/to/python3.11`) and **`ffmpeg` on PATH** --
`mlx-whisper` shells out to it to decode incoming audio and fails at
request time without it (not installable via pip; `setup.sh` checks for it
and exits with instructions if missing):

```bash
brew install ffmpeg
```

```bash
cd backend
./setup.sh
```

This creates two venvs (`asr/.venv`, `tts/.venv`) and `git clone`s
`microsoft/VibeVoice` into `tts/VibeVoice` (not vendored into this repo --
it's GPL-adjacent-sized and not ours to redistribute). It does **not**
download model weights; those are fetched lazily from Hugging Face Hub on
the first real request to each service, so the first `/asr/transcribe` and
first `/tts/speak` call after a fresh `start.sh` will be slow.

## Running

```bash
./backend/start.sh   # starts both, waits for /health, logs to backend/.run/*.log
./backend/stop.sh     # stops both
```

## Why mlx-whisper over Phi-4-multimodal-instruct for ASR

The original request asked about Microsoft's Phi-4-multimodal-instruct for
ASR. It's a strong multilingual model, but it's a multi-billion-parameter
multimodal LLM meant for batch/offline-style inference, not a purpose-built
streaming ASR model -- there's no documented fast local streaming loop for
it comparable to whisper.cpp/mlx-whisper, and loading a model of that size
for every utterance works against the "fast local back-and-forth" goal this
canvas needs. `mlx-whisper` is Apple Silicon-native, has a mature, simple
Python API, and is the well-trodden choice for this exact use case on a Mac.
If you want to try Phi-4-multimodal-instruct instead, swap `asr/server.py`'s
`transcribe()` body for a `transformers`-based pipeline -- the HTTP contract
(`POST /asr/transcribe` with WAV bytes in, `{"text": ...}` out) doesn't need
to change.

Microsoft also ships `VibeVoice-ASR-BitNet` (CPU-only, ~1.58GB, RTF < 1 on 3+
threads) as a from-the-same-family alternative worth trying later, via
[VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) -- not wired up here
yet since it's a separate C++ inference engine, not a drop-in Python import.

## Why VibeVoice-Realtime-0.5B over VibeVoice-TTS-1.5B for TTS

The user's original request named `VibeVoice-TTS-1.5B`. Microsoft removed
that model's inference code from the repo on 2025-09-05 over misuse
concerns ("Quick Try" is now listed as `Disabled` on their own model table).
`VibeVoice-Realtime-0.5B` is the actively maintained sibling in the same
family, specifically built for low-latency conversational use (their own
docs cite ~200ms first-audio-chunk latency, validated on a Mac M4 Pro) --
a better fit for this use case anyway, not just a fallback.

**Honest caveat on latency, read before assuming this is fast:** this MVP's
`tts/server.py` uses VibeVoice's documented *synchronous* `model.generate()`
API (the same one as their own
`demo/realtime_model_inference_from_file.py`) -- it waits for the entire
reply's audio to finish generating before sending anything back. The ~200ms
figure quoted in VibeVoice's docs applies to their separate token-by-token
streaming websocket demo (`demo/vibevoice_realtime_demo.py` / `web/app.py`),
which incrementally decodes audio as text arrives; that incremental decode
loop is *not* implemented here.

**Measured end-to-end latency on this Mac** (M-series, MPS backend, model
already warm/resident -- first call after process start is slower due to
weight loading): generating a full reply's audio took **1.9s-3.2s
wall-clock for 4.7s-5.3s of resulting speech**, i.e. roughly **1.6x-2.4x
faster than real-time**, not the ~200ms *first-audio-chunk* figure VibeVoice
quotes for their incremental streaming demo (that number isn't comparable
to "whole clip, synchronous" latency measured here). In practice this means:
the user finishes talking, then waits silently for ~2-5s (scaling with
reply length) before Mona starts speaking -- noticeably less immediate than
a true streaming assistant, but usable for a request/response voice loop.
Re-run this benchmark yourself if you change hardware, model size, or the
VibeVoice version:

```bash
curl -s -X POST http://127.0.0.1:8766/tts/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello, this is a latency test."}' \
  -o /tmp/mona-tts-test.raw -w '\ntotal time: %{time_total}s\n'
```

If that's too slow for comfortable conversation, the follow-up work is
porting the websocket demo's incremental decode loop into `tts/server.py`
(the HTTP chunked-transfer plumbing on both the Python and extension side is
already in place for it -- `/tts/speak` already streams PCM chunks to the
browser as they're yielded, it's just that today all chunks are produced
from one finished waveform rather than streamed out mid-generation).

## Barge-in / interrupt

`POST /tts/stop` marks the in-flight generation's ID as cancelled so no
further chunks are sent for it, and the canvas iframe independently aborts
its `fetch()` and stops any already-scheduled Web Audio playback immediately
on `/interrupt`. Because generation itself is a single blocking call (see
caveat above), a *already-generating* request can't be stopped mid-model-call
today -- only mid-transport. This is good enough for "stop the words I'm
hearing right now" but not for "cut off generation compute immediately".
