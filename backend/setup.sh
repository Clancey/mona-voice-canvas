#!/usr/bin/env bash
# Sets up Python venvs for both local backends. Downloads model weights
# (several GB total) and vendors the VibeVoice repo via git clone -- run
# this only after you've confirmed that's what you want (it is NOT run
# automatically by anything else in this repo).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${MONA_PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Set MONA_PYTHON to a Python 3.11 interpreter." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found on PATH. mlx-whisper shells out to ffmpeg" >&2
  echo "to decode audio and will fail at request time without it (this" >&2
  echo "isn't installable via pip). Install it first, e.g.:" >&2
  echo "    brew install ffmpeg" >&2
  exit 1
fi

echo "== ASR backend (mlx-whisper) =="
cd asr
"$PYTHON" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
cd ..

echo "== TTS backend (VibeVoice-Realtime-0.5B) =="
cd tts
if [ ! -d VibeVoice ]; then
  git clone https://github.com/microsoft/VibeVoice.git VibeVoice
fi
"$PYTHON" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e "./VibeVoice[streamingtts]"
cd ..

echo
echo "Setup complete. Model weights for both backends download lazily on"
echo "first request (mlx-whisper on first /asr/transcribe call, VibeVoice on"
echo "first /tts/speak call) via Hugging Face Hub -- expect the FIRST"
echo "request after start.sh to be slow while that download happens."
echo "Run ./start.sh to launch both servers."
