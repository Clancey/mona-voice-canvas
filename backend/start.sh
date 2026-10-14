#!/usr/bin/env bash
# Starts both backends as background processes on fixed loopback ports
# (8765 ASR, 8766 TTS) matching extension.mjs's lib/backends.mjs defaults.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .run

asr/.venv/bin/python asr/server.py > .run/asr.log 2>&1 &
echo $! > .run/asr.pid
echo "ASR backend starting (pid $(cat .run/asr.pid)), logs: backend/.run/asr.log"

tts/.venv/bin/python tts/server.py > .run/tts.log 2>&1 &
echo $! > .run/tts.pid
echo "TTS backend starting (pid $(cat .run/tts.pid)), logs: backend/.run/tts.log"

echo "Waiting for health checks..."
for i in $(seq 1 30); do
  asr_ok=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/health || echo 000)
  tts_ok=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8766/health || echo 000)
  if [ "$asr_ok" = "200" ] && [ "$tts_ok" = "200" ]; then
    echo "Both backends healthy."
    exit 0
  fi
  sleep 1
done
echo "warning: backends did not report healthy within 30s -- check backend/.run/*.log"
