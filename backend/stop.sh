#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for name in asr tts; do
  if [ -f ".run/$name.pid" ]; then
    pid=$(cat ".run/$name.pid")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "Stopped $name backend (pid $pid)"
    fi
    rm -f ".run/$name.pid"
  fi
done
