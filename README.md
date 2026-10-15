# mona-voice-canvas

"Hey Mona" -- a voice interface for GitHub Copilot chat. Say the wake word
(or push-to-talk), ask your question out loud, and Copilot's own assistant
for whatever session you have this canvas open in answers -- spoken back to
you. This repo does **not** contain its own conversational AI: the voice
stack only does wake-word detection, speech-to-text, and text-to-speech. The
brain is always the hosting Copilot session.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Canvas iframe (browser, .github/extensions/mona-voice/public/)  │
│                                                                   │
│  mic.js ──► wakeword.js (onnxruntime-web)  ──► app.js            │
│             openWakeWord mel+embedding ONNX     state machine,   │
│             + per-user trained classifier       SSE client,     │
│                                                  streaming audio │
└───────────────────────────┬──────────────────────┬──────────────┘
                             │ loopback HTTP         │ SSE /events
┌────────────────────────────▼──────────────────────▼──────────────┐
│ extension.mjs (Node, .github/extensions/mona-voice/)             │
│  - serves the iframe + models                                    │
│  - POST /utterance: ASR ──► session.sendAndWait() ──► TTS        │
│  - durable wake-word classifier + prefs under $COPILOT_HOME       │
└───────┬───────────────────────────────────────────────┬──────────┘
        │ loopback HTTP                                  │ loopback HTTP
┌───────▼────────────┐                           ┌───────▼────────────┐
│ backend/asr         │                           │ backend/tts         │
│ mlx-whisper          │                           │ VibeVoice-Realtime  │
│ (port 8765)          │                           │ -0.5B (port 8766)   │
└──────────────────────┘                           └──────────────────────┘
```

See `backend/README.md` for backend setup/model choices and honest latency
notes, and `.github/extensions/mona-voice/MODEL-NOTICE.md` for wake-word
model provenance/licensing.

## Quick start

1. **Backends** (one-time setup, downloads model weights -- see
   `backend/README.md` before running):
   ```bash
   cd backend && ./setup.sh && ./start.sh
   ```
2. **Extension**: this is a project-scope canvas extension
   (`.github/extensions/mona-voice/`). In this host app, project extensions
   are registered via `install_extension` pointed at this repo's GitHub
   folder on a pushed branch/commit (plain git-tree presence is not
   auto-discovered here) -- see "Host registration" below if you're
   modifying the extension.
3. Open the `mona-voice` canvas from a Copilot session. Either:
   - Hold **Hold to talk**, speak, release; or
   - Train the wake word (**Train wake word...** panel: a few "Hey Mona"
     takes + background-noise takes), then enable the **Wake word** toggle
     and just say "Hey Mona".
4. Your speech is transcribed locally, sent into the session as a real user
   message, and the session's own reply is spoken back via VibeVoice. Talk
   again while it's speaking to interrupt it (barge-in).

## Host registration (for contributors)

This host app (GitHub Copilot desktop app) does not auto-discover
`.github/extensions/*` the way the Copilot CLI does. The working path found
during development:

```
install_extension({
  url: "https://github.com/<owner>/<repo>/tree/<branch-or-ref>/.github/extensions/mona-voice",
  scope: "project",
})
```

This fetches the folder from GitHub and triggers a real extension reload --
but it enforces a roughly 1MB-per-file / 5MB-total cap, so no binary assets
(ONNX models, model weights) can live under `.github/extensions/mona-voice/`
in this repo. That's why the wake-word ONNX graphs are downloaded at
runtime and cached under `$COPILOT_HOME/extensions/mona-voice/artifacts/`
(see `.github/extensions/mona-voice/lib/models.mjs`) instead of being
vendored in-tree, and why `<branch-or-ref>` must point at a **pushed**
commit -- local-only changes aren't visible to this install path.

## Wake word

Ported from github/github-app PR #12152's Rust wake-word implementation
(melspectrogram → embedding ONNX graphs from openWakeWord v0.5.1, Apache-2.0,
+ a small per-user logistic-regression classifier trained client-side on a
handful of recordings). See `.github/extensions/mona-voice/public/wakeword.js`
for the port and `.github/extensions/mona-voice/MODEL-NOTICE.md` for model
provenance/checksums. Training and classifier storage are entirely local;
nothing leaves your machine.

## Known MVP limitations (read before relying on this)

- **TTS latency, measured on this Mac (M-series, MPS, warm model): 1.9s-3.2s
  wall-clock per reply** (generating 4.7s-5.3s of resulting speech), i.e.
  roughly 1.6x-2.4x faster than real-time, but delivered as one block after
  generation finishes, not streamed token-by-token -- see
  `backend/README.md`'s latency section for the full measurement and how to
  re-run it on your own hardware.
- **`ffmpeg` must be on PATH** (`brew install ffmpeg`) -- `mlx-whisper`
  shells out to it for audio decoding; `backend/setup.sh` now checks for
  this and exits with instructions if it's missing.
- **Wake-word capture is fixed-length (4s), not VAD-endpointed.** After
  "Hey Mona" triggers, it records for a fixed window rather than detecting
  when you stop talking. Fine for short questions, awkward for longer ones.
- **Mid-generation barge-in only stops transport/playback, not the model
  call already in flight** (no true incremental generation yet; verified via
  direct `/tts/stop` HTTP test -- mid-generation cancellation correctly
  suppresses all audio chunk delivery, confirmed empirically, not just by
  reading the code).
- **Voice turns can legitimately take longer than a typical chat reply**
  (tool calls, multi-step reasoning). `session.sendAndWait`'s default 60s
  timeout was too short in real testing and is now configurable via
  `MONA_SESSION_TIMEOUT_MS` (extension.mjs), defaulting to 10 minutes.
- English-only TTS (VibeVoice-Realtime-0.5B's primary focus); ASR language
  support follows whichever Whisper model you configure.
