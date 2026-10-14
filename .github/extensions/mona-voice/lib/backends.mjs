// Thin client for the local Python backends. These are standalone
// processes the user starts themselves via backend/start.sh (see
// backend/README.md) -- this extension never spawns heavy ML processes
// itself. If a backend isn't running, callers get a clear, actionable error
// instead of a silent hang.
const ASR_URL = process.env.MONA_ASR_URL || "http://127.0.0.1:8765";
const TTS_URL = process.env.MONA_TTS_URL || "http://127.0.0.1:8766";

async function checkHealth(baseUrl) {
  try {
    const res = await fetch(`${baseUrl}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function backendHealth() {
  const [asrReachable, ttsReachable] = await Promise.all([
    checkHealth(ASR_URL),
    checkHealth(TTS_URL),
  ]);
  return { asrReachable, ttsReachable };
}

/** POSTs raw WAV bytes (16kHz mono PCM16) to the ASR backend, returns transcript text. */
export async function transcribe(wavBuffer, { signal } = {}) {
  const res = await fetch(`${ASR_URL}/asr/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wavBuffer,
    signal,
  });
  if (!res.ok) {
    throw new Error(
      `ASR backend error (${res.status}): ${await res.text().catch(() => "")} -- is backend/start.sh running?`,
    );
  }
  const data = await res.json();
  return data.text ?? "";
}

/**
 * Requests streamed speech for `text` from the TTS backend. The backend
 * responds with a small JSON header line (sampleRate, channels) followed by
 * raw Float32LE PCM frames, chunked as they're generated. We relay the
 * response stream through verbatim so the browser can start playing audio
 * before generation finishes (this is the "barge-in"-friendly path).
 */
export async function speakStream(text, { signal } = {}) {
  const res = await fetch(`${TTS_URL}/tts/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(
      `TTS backend error (${res.status}): ${await res.text().catch(() => "")} -- is backend/start.sh running?`,
    );
  }
  return res.body; // Node.js ReadableStream, piped straight into our own response.
}

export async function stopSpeaking() {
  try {
    await fetch(`${TTS_URL}/tts/stop`, {
      method: "POST",
      signal: AbortSignal.timeout(1500),
    });
  } catch {
    // Backend may already be idle/unreachable -- not fatal for an interrupt.
  }
}
