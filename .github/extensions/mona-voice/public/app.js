import { MicStream, encodeWav } from "/mic.js";
import { loadEmbeddingPipeline, trainClassifier, WakeWordDetector } from "/wakeword.js";

// This canvas's loopback server requires a per-instance capability token
// (any local process that discovers the ephemeral port could otherwise
// inject real Copilot user turns or wipe wake-word training). The host
// embeds it in the URL it loads this page from; propagate it onto every
// request this page makes back to that same server.
const AUTH_TOKEN = new URLSearchParams(location.search).get("token") || "";

function authUrl(path) {
  const url = new URL(path, location.origin);
  if (AUTH_TOKEN) url.searchParams.set("token", AUTH_TOKEN);
  return url.toString();
}

function authFetch(path, opts) {
  return fetch(authUrl(path), opts);
}

const $ = (id) => document.getElementById(id);
const phaseBadge = $("phase-badge");
const backendWarning = $("backend-warning");
const pttButton = $("ptt-button");
const wakeToggle = $("wake-word-toggle");
const trainButton = $("train-button");
const interruptButton = $("interrupt-button");
const userBubble = $("user-bubble");
const assistantBubble = $("assistant-bubble");
const trainPanel = $("train-panel");
const footerStatus = $("footer-status");

function setPhase(phase) {
  phaseBadge.textContent = phase;
  phaseBadge.dataset.phase = phase;
  interruptButton.classList.toggle("hidden", phase !== "speaking");
}

function showTranscript(text) {
  if (!text) {
    userBubble.classList.add("hidden");
    return;
  }
  userBubble.textContent = text;
  userBubble.classList.remove("hidden");
}

function showAssistantReply(text) {
  if (!text) {
    assistantBubble.classList.add("hidden");
    return;
  }
  assistantBubble.textContent = text;
  assistantBubble.classList.remove("hidden");
}

// ---- SSE: drives all passive state updates pushed from extension.mjs ----
const events = new EventSource(authUrl("/events"));
events.onmessage = (ev) => {
  const data = JSON.parse(ev.data);
  if (data.type === "state") {
    setPhase(data.phase);
    if (data.transcript !== undefined) showTranscript(data.transcript);
    if (data.assistantReply !== undefined) showAssistantReply(data.assistantReply);
    if (data.phase === "error") footerStatus.textContent = `Error: ${data.error}`;
  }
};

// ---- Streaming PCM playback (gapless scheduling; stoppable for barge-in) ----
class StreamPlayer {
  constructor() {
    this.ctx = new AudioContext();
    this.nextStartTime = 0;
    this.nodes = [];
    this.sampleRate = 24000;
  }

  reset(sampleRate) {
    this.stop();
    this.sampleRate = sampleRate || 24000;
    this.nextStartTime = this.ctx.currentTime;
  }

  enqueue(float32) {
    const buffer = this.ctx.createBuffer(1, float32.length, this.sampleRate);
    buffer.copyToChannel(float32, 0);
    const node = this.ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(this.ctx.destination);
    const startAt = Math.max(this.nextStartTime, this.ctx.currentTime);
    node.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.nodes.push(node);
    node.onended = () => {
      this.nodes = this.nodes.filter((n) => n !== node);
    };
  }

  stop() {
    for (const node of this.nodes) {
      try {
        node.stop();
      } catch {
        /* already ended */
      }
    }
    this.nodes = [];
  }
}
const player = new StreamPlayer();

async function playUtteranceResponse(res) {
  const reader = res.body.getReader();
  let headerParsed = false;
  // `carry` holds any bytes not yet consumed: pre-header bytes while waiting
  // for the newline, and post-header PCM bytes left over because the chunk
  // length wasn't a multiple of 4 (a Float32 sample). Every chunk boundary
  // is an arbitrary network/TCP split, not an audio-sample boundary, so this
  // carry buffer must persist for the whole stream -- dropping the leftover
  // bytes (as an earlier version of this code did) corrupts/desyncs audio.
  let carry = new Uint8Array(0);

  function appendCarry(bytes) {
    const combined = new Uint8Array(carry.length + bytes.length);
    combined.set(carry, 0);
    combined.set(bytes, carry.length);
    return combined;
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    let combined = appendCarry(value);

    if (!headerParsed) {
      const text = new TextDecoder().decode(combined);
      const newlineIdx = text.indexOf("\n");
      if (newlineIdx === -1) {
        carry = combined;
        continue;
      }
      const header = JSON.parse(text.slice(0, newlineIdx));
      player.reset(header.sampleRate);
      const headerByteLen = new TextEncoder().encode(text.slice(0, newlineIdx + 1)).length;
      combined = combined.slice(headerByteLen); // fresh buffer, byteOffset 0
      headerParsed = true;
    }

    const alignedLen = combined.length - (combined.length % 4);
    if (alignedLen === 0) {
      carry = combined;
      continue;
    }
    const aligned = combined.slice(0, alignedLen); // fresh buffer, byteOffset 0 -- safe for Float32Array
    carry = combined.slice(alignedLen); // leftover 0-3 bytes, carried to next read
    const float32 = new Float32Array(aligned.buffer, 0, alignedLen / 4);
    player.enqueue(float32.slice());
  }
}

// ---- Recording + utterance round-trip (shared by push-to-talk and wake word) ----
let activeMic = null;
let recordingBuffer = [];
let abortController = null;

async function sendUtterance() {
  const wav = encodeWav(Float32Array.from(recordingBuffer));
  recordingBuffer = [];
  abortController = new AbortController();
  try {
    const res = await authFetch("/utterance", { method: "POST", body: wav, signal: abortController.signal });
    if (res.status === 204) {
      setPhase("listening");
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      footerStatus.textContent = `Error: ${err.error}`;
      setPhase("error");
      return;
    }
    await playUtteranceResponse(res);
  } catch (err) {
    if (err.name !== "AbortError") {
      footerStatus.textContent = `Error: ${err.message}`;
    }
  }
}

async function interrupt() {
  player.stop();
  abortController?.abort();
  await authFetch("/interrupt", { method: "POST" }).catch(() => {});
}
interruptButton.onclick = interrupt;

// ---- Push-to-talk ----
let pttMic = null;
async function startPushToTalk() {
  await interrupt();
  recordingBuffer = [];
  pttButton.classList.add("active");
  setPhase("recording");
  pttMic = new MicStream({ onFrame: (frame) => recordingBuffer.push(...frame) });
  await pttMic.start();
}
function stopPushToTalk() {
  pttButton.classList.remove("active");
  pttMic?.stop();
  pttMic = null;
  if (recordingBuffer.length > 1600) {
    // At least 100ms of audio -- avoid firing on accidental taps.
    sendUtterance();
  } else {
    setPhase("listening");
  }
}
pttButton.addEventListener("mousedown", startPushToTalk);
pttButton.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startPushToTalk();
});
pttButton.addEventListener("mouseup", stopPushToTalk);
pttButton.addEventListener("mouseleave", () => pttMic && stopPushToTalk());
pttButton.addEventListener("touchend", stopPushToTalk);

// ---- Wake word ----
let wakeWordState = null; // { mic, detector, pipeline, armed }
const WAKE_RECORD_MS = 4000; // fixed-length capture after trigger; no VAD endpointing in this MVP.

async function enableWakeWord() {
  footerStatus.textContent = "Loading wake-word models…";
  const pipeline = await loadEmbeddingPipeline(authUrl);
  const classifierRes = await authFetch("/wakeword/classifier").then((r) => r.json());
  if (!classifierRes.trained) {
    footerStatus.textContent = "No trained wake-word classifier yet -- use Train wake word first.";
    wakeToggle.checked = false;
    return;
  }
  const detector = new WakeWordDetector(classifierRes.classifier);
  detector.setListening(true);

  const mic = new MicStream({
    onFrame: async (frame) => {
      if (!wakeWordState) return;
      if (wakeWordState.capturing) {
        recordingBuffer.push(...frame);
        return;
      }
      const embeddings = await pipeline.pushSamples(frame);
      for (const emb of embeddings) {
        if (detector.pushEmbedding(emb)) {
          wakeWordState.capturing = true;
          recordingBuffer = [];
          setPhase("wake-detected");
          setTimeout(async () => {
            setPhase("recording");
            await new Promise((r) => setTimeout(r, WAKE_RECORD_MS));
            wakeWordState.capturing = false;
            detector.setListening(true);
            if (recordingBuffer.length > 1600) await sendUtterance();
            else setPhase("listening");
          }, 50);
        }
      }
    },
  });
  await mic.start();
  wakeWordState = { mic, detector, pipeline, capturing: false };
  setPhase("listening");
  footerStatus.textContent = "Wake word active -- say \"Hey Mona\".";
}

function disableWakeWord() {
  wakeWordState?.mic.stop();
  wakeWordState = null;
  setPhase("idle");
  footerStatus.textContent = "Wake word off -- use push-to-talk.";
}

wakeToggle.addEventListener("change", () => {
  if (wakeToggle.checked) enableWakeWord().catch((err) => {
    footerStatus.textContent = `Wake word failed to start: ${err.message}`;
    wakeToggle.checked = false;
  });
  else disableWakeWord();
});

// ---- Training UI ----
let trainPipeline = null;
let positiveEmbeddings = [];
let negativeEmbeddings = [];

async function recordTake(seconds, onEmbedding) {
  if (!trainPipeline) trainPipeline = await loadEmbeddingPipeline(authUrl);
  trainPipeline.reset();
  const collected = [];
  const mic = new MicStream({
    onFrame: async (frame) => {
      const embeddings = await trainPipeline.pushSamples(frame);
      collected.push(...embeddings);
    },
  });
  await mic.start();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  mic.stop();
  onEmbedding(collected);
}

trainButton.onclick = () => trainPanel.classList.remove("hidden");
$("close-train").onclick = () => trainPanel.classList.add("hidden");

$("record-positive").onclick = async () => {
  $("train-status").textContent = "Recording… say \"Hey Mona\" now.";
  await recordTake(2, (embeddings) => {
    positiveEmbeddings.push(...embeddings);
    $("positive-count").textContent = `${positiveEmbeddings.length} embeddings`;
    $("train-status").textContent = "Got it.";
    updateTrainButton();
  });
};

$("record-negative").onclick = async () => {
  $("train-status").textContent = "Recording 5s of background noise…";
  await recordTake(5, (embeddings) => {
    negativeEmbeddings.push(...embeddings);
    $("negative-count").textContent = `${negativeEmbeddings.length} embeddings`;
    $("train-status").textContent = "Got it.";
    updateTrainButton();
  });
};

function updateTrainButton() {
  $("run-training").disabled = positiveEmbeddings.length < 3 || negativeEmbeddings.length < 10;
}

$("run-training").onclick = async () => {
  $("train-status").textContent = "Training…";
  const classifier = trainClassifier(positiveEmbeddings, negativeEmbeddings);
  await authFetch("/wakeword/classifier", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(classifier),
  });
  $("train-status").textContent = "Trained and saved. You can enable the wake-word toggle now.";
};

$("reset-training").onclick = async () => {
  positiveEmbeddings = [];
  negativeEmbeddings = [];
  $("positive-count").textContent = "0 takes";
  $("negative-count").textContent = "0 takes";
  updateTrainButton();
  await authFetch("/wakeword/reset", { method: "POST" });
  $("train-status").textContent = "Reset.";
};

// ---- Startup: surface backend reachability honestly ----
async function checkHealth() {
  const health = await authFetch("/health").then((r) => r.json()).catch(() => ({ asrReachable: false, ttsReachable: false }));
  backendWarning.classList.toggle("hidden", health.asrReachable && health.ttsReachable);
  footerStatus.textContent = health.asrReachable && health.ttsReachable ? "Ready." : "Backend(s) not running.";
}
checkHealth();
setInterval(checkHealth, 5000);
setPhase("idle");
