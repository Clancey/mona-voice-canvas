// Faithful JS port of github/github-app's Rust wake-word pipeline
// (crates/wake-word/src/{embedding.rs,detector.rs,classifier.rs}), using the
// same two frozen openWakeWord v0.5.1 ONNX graphs via onnxruntime-web.
// Constants and formulas below are copied 1:1 from that Rust source so this
// JS port is actually equivalent, not just "inspired by" it.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.webgpu.min.mjs";

const EMBEDDING_WINDOW_FRAMES = 76;
const MEL_BINS = 32;
const MEL_STEP_SAMPLES = 1280; // 80ms at 16kHz
const MEL_LOOKBACK_SAMPLES = 480;
export const EMBEDDING_DIM = 96;
const MEL_SCALE = 0.1;
const MEL_OFFSET = 2.0;

export async function loadEmbeddingPipeline(authUrl = (p) => p) {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  const [melBytes, embBytes] = await Promise.all([
    fetch(authUrl("/models/melspectrogram.onnx")).then((r) => r.arrayBuffer()),
    fetch(authUrl("/models/embedding_model.onnx")).then((r) => r.arrayBuffer()),
  ]);
  const melspecSession = await ort.InferenceSession.create(melBytes, { executionProviders: ["wasm"] });
  const embeddingSession = await ort.InferenceSession.create(embBytes, { executionProviders: ["wasm"] });
  return new EmbeddingPipeline(melspecSession, embeddingSession);
}

export class EmbeddingPipeline {
  constructor(melspecSession, embeddingSession) {
    this.melspecSession = melspecSession;
    this.embeddingSession = embeddingSession;
    this.pendingPcm = [];
    this.pcmLookback = [];
    this.melWindow = Array.from({ length: EMBEDDING_WINDOW_FRAMES }, () => new Float32Array(MEL_BINS).fill(1.0));
  }

  reset() {
    this.pendingPcm = [];
    this.pcmLookback = [];
    this.melWindow = Array.from({ length: EMBEDDING_WINDOW_FRAMES }, () => new Float32Array(MEL_BINS).fill(1.0));
  }

  /** Feeds 16kHz mono f32 PCM (range ~[-1,1]); returns zero or more 96-dim embeddings. */
  async pushSamples(samples) {
    for (const s of samples) this.pendingPcm.push(s);
    const embeddings = [];
    while (this.pendingPcm.length >= MEL_STEP_SAMPLES) {
      const chunk = this.pendingPcm.splice(0, MEL_STEP_SAMPLES);
      const melspecInput = [...this.pcmLookback, ...chunk];
      const lookbackStart = Math.max(0, melspecInput.length - MEL_LOOKBACK_SAMPLES);
      this.pcmLookback = melspecInput.slice(lookbackStart);

      const frames = await this.runMelspec(melspecInput);
      for (const f of frames) this.melWindow.push(f);
      if (this.melWindow.length > EMBEDDING_WINDOW_FRAMES) {
        this.melWindow.splice(0, this.melWindow.length - EMBEDDING_WINDOW_FRAMES);
      }
      embeddings.push(await this.runEmbedding());
    }
    return embeddings;
  }

  async runMelspec(pcmChunk) {
    const pcmI16Range = new Float32Array(pcmChunk.length);
    for (let i = 0; i < pcmChunk.length; i++) {
      const s = Math.max(-1, Math.min(1, pcmChunk[i]));
      pcmI16Range[i] = s < 0 ? s * 32768.0 : s * 32767.0;
    }
    const input = new ort.Tensor("float32", pcmI16Range, [1, pcmI16Range.length]);
    const feeds = {};
    feeds[this.melspecSession.inputNames[0]] = input;
    const outputs = await this.melspecSession.run(feeds);
    const out = outputs[this.melspecSession.outputNames[0]];
    const data = out.data;
    const numFrames = out.dims[out.dims.length - 2];
    const frames = [];
    for (let i = 0; i < numFrames; i++) {
      const frame = new Float32Array(MEL_BINS);
      for (let b = 0; b < MEL_BINS; b++) {
        frame[b] = data[i * MEL_BINS + b] * MEL_SCALE + MEL_OFFSET;
      }
      frames.push(frame);
    }
    return frames;
  }

  async runEmbedding() {
    const flat = new Float32Array(EMBEDDING_WINDOW_FRAMES * MEL_BINS);
    this.melWindow.forEach((frame, i) => flat.set(frame, i * MEL_BINS));
    const input = new ort.Tensor("float32", flat, [1, EMBEDDING_WINDOW_FRAMES, MEL_BINS, 1]);
    const feeds = {};
    feeds[this.embeddingSession.inputNames[0]] = input;
    const outputs = await this.embeddingSession.run(feeds);
    const out = outputs[this.embeddingSession.outputNames[0]];
    return Float32Array.from(out.data.slice(0, EMBEDDING_DIM));
  }
}

// ---- Classifier (per-user logistic regression over standardized 96-dim embeddings) ----

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function standardizationStats(positives, negatives) {
  const all = [...positives, ...negatives];
  const n = all.length;
  const mean = new Float32Array(EMBEDDING_DIM);
  for (const sample of all) for (let i = 0; i < EMBEDDING_DIM; i++) mean[i] += sample[i];
  for (let i = 0; i < EMBEDDING_DIM; i++) mean[i] /= n;

  const variance = new Float32Array(EMBEDDING_DIM);
  for (const sample of all) {
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const diff = sample[i] - mean[i];
      variance[i] += diff * diff;
    }
  }
  const std = new Float32Array(EMBEDDING_DIM);
  for (let i = 0; i < EMBEDDING_DIM; i++) std[i] = Math.max(Math.sqrt(variance[i] / n), 1e-3);
  return { mean, std };
}

function standardize(embedding, mean, std) {
  const out = new Float32Array(EMBEDDING_DIM);
  for (let i = 0; i < EMBEDDING_DIM; i++) out[i] = (embedding[i] - mean[i]) / std[i];
  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) sum += a[i] * b[i];
  return sum;
}

const TRAIN_CONFIG = { learningRate: 0.1, epochs: 300, l2: 0.01 };

/** Trains a classifier from arrays of raw (unstandardized) 96-dim embeddings. */
export function trainClassifier(positives, negatives, config = TRAIN_CONFIG) {
  if (!positives.length || !negatives.length) return null;

  const { mean, std } = standardizationStats(positives, negatives);
  const positiveWeight = 0.5 / positives.length;
  const negativeWeight = 0.5 / negatives.length;

  const examples = [
    ...positives.map((p) => [standardize(p, mean, std), 1.0, positiveWeight]),
    ...negatives.map((n) => [standardize(n, mean, std), 0.0, negativeWeight]),
  ];

  let weights = new Float32Array(EMBEDDING_DIM);
  let bias = 0;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    const gradW = new Float32Array(EMBEDDING_DIM);
    let gradB = 0;
    for (const [features, label, weight] of examples) {
      const z = dot(weights, features) + bias;
      const pred = sigmoid(z);
      const error = (pred - label) * weight;
      for (let i = 0; i < EMBEDDING_DIM; i++) gradW[i] += error * features[i];
      gradB += error;
    }
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const reg = config.l2 * weights[i];
      weights[i] -= config.learningRate * (gradW[i] + reg);
    }
    bias -= config.learningRate * gradB;
  }

  return {
    mean: Array.from(mean),
    std: Array.from(std),
    weights: Array.from(weights),
    bias,
    version: 1,
  };
}

export function scoreEmbedding(classifier, embedding) {
  const features = standardize(embedding, classifier.mean, classifier.std);
  return sigmoid(dot(classifier.weights, features) + classifier.bias);
}

// ---- Detector (smoothing + consecutive-run threshold + cooldown) ----

const DEFAULT_DETECTOR_CONFIG = {
  threshold: 0.85,
  smoothingWindow: 8,
  requiredFramesOverThreshold: 3,
  cooldownFrames: 30,
};

export class WakeWordDetector {
  constructor(classifier, config = DEFAULT_DETECTOR_CONFIG) {
    this.classifier = classifier;
    this.config = config;
    this.recentScores = [];
    this.framesSinceTrigger = Number.MAX_SAFE_INTEGER / 2;
    this.listening = false;
  }

  setListening(listening) {
    this.listening = listening;
  }

  resetSmoothing() {
    this.recentScores = [];
  }

  /** Feeds one embedding; returns true if this frame triggers a detection. */
  pushEmbedding(embedding) {
    this.framesSinceTrigger += 1;
    if (!this.listening) return false;

    const score = scoreEmbedding(this.classifier, embedding);
    this.recentScores.push(score);
    if (this.recentScores.length > this.config.smoothingWindow) this.recentScores.shift();
    if (this.recentScores.length < this.config.smoothingWindow) return false;

    let run = 0;
    let longestRun = 0;
    for (const s of this.recentScores) {
      if (s >= this.config.threshold) {
        run += 1;
        longestRun = Math.max(longestRun, run);
      } else {
        run = 0;
      }
    }

    if (longestRun >= this.config.requiredFramesOverThreshold && this.framesSinceTrigger >= this.config.cooldownFrames) {
      this.framesSinceTrigger = 0;
      this.listening = false; // caller re-enables after handling the trigger
      return true;
    }
    return false;
  }
}
