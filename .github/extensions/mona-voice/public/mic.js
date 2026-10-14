// Microphone capture: gets the browser's native mic stream (typically 44.1
// or 48kHz Float32) and resamples it to 16kHz mono, which is what every
// stage downstream (wake word, ASR) expects.
const TARGET_RATE = 16000;

export class MicStream {
  constructor({ onFrame }) {
    this.onFrame = onFrame; // (Float32Array at 16kHz) => void
    this.audioCtx = null;
    this.source = null;
    this.worklet = null;
    this.stream = null;
  }

  async start() {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.audioCtx = new AudioContext();
    this.source = this.audioCtx.createMediaStreamSource(this.stream);

    // ScriptProcessorNode is deprecated but universally supported and
    // simplest for a same-origin iframe module script (no worklet file
    // registration/MIME complications). Buffer size 2048 at native rate.
    const nativeRate = this.audioCtx.sampleRate;
    const ratio = TARGET_RATE / nativeRate;
    let resampleAccumulator = [];

    this.processor = this.audioCtx.createScriptProcessor(2048, 1, 1);
    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      // Linear-interpolation resample down to 16kHz.
      const outLength = Math.floor(input.length * ratio);
      const out = new Float32Array(outLength);
      for (let i = 0; i < outLength; i++) {
        const srcPos = i / ratio;
        const i0 = Math.floor(srcPos);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const frac = srcPos - i0;
        out[i] = input[i0] * (1 - frac) + input[i1] * frac;
      }
      this.onFrame(out);
    };
    this.source.connect(this.processor);
    // Must connect to a destination for onaudioprocess to fire reliably in
    // all browsers; route through a silent gain node so nothing is audible.
    const sink = this.audioCtx.createGain();
    sink.gain.value = 0;
    this.processor.connect(sink);
    sink.connect(this.audioCtx.destination);
  }

  stop() {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.audioCtx?.close();
    this.stream = null;
  }
}

/** Encodes a Float32Array (16kHz mono, range [-1,1]) as a 16-bit PCM WAV blob. */
export function encodeWav(float32, sampleRate = TARGET_RATE) {
  const buffer = new ArrayBuffer(44 + float32.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + float32.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, float32.length * 2, true);
  let offset = 44;
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}
