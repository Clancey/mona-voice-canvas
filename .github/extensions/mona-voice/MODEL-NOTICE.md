# Vendored model provenance

These ONNX graphs are vendored unmodified from the
[openWakeWord](https://github.com/dscripka/openWakeWord) project, release
`v0.5.1`, which is licensed under Apache License 2.0 (see `LICENSE` in this
directory). openWakeWord itself describes these as a reimplementation of
Google's `speech_embedding` model (https://tfhub.dev/google/speech_embedding/1).

| File | Source URL | SHA-256 |
| --- | --- | --- |
| `melspectrogram.onnx` | https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` |
| `embedding_model.onnx` | https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` |

## Role in the pipeline

- `melspectrogram.onnx`: converts raw 16kHz mono PCM into log-mel spectrogram
  frames (input: variable-length PCM samples; output: `[N, 32]` mel frames).
- `embedding_model.onnx`: converts a `76`-frame x `32`-mel window (with a
  trailing singleton channel dim, shape `[1, 76, 32, 1]`) into a 96-dimensional
  embedding vector (output `conv2d_19`, shape `[1, 1, 1, 96]`). This is the
  frozen, shared "AI model" stage of wake-word detection: identical for every
  user and phrase.

Both graphs are frozen and never fine-tuned in this app. Per-user
personalization happens entirely downstream, as a small classifier trained on
embeddings these graphs produce (see `crates/wake-word/src/classifier.rs`).

Do not replace these files without updating the SHA-256 values above and
re-running the conformance smoke test
(`crates/wake-word/tests/conformance.rs`), since the ORT pipeline's
preprocessing constants (window length, hop size, normalization) are tuned to
match this exact graph version. Note that test only checks the pipeline runs
deterministically and produces finite output on synthetic audio -- it does
not compare against upstream openWakeWord reference vectors byte-for-byte.
