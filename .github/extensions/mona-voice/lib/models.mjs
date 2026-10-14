import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, stat } from "node:fs/promises";
import { MODELS_DIR } from "./paths.mjs";

// Vendored-model provenance, copied verbatim from the github/github-app
// wake-word PR's NOTICE.md (Apache-2.0, openWakeWord v0.5.1). Binaries are
// NOT committed to this repo (install_extension enforces a ~1MB per-file /
// ~5MB total cap on extension bundles) -- instead they are downloaded once
// on first use and cached under $COPILOT_HOME, then verified against these
// pinned hashes on every load so a corrupt/tampered cache is never trusted.
export const WAKE_WORD_MODELS = [
  {
    file: "melspectrogram.onnx",
    url: "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    sha256:
      "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
  },
  {
    file: "embedding_model.onnx",
    url: "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    sha256:
      "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
  },
];

async function sha256File(path) {
  const data = await readFile(path);
  return createHash("sha256").update(data).digest("hex");
}

async function fileExists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Ensures a single wake-word ONNX graph is present and hash-verified in the
 * local cache, downloading it from the pinned openWakeWord release if
 * missing or corrupt. Never silently uses a file that fails verification.
 */
export async function ensureModel(entry, { onProgress } = {}) {
  await mkdir(MODELS_DIR, { recursive: true });
  const path = `${MODELS_DIR}/${entry.file}`;

  if (await fileExists(path)) {
    const hash = await sha256File(path);
    if (hash === entry.sha256) return path;
    // Stale/corrupt cache entry -- fall through and re-download.
  }

  onProgress?.({ file: entry.file, status: "downloading" });
  const res = await fetch(entry.url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(
      `failed to download ${entry.file} from ${entry.url}: HTTP ${res.status}`,
    );
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const hash = createHash("sha256").update(buf).digest("hex");
  if (hash !== entry.sha256) {
    throw new Error(
      `${entry.file} failed checksum verification (got ${hash}, expected ${entry.sha256}); refusing to cache`,
    );
  }
  await writeFile(path, buf);
  onProgress?.({ file: entry.file, status: "ready" });
  return path;
}

export async function ensureAllModels(opts) {
  const paths = {};
  for (const entry of WAKE_WORD_MODELS) {
    paths[entry.file] = await ensureModel(entry, opts);
  }
  return paths;
}

export async function modelsCached() {
  for (const entry of WAKE_WORD_MODELS) {
    const path = `${MODELS_DIR}/${entry.file}`;
    if (!(await fileExists(path))) return false;
    const hash = await sha256File(path);
    if (hash !== entry.sha256) return false;
  }
  return true;
}
