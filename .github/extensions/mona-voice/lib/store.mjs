import { mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { dirname } from "node:path";
import { ARTIFACTS_DIR, CLASSIFIER_PATH, PREFERENCES_PATH } from "./paths.mjs";

async function readJson(path, fallback) {
  try {
    return JSON.parse(await readFile(path, "utf-8"));
  } catch {
    return fallback;
  }
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2));
}

export async function loadClassifier() {
  return readJson(CLASSIFIER_PATH, null);
}

export async function saveClassifier(classifier) {
  await writeJson(CLASSIFIER_PATH, classifier);
}

export async function resetClassifier() {
  await rm(CLASSIFIER_PATH, { force: true });
}

const DEFAULT_PREFERENCES = {
  wakeWordEnabled: false,
  pushToTalkOnly: true,
  wakeThreshold: 0.85,
};

export async function loadPreferences() {
  const stored = await readJson(PREFERENCES_PATH, {});
  return { ...DEFAULT_PREFERENCES, ...stored };
}

export async function savePreferences(partial) {
  const current = await loadPreferences();
  const next = { ...current, ...partial };
  await writeJson(PREFERENCES_PATH, next);
  return next;
}

export async function ensureArtifactsDir() {
  await mkdir(ARTIFACTS_DIR, { recursive: true });
}
