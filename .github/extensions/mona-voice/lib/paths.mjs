import { homedir } from "node:os";
import { join } from "node:path";

// Durable, user-global storage (per skill guidance): follows the user across
// sessions/projects, NOT keyed by canvas instanceId. Holds the downloaded
// openWakeWord ONNX graphs (cache) and the user's trained wake-word
// classifier + preferences (real state).
export const COPILOT_HOME =
  process.env.COPILOT_HOME || join(homedir(), ".copilot");

export const ARTIFACTS_DIR = join(
  COPILOT_HOME,
  "extensions",
  "mona-voice",
  "artifacts",
);

export const MODELS_DIR = join(ARTIFACTS_DIR, "models");
export const CLASSIFIER_PATH = join(ARTIFACTS_DIR, "wake-word-classifier.json");
export const PREFERENCES_PATH = join(ARTIFACTS_DIR, "preferences.json");
