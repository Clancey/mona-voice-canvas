import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, join, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { joinSession, createCanvas } from "@github/copilot-sdk/extension";
import { ensureAllModels, modelsCached } from "./lib/models.mjs";
import {
  loadClassifier,
  saveClassifier,
  resetClassifier,
  loadPreferences,
  savePreferences,
} from "./lib/store.mjs";
import { backendHealth, transcribe, speakStream, stopSpeaking } from "./lib/backends.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = join(HERE, "public");

// How long a voice utterance will wait for the hosting session to finish
// replying before giving up. Default 10 minutes -- real turns can involve
// tool calls and multi-step reasoning, not just a quick chat completion.
const SESSION_REPLY_TIMEOUT_MS = Number(process.env.MONA_SESSION_TIMEOUT_MS || 600_000);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

// One entry per open canvas instance: HTTP server, SSE subscriber set, and
// an AbortController for whatever utterance is currently in flight (so
// /interrupt can cancel it for barge-in).
const instances = new Map();

function broadcast(instance, event) {
  const payload = `data: ${JSON.stringify(event)}\n\n`;
  for (const res of instance.sseClients) {
    res.write(payload);
  }
}

async function serveStatic(req, res, pathname) {
  const rel = pathname === "/" ? "index.html" : pathname.slice(1);
  const safe = rel.replace(/\.\./g, "");
  const path = join(PUBLIC_DIR, safe);
  if (!path.startsWith(PUBLIC_DIR)) {
    res.writeHead(403).end("forbidden");
    return;
  }
  try {
    const body = await readFile(path);
    res.writeHead(200, { "Content-Type": MIME[extname(path)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf-8")) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

async function handleUtterance(req, res, instance, session) {
  instance.abortController?.abort();
  const controller = new AbortController();
  instance.abortController = controller;

  try {
    const wav = await readRawBody(req);

    broadcast(instance, { type: "state", phase: "transcribing" });
    const transcript = await transcribe(wav, { signal: controller.signal });
    if (!transcript.trim()) {
      broadcast(instance, { type: "state", phase: "listening" });
      res.writeHead(204).end();
      return;
    }
    broadcast(instance, { type: "state", phase: "thinking", transcript });

    // This is the core behavioral requirement: the voice stack does NOT
    // generate its own answer. The transcript is delivered as a real user
    // turn into whatever Copilot session this canvas is open inside, and we
    // speak back whatever that session's own assistant actually replied.
    // The default sendAndWait timeout (60s) is too short for real assistant
    // turns that do actual work (tool calls, multi-step reasoning). A voice
    // utterance should wait as long as a normal chat turn would.
    const reply = await session.sendAndWait(
      { prompt: transcript, mode: "immediate" },
      SESSION_REPLY_TIMEOUT_MS,
    );
    const replyText = reply?.data?.content ?? "";
    if (!replyText.trim()) {
      broadcast(instance, { type: "state", phase: "listening" });
      res.writeHead(204).end();
      return;
    }
    broadcast(instance, { type: "state", phase: "speaking", transcript, assistantReply: replyText });

    const stream = await speakStream(replyText, { signal: controller.signal });
    res.writeHead(200, { "Content-Type": "application/octet-stream" });
    for await (const chunk of stream) {
      if (controller.signal.aborted) break;
      res.write(chunk);
    }
    res.end();
    if (!controller.signal.aborted) {
      broadcast(instance, { type: "state", phase: "listening" });
    }
  } catch (err) {
    if (controller.signal.aborted) {
      res.end();
      return;
    }
    broadcast(instance, { type: "state", phase: "error", error: String(err?.message || err) });
    if (!res.headersSent) res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: String(err?.message || err) }));
  } finally {
    if (instance.abortController === controller) instance.abortController = null;
  }
}

async function startServer(instanceId, session) {
  const instance = { sseClients: new Set(), abortController: null };

  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");

    if (url.pathname === "/events") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write(": connected\n\n");
      instance.sseClients.add(res);
      req.on("close", () => instance.sseClients.delete(res));
      return;
    }

    if (url.pathname === "/health" && req.method === "GET") {
      const health = await backendHealth();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ...health, modelsCached: await modelsCached() }));
      return;
    }

    if (url.pathname.startsWith("/models/") && req.method === "GET") {
      try {
        const paths = await ensureAllModels({
          onProgress: (p) => broadcast(instance, { type: "model-download", ...p }),
        });
        const file = url.pathname.slice("/models/".length);
        const path = paths[file];
        if (!path) {
          res.writeHead(404).end("unknown model");
          return;
        }
        const body = await readFile(path);
        res.writeHead(200, { "Content-Type": "application/octet-stream" });
        res.end(body);
      } catch (err) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: String(err?.message || err) }));
      }
      return;
    }

    if (url.pathname === "/wakeword/classifier" && req.method === "GET") {
      const classifier = await loadClassifier();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ trained: !!classifier, classifier }));
      return;
    }

    if (url.pathname === "/wakeword/classifier" && req.method === "POST") {
      const body = await readJsonBody(req);
      await saveClassifier(body);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    if (url.pathname === "/wakeword/reset" && req.method === "POST") {
      await resetClassifier();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    if (url.pathname === "/preferences" && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(await loadPreferences()));
      return;
    }

    if (url.pathname === "/preferences" && req.method === "POST") {
      const body = await readJsonBody(req);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(await savePreferences(body)));
      return;
    }

    if (url.pathname === "/utterance" && req.method === "POST") {
      await handleUtterance(req, res, instance, session);
      return;
    }

    if (url.pathname === "/interrupt" && req.method === "POST") {
      instance.abortController?.abort();
      await stopSpeaking();
      broadcast(instance, { type: "state", phase: "listening" });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    if (req.method === "GET") {
      await serveStatic(req, res, url.pathname);
      return;
    }

    res.writeHead(404).end("not found");
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  instance.server = server;
  instance.url = `http://127.0.0.1:${port}/`;
  instances.set(instanceId, instance);
  return instance;
}

const session = await joinSession({
  canvases: [
    createCanvas({
      id: "mona-voice",
      displayName: "Mona voice agent",
      description:
        "Hey Mona: a voice interface for this Copilot session. Wake word or push-to-talk, local ASR, and local TTS -- the session's own assistant generates every reply.",
      actions: [
        {
          name: "get_status",
          description: "Returns backend reachability, model cache state, and wake-word training status.",
          handler: async () => ({
            backends: await backendHealth(),
            modelsCached: await modelsCached(),
            wakeWordTrained: !!(await loadClassifier()),
            preferences: await loadPreferences(),
          }),
        },
        {
          name: "reset_wake_word_training",
          description: "Deletes the user's trained wake-word classifier so they can retrain from scratch.",
          handler: async () => {
            await resetClassifier();
            return { ok: true };
          },
        },
      ],
      open: async (ctx) => {
        let instance = instances.get(ctx.instanceId);
        if (!instance) {
          instance = await startServer(ctx.instanceId, session);
        }
        return { title: "Hey Mona", url: instance.url };
      },
      onClose: async (ctx) => {
        const instance = instances.get(ctx.instanceId);
        if (instance) {
          instances.delete(ctx.instanceId);
          instance.abortController?.abort();
          for (const res of instance.sseClients) res.end();
          await new Promise((resolve) => instance.server.close(() => resolve()));
        }
      },
    }),
  ],
});
