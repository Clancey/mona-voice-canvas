import { createServer } from "node:http";
import { joinSession, createCanvas } from "@github/copilot-sdk/extension";

const servers = new Map();

async function startServer(instanceId) {
  const server = createServer((req, res) => {
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end(`<html><body><h1>mona-voice probe ok (${instanceId})</h1></body></html>`);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  return { server, url: `http://127.0.0.1:${port}/` };
}

await joinSession({
  canvases: [
    createCanvas({
      id: "mona-voice",
      displayName: "Mona voice agent",
      description: "Probe build: minimal canvas used to validate extension discovery in this app.",
      actions: [
        {
          name: "ping",
          description: "Health check action.",
          handler: async () => ({ pong: true, at: new Date().toISOString() }),
        },
      ],
      open: async (ctx) => {
        let entry = servers.get(ctx.instanceId);
        if (!entry) {
          entry = await startServer(ctx.instanceId);
          servers.set(ctx.instanceId, entry);
        }
        return { title: "Mona voice (probe)", url: entry.url };
      },
      onClose: async (ctx) => {
        const entry = servers.get(ctx.instanceId);
        if (entry) {
          servers.delete(ctx.instanceId);
          await new Promise((resolve) => entry.server.close(() => resolve()));
        }
      },
    }),
  ],
});
