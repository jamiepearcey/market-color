// market-color API server — the single backend all clients talk to.
//
// One standalone Node server that owns Postgres (config + state), the run
// orchestration, report generation, and the chat/Qdrant-MCP retrieval. It is
// CORS-enabled so the static UI (served by Tauri or any static host, possibly
// on another origin) can call it. Endpoints:
//   /api/*        JSON — desks, sources, runs, reports, health (server/api.mjs)
//   /chat         NDJSON stream — grounded chat turn
//   /chat/health  JSON — is the chat backend configured?
//   /health       JSON — liveness
//   /*            static dist/ (optional single-origin deploy)
//
// Run: `npm run api` (or node server/market-color-server.mjs). PORT=8787 by default.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chatReady, runChat, shutdown } from "./chat-core.mjs";
import { handleApi } from "./api.mjs";
import { migrate } from "./db.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(HERE, "..", "dist");
const PORT = Number(process.env.PORT || 8787);
const HOST = process.env.HOST || "0.0.0.0";
const CORS_ORIGIN = process.env.MARKET_CORS_ORIGIN || "*";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

function cors(res) {
  res.setHeader("access-control-allow-origin", CORS_ORIGIN);
  res.setHeader("access-control-allow-methods", "GET,POST,PATCH,DELETE,OPTIONS");
  res.setHeader("access-control-allow-headers", "content-type,authorization");
  res.setHeader("access-control-max-age", "86400");
}

function readBody(req) {
  return new Promise((resolve) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => resolve(b));
  });
}

function sendJson(res, status, obj) {
  res.statusCode = status;
  res.setHeader("content-type", "application/json; charset=utf-8");
  res.end(JSON.stringify(obj));
}

function serveStatic(req, res) {
  const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
  let file = path.join(DIST, urlPath === "/" ? "index.html" : urlPath);
  if (!file.startsWith(DIST) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    file = path.join(DIST, "index.html");
  }
  if (!fs.existsSync(file)) {
    res.statusCode = 404;
    return res.end("not built — run `npm run build`, or this is an API-only deploy");
  }
  res.setHeader("content-type", MIME[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
}

const server = http.createServer(async (req, res) => {
  cors(res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    return res.end();
  }

  const pathname = (req.url || "/").split("?")[0];

  // --- REST API ---
  if (pathname === "/api" || pathname.startsWith("/api/")) {
    const bodyStr = req.method === "GET" ? "" : await readBody(req);
    let body;
    try {
      body = bodyStr ? JSON.parse(bodyStr) : undefined;
    } catch {
      return sendJson(res, 400, { error: "invalid JSON body" });
    }
    const sub = (req.url || "/").slice(4) || "/"; // strip "/api", keep the ?query
    const { status, json } = await handleApi(req.method, sub, body);
    return sendJson(res, status, json);
  }

  // --- chat ---
  if (pathname === "/chat/health") {
    return sendJson(res, 200, { available: chatReady() });
  }
  if (pathname === "/chat" && req.method === "POST") {
    try {
      const { prompt, threadId, model } = JSON.parse((await readBody(req)) || "{}");
      if (!prompt?.trim()) return sendJson(res, 400, { error: "empty prompt" });
      res.setHeader("content-type", "application/x-ndjson");
      res.setHeader("cache-control", "no-cache");
      const result = await runChat(prompt, threadId ?? null, model ?? null, (e) => {
        res.write(JSON.stringify({ type: "event", event: e }) + "\n");
      });
      return res.end(JSON.stringify({ type: "result", result }) + "\n");
    } catch (e) {
      return sendJson(res, 400, { error: String(e) });
    }
  }

  if (pathname === "/health") {
    return sendJson(res, 200, { ok: true });
  }

  return serveStatic(req, res);
});

// Best-effort migrate on boot so a fresh DB is ready; don't crash if DB is down.
migrate().catch((e) => console.warn("[market-color] DB migrate deferred:", String(e?.message || e)));

server.listen(PORT, HOST, () => {
  console.log(`market-color API on http://${HOST}:${PORT}  (CORS: ${CORS_ORIGIN})`);
});

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, async () => {
    await shutdown();
    server.close(() => process.exit(0));
  });
}
