// Stdio MCP shim for codex: instead of spawning its own risk_mcp (1-3s state
// load per chat turn), codex talks to this proxy, which forwards tools/list
// and tools/call to the vite bridge's ALREADY-HOT engine child over HTTP.
// Startup is ~50ms, stdout is guaranteed pure JSON (codex's reader dies on
// any non-JSON line), and every chat turn prices against the same resident
// state the UI uses.

import { createInterface } from "node:readline";

// "localhost", not 127.0.0.1: vite binds the resolver's first answer (::1 on
// macOS), and a stray IPv4 listener once shadowed the real bridge here.
const BRIDGE = process.env.QUANT_BRIDGE_URL ?? "http://localhost:1420";
const out = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");

const rl = createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let req;
  try {
    req = JSON.parse(line);
  } catch {
    return;
  }
  const { id, method, params } = req;
  if (typeof method !== "string" || method.startsWith("notifications/")) return;
  if (method === "initialize") {
    return out({
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "risk-engine-proxy", version: "0.1.0" },
      },
    });
  }
  if (method === "ping") return out({ jsonrpc: "2.0", id, result: {} });
  void (async () => {
    try {
      const res = await fetch(`${BRIDGE}/engine/jsonrpc`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ method, params: params ?? {} }),
      });
      const body = await res.json();
      if (body.error) {
        out({ jsonrpc: "2.0", id, error: { code: -32000, message: String(body.error) } });
      } else {
        out({ jsonrpc: "2.0", id, result: body.result });
      }
    } catch (e) {
      out({ jsonrpc: "2.0", id, error: { code: -32001, message: `bridge unreachable: ${e}` } });
    }
  })();
});
