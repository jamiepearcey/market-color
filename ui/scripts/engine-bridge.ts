// Vite dev-server plugin: spawn the repo's risk_mcp binary and bridge HTTP →
// stdio JSON-RPC so `npm run dev` runs real simulations with no Tauri backend
// and no separate process to manage. Endpoints:
//   GET  /engine/health  {available}
//   POST /engine/rpc     {name, arguments} → {isError, text}
//   GET  /chat/health    {available}      — codex CLI present?
//   POST /chat           {prompt, threadId?} → {threadId, reply, toolCalls}
// /chat runs `codex exec --json` with cwd at the repo root (so AGENTS.md is
// the operating contract) and the risk_engine MCP server injected via -c
// overrides; tool calls codex makes are reported back so the frontend can
// replay them through /engine/rpc into its own evidence store.
// If the engine binary is missing the bridge reports unavailable and the
// frontend falls back to the clearly-badged preview runner.

import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { Plugin } from "vite";

export function engineBridge(): Plugin {
  let child: ChildProcessWithoutNullStreams | null = null;
  let nextId = 1;
  const pending = new Map<number, (line: string) => void>();

  const binary =
    process.env.QUANT_RISK_MCP ??
    path.resolve(__dirname, "../../target/release/risk_mcp");

  function ensureChild(): ChildProcessWithoutNullStreams | null {
    if (child && child.exitCode === null) return child;
    if (!existsSync(binary)) return null;
    child = spawn(binary, [], { stdio: ["pipe", "pipe", "inherit"] });
    let buf = "";
    child.stdout.on("data", (chunk: Buffer) => {
      buf += chunk.toString("utf8");
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        try {
          const msg = JSON.parse(line) as { id?: number };
          const resolve = msg.id !== undefined ? pending.get(msg.id) : undefined;
          if (resolve && msg.id !== undefined) {
            pending.delete(msg.id);
            resolve(line);
          }
        } catch {
          // non-JSON noise on stdout — ignore
        }
      }
    });
    child.on("exit", () => {
      for (const resolve of pending.values()) resolve("");
      pending.clear();
      child = null;
    });
    // stdin buffers while the engine loads state (~3s), so no readiness gate.
    child.stdin.write(
      JSON.stringify({
        jsonrpc: "2.0",
        id: nextId++,
        method: "initialize",
        params: {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "quant-brain-ui", version: "0.1.0" },
        },
      }) + "\n",
    );
    return child;
  }

  /** Raw JSON-RPC to the hot child: any method, full response. Shared by
   *  the UI path (tools/call) and the codex MCP proxy (tools/list too). */
  function rpcRaw(
    method: string,
    params: unknown,
  ): Promise<{ result?: unknown; error?: string }> {
    const proc = ensureChild();
    if (!proc)
      return Promise.resolve({
        error: `engine binary not found at ${binary} — cargo build --release --bin risk_mcp`,
      });
    const id = nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        resolve({ error: "engine call timed out (120s)" });
      }, 120_000);
      pending.set(id, (line) => {
        clearTimeout(timer);
        if (!line) return resolve({ error: "engine process exited" });
        try {
          const msg = JSON.parse(line) as {
            result?: unknown;
            error?: { message?: string };
          };
          if (msg.error) return resolve({ error: msg.error.message ?? "rpc error" });
          resolve({ result: msg.result });
        } catch {
          resolve({ error: "unparseable engine response" });
        }
      });
      proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  async function rpc(name: string, args: unknown): Promise<{ isError: boolean; text: string }> {
    const r = await rpcRaw("tools/call", { name, arguments: args });
    if (r.error) return { isError: true, text: r.error };
    const result = r.result as { isError?: boolean; content?: { text?: string }[] };
    return { isError: result?.isError === true, text: result?.content?.[0]?.text ?? "" };
  }

  // --- chat over codex exec --------------------------------------------------

  const repoRoot = path.resolve(__dirname, "../..");
  let codexAvailable: boolean | null = null;
  function hasCodex(): boolean {
    codexAvailable ??= spawnSync("codex", ["--version"], { stdio: "ignore" }).status === 0;
    return codexAvailable;
  }

  /** Codex's MCP server is a thin proxy onto OUR hot engine child: ~50ms
   *  startup instead of a 1-3s state load per chat turn, and stdout is pure
   *  JSON (codex's reader treats any non-JSON stream line as fatal). The
   *  wrapper pins node + script + bridge URL into one argv-free command. */
  let quietWrapper: string | null = null;
  function mcpWrapper(port: number): string {
    if (quietWrapper) return quietWrapper;
    const proxy = path.resolve(__dirname, "engine-mcp-proxy.mjs");
    const dir = mkdtempSync(path.join(os.tmpdir(), "quant-brain-"));
    quietWrapper = path.join(dir, "risk-mcp-proxy.sh");
    writeFileSync(
      quietWrapper,
      `#!/bin/sh\nQUANT_BRIDGE_URL="http://localhost:${port}" exec "${process.execPath}" "${proxy}" 2>/dev/null\n`,
    );
    chmodSync(quietWrapper, 0o755);
    return quietWrapper;
  }

  interface ChatToolCall {
    tool: string;
    args: Record<string, unknown>;
    status: string;
  }

  interface ChatResult {
    threadId: string | null;
    reply: string;
    toolCalls: ChatToolCall[];
    error?: string;
    /** A nudge surfaced when the agent repeated an action without progress. */
    hint?: string;
    /** True when the turn was cancelled for inactivity (vs an explicit error). */
    stalled?: boolean;
  }

  /** Extract engine invocations from a shell command codex ran (the AGENTS.md
   *  fallback path when it shells out to risk_report instead of using MCP). */
  function specsFromCommand(command: string): ChatToolCall[] {
    const out: ChatToolCall[] = [];
    const story = /--story[=\s]+"((?:[^"\\]|\\.)*)"/g;
    let m: RegExpExecArray | null;
    while ((m = story.exec(command))) {
      out.push({ tool: "run_story", args: { spec: m[1] }, status: "completed" });
    }
    const chain = /--chain[=\s]+"((?:[^"\\]|\\.)*)"/g;
    while ((m = chain.exec(command))) {
      out.push({
        tool: "run_chain",
        args: { specs: m[1].split("||").map((s) => s.trim()).filter(Boolean) },
        status: "completed",
      });
    }
    return out;
  }

  // One codex turn at a time: parallel sessions on one thread interleave
  // badly and the engine evidence replay is cheap anyway.
  let chatQueue: Promise<unknown> = Promise.resolve();
  let bridgePort = 1420;

  /** Live progress event forwarded to the UI while a turn runs. `delta` is a
   *  token chunk of the assistant message — real streaming. */
  interface ChatEvent {
    kind: "thinking" | "tool" | "shell" | "delta" | "hint";
    label: string;
  }

  // --- turn lifecycle: idle-based cancel + stuck-pattern hints --------------
  // Codex is cancelled only when it STOPS producing events for CODEX_IDLE_MS,
  // not on a fixed wall-clock cap — an agent that is still working is never
  // arbitrarily killed. An OPTIONAL absolute cap (CODEX_MAX_MS, 0 = off, the
  // default) guards true runaways. Both are env-overridable.
  const CODEX_IDLE_MS = Number(process.env.CODEX_IDLE_MS) || 120_000;
  const CODEX_MAX_MS = Number(process.env.CODEX_MAX_MS) || 0;

  function makeTurnTimer(onCancel: (reason: string) => void) {
    let idle: ReturnType<typeof setTimeout>;
    const armIdle = () => {
      idle = setTimeout(
        () => onCancel(`codex idle ${Math.round(CODEX_IDLE_MS / 1000)}s — no activity`),
        CODEX_IDLE_MS,
      );
    };
    armIdle();
    const max =
      CODEX_MAX_MS > 0
        ? setTimeout(
            () => onCancel(`codex max turn ${Math.round(CODEX_MAX_MS / 1000)}s`),
            CODEX_MAX_MS,
          )
        : null;
    return {
      activity: () => {
        clearTimeout(idle);
        armIdle();
      },
      clear: () => {
        clearTimeout(idle);
        if (max) clearTimeout(max);
      },
    };
  }

  // Detect the agent repeating the same action without progress (the "flailing
  // on how analogs accepts horizon controls" failure). On the Nth near-
  // identical tool/shell label, emit a hint pointing at the data-driven
  // shortcuts instead of more source-probing. An agent message resets it.
  const CODEX_STUCK_REPEATS = Number(process.env.CODEX_STUCK_REPEATS) || 3;
  const STUCK_HINT =
    "You seem to be repeating the same probe. Don't keep reading source — use the " +
    "data-driven shortcuts: `scripts/discover.py` (what data exists), the `lift` command " +
    "(audited conditioning->target by argument), or ad-hoc Python over the data/ folder. " +
    "See docs/extending-the-engine.md.";

  function makeStuckDetector(onStuck: (hint: string) => void) {
    const recent: string[] = [];
    let fired = false;
    return {
      saw: (label: string) => {
        const norm = label.replace(/\d+/g, "#").replace(/\s+/g, " ").trim().slice(0, 90);
        recent.push(norm);
        if (recent.length > 6) recent.shift();
        if (!fired && recent.filter((x) => x === norm).length >= CODEX_STUCK_REPEATS) {
          fired = true;
          onStuck(STUCK_HINT);
        }
      },
      progress: () => {
        recent.length = 0;
      },
    };
  }

  // --- persistent codex app-server (the Zed model: one long-lived agent
  // process speaking JSON-RPC over stdio; initialize once, a thread per
  // project, a turn per message, notifications stream the work) -------------

  let app: ChildProcessWithoutNullStreams | null = null;
  let appReady: Promise<boolean> | null = null;
  let appNextId = 1;
  const appPending = new Map<number, (msg: { result?: unknown; error?: { message?: string } }) => void>();
  const threadListeners = new Map<string, (method: string, params: Record<string, unknown>) => void>();

  function appAlive(): boolean {
    return app !== null && app.exitCode === null;
  }

  function spawnAppServer(): ChildProcessWithoutNullStreams | null {
    if (!hasCodex()) return null;
    const isolation = ["shadcn", "originui", "node_repl", "chrome-devtools"].flatMap(
      (s) => ["-c", `mcp_servers.${s}.enabled=false`],
    );
    const child = spawn(
      "codex",
      [
        "app-server",
        ...isolation,
        "-c",
        `mcp_servers.risk_engine.command="${mcpWrapper(bridgePort)}"`,
        "-c",
        "mcp_servers.risk_engine.startup_timeout_sec=20",
        "-c",
        'mcp_servers.risk_engine.default_tools_approval_mode="approve"',
      ],
      { cwd: repoRoot, stdio: ["pipe", "pipe", "inherit"] },
    );
    let buf = "";
    child.stdout.on("data", (chunk: Buffer) => {
      buf += chunk.toString("utf8");
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line) as {
            id?: number;
            method?: string;
            params?: Record<string, unknown>;
            result?: unknown;
            error?: { message?: string };
          };
          if (msg.id !== undefined && appPending.has(msg.id) && msg.method === undefined) {
            const resolve = appPending.get(msg.id)!;
            appPending.delete(msg.id);
            resolve(msg);
          } else if (msg.method && msg.params) {
            const threadId = (msg.params as { threadId?: string }).threadId;
            if (threadId) threadListeners.get(threadId)?.(msg.method, msg.params);
          }
        } catch {
          // non-JSON noise — ignore
        }
      }
    });
    child.on("exit", () => {
      for (const resolve of appPending.values()) resolve({ error: { message: "app-server exited" } });
      appPending.clear();
      app = null;
      appReady = null;
    });
    return child;
  }

  function appRpc(
    method: string,
    params: unknown,
    timeoutMs = 300_000,
  ): Promise<{ result?: unknown; error?: { message?: string } }> {
    if (!appAlive()) return Promise.resolve({ error: { message: "app-server not running" } });
    const id = appNextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        appPending.delete(id);
        resolve({ error: { message: `${method} timed out` } });
      }, timeoutMs);
      appPending.set(id, (msg) => {
        clearTimeout(timer);
        resolve(msg);
      });
      app!.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  function appErrorMessage(prefix: string, error: unknown): string {
    if (!error || typeof error !== "object") return `${prefix}: ${String(error)}`;
    const rec = error as { message?: unknown; data?: unknown; code?: unknown };
    const message = typeof rec.message === "string" ? rec.message : JSON.stringify(error);
    const code = rec.code === undefined ? "" : ` [${String(rec.code)}]`;
    const data = rec.data === undefined ? "" : ` ${JSON.stringify(rec.data)}`;
    return `${prefix}: ${message}${code}${data}`;
  }

  /** Spawn + initialize once; reused across every turn until exit. */
  function ensureAppServer(): Promise<boolean> {
    if (appAlive() && appReady) return appReady;
    app = spawnAppServer();
    if (!app) return Promise.resolve(false);
    appReady = appRpc(
      "initialize",
      {
        clientInfo: { name: "quant-brain-ui", title: "QuantBrain", version: "0.1.0" },
      },
      30_000,
    ).then((r) => !r.error);
    return appReady;
  }

  /** One turn through the persistent server. The threadId returned is the
   *  app-server thread id; resume after a bridge restart loads it from disk. */
  async function runCodexAppServer(
    prompt: string,
    threadId: string | null,
    model: string | null,
    onEvent?: (e: ChatEvent) => void,
  ): Promise<ChatResult> {
    if (!(await ensureAppServer()))
      return { threadId, reply: "", toolCalls: [], error: "codex app-server failed to start" };

    let tid = threadId;
    if (tid) {
      // Threads from a previous bridge process live on disk; resume is
      // idempotent for already-loaded threads.
      const resumed = await appRpc("thread/resume", { threadId: tid, ...(model ? { model } : {}) }, 60_000);
      if (resumed.error) tid = null;
    }
    if (!tid) {
      const started = await appRpc(
        "thread/start",
        { cwd: repoRoot, approvalPolicy: "never", ...(model ? { model } : {}) },
        60_000,
      );
      if (started.error)
        return { threadId, reply: "", toolCalls: [], error: appErrorMessage("thread/start", started.error) };
      tid = (started.result as { thread?: { id?: string } }).thread?.id ?? null;
      if (!tid) return { threadId, reply: "", toolCalls: [], error: "thread/start returned no id" };
    }

    const result: ChatResult = { threadId: tid, reply: "", toolCalls: [] };
    const turnDone = new Promise<void>((resolve) => {
      const finish = () => {
        threadListeners.delete(tid!);
        resolve();
      };
      const timer = makeTurnTimer((reason) => {
        result.error ??= reason;
        result.stalled = true;
        finish();
      });
      const stuck = makeStuckDetector((hint) => {
        result.hint = hint;
        onEvent?.({ kind: "hint", label: hint });
      });
      threadListeners.set(tid!, (method, params) => {
        timer.activity();
        if (method === "item/agentMessage/delta") {
          onEvent?.({ kind: "delta", label: String((params as { delta?: string }).delta ?? "") });
        } else if (method === "item/started") {
          const item = (params as { item?: { type?: string; tool?: string; arguments?: unknown; command?: string } }).item;
          if (item?.type === "mcpToolCall" && item.tool) {
            const spec = (item.arguments as { spec?: string } | undefined)?.spec;
            const label = spec ? `${item.tool} · ${spec}` : item.tool;
            stuck.saw(label);
            onEvent?.({ kind: "tool", label });
          }
          if (item?.type === "commandExecution" && item.command) {
            const label = String(item.command).slice(0, 120);
            stuck.saw(label);
            onEvent?.({ kind: "shell", label });
          }
        } else if (method === "item/completed") {
          const item = (params as {
            item?: {
              type?: string;
              text?: string;
              tool?: string;
              arguments?: unknown;
              status?: string;
              command?: string;
            };
          }).item;
          if (item?.type === "agentMessage" && item.text) {
            result.reply = item.text;
            stuck.progress(); // an agent message = real progress; reset the loop window
          }
          if (item?.type === "mcpToolCall" && item.tool) {
            let args: Record<string, unknown> = {};
            if (typeof item.arguments === "string") {
              try {
                args = JSON.parse(item.arguments) as Record<string, unknown>;
              } catch {
                args = {};
              }
            } else if (item.arguments && typeof item.arguments === "object") {
              args = item.arguments as Record<string, unknown>;
            }
            result.toolCalls.push({ tool: item.tool, args, status: item.status ?? "unknown" });
          }
          if (item?.type === "commandExecution" && item.command)
            result.toolCalls.push(...specsFromCommand(String(item.command)));
        } else if (method === "turn/completed") {
          timer.clear();
          finish();
        } else if (method === "error") {
          result.error = appErrorMessage("codex", params);
          timer.clear();
          finish();
        }
      });
    });

    const turnParams: Record<string, unknown> = {
      threadId: tid,
      input: [{ type: "text", text: prompt }],
      // Spec translation is mechanical; the user's global "high" effort
      // turns a 5s turn into a 60s one for no gain.
      effort: "low",
    };
    // User-selected model (empty = codex default). Per-turn so a model switch
    // takes effect on the next message without restarting the thread.
    if (model) turnParams.model = model;
    const turn = await appRpc("turn/start", turnParams);
    if (turn.error) {
      threadListeners.delete(tid);
      return { threadId: tid, reply: "", toolCalls: [], error: appErrorMessage("turn/start", turn.error) };
    }
    await turnDone;
    return result;
  }

  /** Preferred path: the persistent app-server (session stays warm; only
   *  model latency per turn). Falls back to one-shot `codex exec` when the
   *  app-server can't start (e.g. older codex builds). Serialized either way. */
  function runCodex(
    prompt: string,
    threadId: string | null,
    model: string | null,
    onEvent?: (e: ChatEvent) => void,
  ): Promise<ChatResult> {
    const run = async (): Promise<ChatResult> => {
      const viaApp = await runCodexAppServer(prompt, threadId, model, onEvent);
      // exec-thread ids are UUIDs from `codex exec`; app-server ids are its
      // own — a failed resume already falls back to a new thread above, so
      // only a startup failure routes to exec.
      if (viaApp.error === "codex app-server failed to start")
        return runCodexExec(prompt, threadId, model, onEvent);
      return viaApp;
    };
    const next = chatQueue.then(run, run);
    chatQueue = next;
    return next;
  }

  function runCodexExec(
    prompt: string,
    threadId: string | null,
    model: string | null,
    onEvent?: (e: ChatEvent) => void,
  ): Promise<ChatResult> {
    const run = () =>
      new Promise<ChatResult>((resolve) => {
        if (!hasCodex())
          return resolve({ threadId, reply: "", toolCalls: [], error: "codex CLI not found on PATH" });
        const isolation = ["shadcn", "originui", "node_repl", "chrome-devtools"].flatMap(
          (s) => ["-c", `mcp_servers.${s}.enabled=false`],
        );
        const args = [
          "exec",
          ...(threadId ? ["resume", threadId] : []),
          "--json",
          // User-selected model (empty = codex default).
          ...(model ? ["--model", model] : []),
          ...isolation,
          "-c",
          `mcp_servers.risk_engine.command="${mcpWrapper(bridgePort)}"`,
          "-c",
          "mcp_servers.risk_engine.startup_timeout_sec=20",
          // codex exec auto-cancels MCP calls that would prompt (openai/codex
          // #16685); "approve" pre-approves the read-only engine tools.
          "-c",
          'mcp_servers.risk_engine.default_tools_approval_mode="approve"',
          // Spec translation is mechanical — the user's global "high"
          // reasoning effort turns a 5s turn into a 60s one for no gain.
          "-c",
          'model_reasoning_effort="low"',
          prompt,
        ];
        const child = spawn("codex", args, {
          cwd: repoRoot, // AGENTS.md is the operating contract
          stdio: ["ignore", "pipe", "pipe"],
        });
        const result: ChatResult = { threadId, reply: "", toolCalls: [] };
        let buf = "";
        let stderrTail = "";
        const timer = makeTurnTimer((reason) => {
          result.error = reason;
          result.stalled = true;
          child.kill();
        });
        const stuck = makeStuckDetector((hint) => {
          result.hint = hint;
          onEvent?.({ kind: "hint", label: hint });
        });
        child.stdout.on("data", (chunk: Buffer) => {
          buf += chunk.toString("utf8");
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl);
            buf = buf.slice(nl + 1);
            if (!line.trim()) continue;
            timer.activity();
            try {
              const e = JSON.parse(line) as {
                type?: string;
                thread_id?: string;
                message?: string;
                item?: {
                  type?: string;
                  text?: string;
                  tool?: string;
                  arguments?: Record<string, unknown>;
                  status?: string;
                  command?: string;
                };
              };
              if (e.type === "thread.started" && e.thread_id) result.threadId = e.thread_id;
              if (e.type === "error" && e.message) result.error = e.message;
              if (e.type === "item.started" && e.item) {
                const item = e.item;
                if (item.type === "mcp_tool_call" && item.tool) {
                  const spec = (item.arguments as { spec?: string } | undefined)?.spec;
                  const label = spec ? `${item.tool} · ${spec}` : item.tool;
                  stuck.saw(label);
                  onEvent?.({ kind: "tool", label });
                }
                if (item.type === "command_execution" && item.command) {
                  const label = item.command.slice(0, 120);
                  stuck.saw(label);
                  onEvent?.({ kind: "shell", label });
                }
              }
              if (e.type === "item.completed" && e.item) {
                const item = e.item;
                if (item.type === "agent_message" && item.text) {
                  result.reply = item.text;
                  stuck.progress();
                  onEvent?.({ kind: "thinking", label: item.text.slice(0, 110) });
                }
                if (item.type === "mcp_tool_call" && item.tool) {
                  result.toolCalls.push({
                    tool: item.tool,
                    args: item.arguments ?? {},
                    status: item.status ?? "unknown",
                  });
                }
                if (item.type === "command_execution" && item.command) {
                  result.toolCalls.push(...specsFromCommand(item.command));
                }
              }
            } catch {
              // non-JSON noise on stdout — ignore
            }
          }
        });
        child.stderr.on("data", (chunk: Buffer) => {
          stderrTail = (stderrTail + chunk.toString("utf8")).slice(-2000);
        });
        child.on("close", (code) => {
          timer.clear();
          if (code !== 0 && !result.reply && !result.error)
            result.error = `codex exited ${code}: ${stderrTail.split("\n").filter(Boolean).pop() ?? ""}`;
          resolve(result);
        });
        child.on("error", (err) => {
          timer.clear();
          resolve({ threadId, reply: "", toolCalls: [], error: String(err) });
        });
      });
    // No queueing here — runCodex (the only caller) already serializes.
    return run();
  }

  return {
    name: "quant-brain-engine-bridge",
    configureServer(server) {
      bridgePort = server.config.server.port ?? 1420;
      server.middlewares.use("/engine/jsonrpc", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", async () => {
          res.setHeader("content-type", "application/json");
          try {
            const { method, params } = JSON.parse(body) as { method: string; params: unknown };
            res.end(JSON.stringify(await rpcRaw(method, params)));
          } catch (e) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: String(e) }));
          }
        });
      });
      server.middlewares.use("/chat/health", (_req, res) => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ available: hasCodex() }));
      });
      // NDJSON stream: live {type:"event"} lines while codex works, then one
      // {type:"result"} line — so the UI narrates the turn instead of
      // freezing on a spinner.
      server.middlewares.use("/chat", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", async () => {
          try {
            const { prompt, threadId, model } = JSON.parse(body) as {
              prompt: string;
              threadId?: string | null;
              model?: string | null;
            };
            if (!prompt?.trim()) {
              res.statusCode = 400;
              res.setHeader("content-type", "application/json");
              return res.end(JSON.stringify({ error: "empty prompt" }));
            }
            res.setHeader("content-type", "application/x-ndjson");
            res.setHeader("cache-control", "no-cache");
            res.flushHeaders?.();
            const result = await runCodex(prompt, threadId ?? null, model ?? null, (e) => {
              res.write(JSON.stringify({ type: "event", event: e }) + "\n");
            });
            res.end(JSON.stringify({ type: "result", result }) + "\n");
          } catch (e) {
            res.statusCode = 400;
            res.setHeader("content-type", "application/json");
            res.end(JSON.stringify({ error: String(e) }));
          }
        });
      });
      server.middlewares.use("/engine/health", (_req, res) => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ available: ensureChild() !== null, binary }));
      });
      server.middlewares.use("/engine/rpc", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", async () => {
          res.setHeader("content-type", "application/json");
          try {
            const { name, arguments: args } = JSON.parse(body) as {
              name: string;
              arguments: unknown;
            };
            res.end(JSON.stringify(await rpc(name, args ?? {})));
          } catch (e) {
            res.statusCode = 400;
            res.end(JSON.stringify({ isError: true, text: String(e) }));
          }
        });
      });
    },
    closeBundle() {
      child?.kill();
      app?.kill();
    },
  };
}
