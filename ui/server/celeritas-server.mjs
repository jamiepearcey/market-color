#!/usr/bin/env node
// Celeritas UI ⇄ Engine sidecar HTTP server.
//
// A local Node sidecar that wires the Celeritas control UI to the REAL engine by
// shelling out to the prebuilt `celeritas` binary. See ui/docs/WIRING.md and
// ADR-0015. Node built-ins only — no npm deps.
//
// Endpoints are documented in server/README.md and WIRING.md §Endpoints. Every
// response body maps engine output INTO the existing TS interfaces in
// src/lib/types.ts (ConnectorSpec, ConnectorParam, ConnectorInstanceRecord,
// ConnectorTestResult, ConnectorActionResult, SourceSchema, Target, Job, JobRun,
// JobRunStep, EmbeddedConfig).

import http from "node:http";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import {
  KeyedAsyncQueue,
  assertSafeIdentifier,
  isLoopbackHost,
  isMutatingMethod,
  isOriginAllowed,
  resolveCorsOrigins,
  validateBody,
} from "./core.mjs";
import { classifyEngineFailure } from "./engine-errors.mjs";
import { ERROR_CODES, appError, errorBody } from "./errors.mjs";
import { parseRunSummary } from "./parser.mjs";
import { readStoreFile, writeStoreFile } from "./store-files.mjs";
import {
  initRegistry,
  handleRegistryList,
  handleRegistryPackage,
  handleRegistryInstall,
  handleRegistryInstallStream,
  handleRegistryUrlsGet,
  handleRegistryUrlsPut,
} from "./registry.mjs";

// ---------------------------------------------------------------------------
// Paths / config
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UI_PACKAGE_JSON = path.join(__dirname, "..", "package.json");

const HOST = process.env.HOST?.trim() || "127.0.0.1";
const PORT = Number(process.env.PORT) || 8787;
const API_TOKEN = process.env.CELERITAS_API_TOKEN?.trim() || "";
const CORS_ORIGINS = resolveCorsOrigins(process.env.CORS_ORIGINS);

// Celeritas repo root = two levels up from `ui/` (server lives in ui/server).
const UI_DIR = path.resolve(__dirname, "..");
const CELERITAS_ROOT = process.env.CELERITAS_ROOT
  ? path.resolve(process.env.CELERITAS_ROOT)
  : path.resolve(UI_DIR, "..");

const CELERITAS_BIN = process.env.CELERITAS_BIN
  ? path.resolve(process.env.CELERITAS_BIN)
  : path.join(CELERITAS_ROOT, "bin", "celeritas");

// The sidecar's working Celeritas project. `celeritas init` creates a directory
// NAMED <name> under cwd, so we init "project" inside ~/.celeritas-ui.
const PROJECT_PARENT = path.join(os.homedir(), ".celeritas-ui");
const PROJECT_DIR = path.join(PROJECT_PARENT, "project");
const RUN_STEPS_FILE = path.join(PROJECT_DIR, "run-steps.json");
const FS_ALLOWED_ROOTS = String(process.env.FS_ALLOWED_ROOTS || `${os.homedir()},${PROJECT_PARENT}`)
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean)
  .map((entry) => path.resolve(entry.replace(/^~(?=$|\/)/, os.homedir())));

// Pre-generated authoritative catalog (settings + scope + category).
const CATALOG_JSON = path.join(
  CELERITAS_ROOT,
  "site",
  "src",
  "data",
  "connectors.json",
);

const UI_PACKAGE = readJsonFile(UI_PACKAGE_JSON, {});
const BUILD_VERSION =
  process.env.CELERITAS_BUILD_VERSION?.trim()
  || UI_PACKAGE.version
  || "0.1.0-dev";
const BUILD_SHA = process.env.CELERITAS_BUILD_SHA?.trim() || "dev";
const LOG_FORMAT = String(process.env.LOG_FORMAT || "pretty").toLowerCase() === "json"
  ? "json"
  : "pretty";
const RUN_HISTORY_LIMIT = Number(process.env.RUN_HISTORY_LIMIT) > 0
  ? Number(process.env.RUN_HISTORY_LIMIT)
  : 200;
const RATE_LIMIT_WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MS) > 0
  ? Number(process.env.RATE_LIMIT_WINDOW_MS)
  : 60_000;
const INSTALL_RATE_LIMIT_COUNT = Number(process.env.INSTALL_RATE_LIMIT_COUNT) > 0
  ? Number(process.env.INSTALL_RATE_LIMIT_COUNT)
  : 3;
const RUN_RATE_LIMIT_COUNT = Number(process.env.RUN_RATE_LIMIT_COUNT) > 0
  ? Number(process.env.RUN_RATE_LIMIT_COUNT)
  : 10;
const INSTALL_MAX_INFLIGHT = Number(process.env.INSTALL_MAX_INFLIGHT) > 0
  ? Number(process.env.INSTALL_MAX_INFLIGHT)
  : 1;
const RUN_MAX_INFLIGHT = Number(process.env.RUN_MAX_INFLIGHT) > 0
  ? Number(process.env.RUN_MAX_INFLIGHT)
  : 2;

// Persisted JSON stores live in the project dir.
const STORE = {
  instances: () => path.join(PROJECT_DIR, "instances.json"),
  targets: () => path.join(PROJECT_DIR, "targets.json"),
  jobs: () => path.join(PROJECT_DIR, "jobs.json"),
  runs: () => path.join(PROJECT_DIR, "runs.json"),
  prefs: () => path.join(PROJECT_DIR, "prefs.json"),
};

const ENGINE_VERSION_CACHE = { value: null };
const LOG_LEVELS = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};
const REQUEST_DRAIN_TIMEOUT_MS = 10_000;
const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES) > 0
  ? Number(process.env.MAX_BODY_BYTES)
  : 1024 * 1024;
const MIN_SUPPORTED_ENGINE_VERSION = "0.1.0";
const MAX_SUPPORTED_ENGINE_VERSION_EXCLUSIVE = "0.2.0";
const ACTIVE_ENVIRONMENT_PREF = "active-environment";
const PREVIEW_ROW_LIMIT = 20;
const VOLATILE_SECRET_PREFIX = "secret-session://";
const CONFIGURED_LOG_LEVEL = (() => {
  const raw = String(process.env.LOG_LEVEL || "info").toLowerCase();
  return Object.hasOwn(LOG_LEVELS, raw) ? raw : "info";
})();
const activeRequests = new Set();
const engineQueue = new KeyedAsyncQueue();
const operationStreams = new Map();
const sessionSecrets = new Map();
let shutdownState = null;
const requestMetrics = {
  requestsTotal: new Map(),
  errorsTotal: new Map(),
  requestDurationMs: [],
  runsStarted: 0,
  runsCompleted: 0,
  installsStarted: 0,
  installsCompleted: 0,
};
const rateLimitState = {
  install: { timestamps: [], inFlight: 0 },
  run: { timestamps: [], inFlight: 0 },
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function shouldLog(level) {
  return LOG_LEVELS[level] >= LOG_LEVELS[CONFIGURED_LOG_LEVEL];
}

function serializeError(err) {
  if (!err) return null;
  if (err instanceof Error) {
    return {
      name: err.name,
      message: err.message,
      stack: err.stack,
      code: err.code,
    };
  }
  return { message: String(err) };
}

function writeLog(level, msg, fields = {}) {
  if (!shouldLog(level)) return;
  const entry = {
    level,
    ts: nowIso(),
    msg,
    ...fields,
  };
  // eslint-disable-next-line no-console
  console.log(LOG_FORMAT === "json"
    ? JSON.stringify(entry)
    : `${entry.ts} ${level.toUpperCase()} ${msg}${Object.keys(fields).length ? ` ${JSON.stringify(fields)}` : ""}`);
}

function log(...args) {
  writeLog("info", args.map((arg) => {
    if (typeof arg === "string") return arg;
    try {
      return JSON.stringify(arg);
    } catch {
      return String(arg);
    }
  }).join(" "));
}

function requestLog(ctx, level, msg, fields = {}) {
  writeLog(level, msg, {
    request_id: ctx?.requestId ?? null,
    ...fields,
  });
}

function trackCliCommand(ctx, args) {
  if (!ctx) return;
  ctx.cliCommands.push(`celeritas ${args.join(" ")}`);
}

function tail(text, max = 4000) {
  if (!text) return "";
  const s = String(text);
  return s.length > max ? s.slice(s.length - max) : s;
}

function setSecurityHeaders(res) {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
}

function metricKey(parts) {
  return parts.join("|");
}

function incrementCounter(map, parts, amount = 1) {
  const key = metricKey(parts);
  map.set(key, (map.get(key) || 0) + amount);
}

function observeRequestMetrics(method, pathName, statusCode, durationMs) {
  incrementCounter(requestMetrics.requestsTotal, [method, pathName, String(statusCode)]);
  if (statusCode >= 400) {
    incrementCounter(requestMetrics.errorsTotal, [String(statusCode)]);
  }
  requestMetrics.requestDurationMs.push({ method, pathName, statusCode, durationMs });
  if (requestMetrics.requestDurationMs.length > 5000) {
    requestMetrics.requestDurationMs.shift();
  }
}

function responsePathLabel(pathName) {
  return String(pathName || "/").replace(/\/[A-Za-z0-9._-]{8,}(?=\/|$)/g, "/:id");
}

function parseRequestOrigin(req) {
  const origin = req.headers.origin;
  if (typeof origin === "string" && origin.trim()) return origin.trim();
  const referer = req.headers.referer;
  if (typeof referer === "string" && referer.trim()) {
    try {
      const parsed = new URL(referer);
      return parsed.origin;
    } catch {
      return null;
    }
  }
  return null;
}

function isWriteOriginAllowed(req) {
  if (!isMutatingMethod(req.method)) return true;
  if (API_TOKEN && req.headers.authorization === `Bearer ${API_TOKEN}`) return true;
  const origin = parseRequestOrigin(req);
  return origin ? isOriginAllowed(origin, CORS_ORIGINS) : false;
}

function normalizePathInsideRoots(input, { allowFile = true } = {}) {
  if (typeof input !== "string" || !input.trim()) {
    throw httpError(400, "path must be a non-empty string", {
      code: ERROR_CODES.VALIDATION,
      field: "path",
    });
  }
  const raw = input.replace(/^~(?=$|\/)/, os.homedir());
  const resolved = path.resolve(raw);
  const allowed = FS_ALLOWED_ROOTS.find((root) =>
    resolved === root || resolved.startsWith(`${root}${path.sep}`)
  );
  if (!allowed) {
    throw httpError(400, `path must stay within an allowed root (${FS_ALLOWED_ROOTS.join(", ")})`, {
      code: ERROR_CODES.VALIDATION,
      field: "path",
    });
  }
  if (!allowFile) {
    const stat = fs.existsSync(resolved) ? fs.statSync(resolved) : null;
    if (stat && !stat.isDirectory()) {
      throw httpError(400, "path must point to a directory", {
        code: ERROR_CODES.VALIDATION,
        field: "path",
      });
    }
  }
  return resolved;
}

function validateFilesystemLikeParams(values) {
  for (const [key, value] of Object.entries(values || {})) {
    if (typeof value !== "string" || !value.trim()) continue;
    if (/^[a-z]+:\/\//i.test(value)) continue;
    if (!/(^|_)(path|file|dir|directory)$/i.test(key)) continue;
    normalizePathInsideRoots(value);
  }
}

function appendAuditRecord(record) {
  const auditFile = path.join(PROJECT_DIR, "audit.log");
  fs.mkdirSync(path.dirname(auditFile), { recursive: true });
  fs.appendFileSync(auditFile, `${JSON.stringify(record)}\n`, "utf8");
}

function auditAction(requestContext, action, target, result, details = {}) {
  appendAuditRecord({
    ts: nowIso(),
    requestId: requestContext?.requestId ?? null,
    action,
    target,
    result,
    ...details,
  });
}

function withRateLimit(kind, countLimit, inFlightLimit, task) {
  const state = rateLimitState[kind];
  const now = Date.now();
  state.timestamps = state.timestamps.filter((ts) => now - ts < RATE_LIMIT_WINDOW_MS);
  if (state.timestamps.length >= countLimit || state.inFlight >= inFlightLimit) {
    const retryAfter = Math.ceil(RATE_LIMIT_WINDOW_MS / 1000);
    const err = httpError(429, `${kind} rate limit exceeded; retry later`, {
      code: ERROR_CODES.RATE_LIMITED,
      details: { retryAfter },
    });
    err.retryAfter = retryAfter;
    throw err;
  }
  state.timestamps.push(now);
  state.inFlight += 1;
  return Promise.resolve()
    .then(task)
    .finally(() => {
      state.inFlight = Math.max(0, state.inFlight - 1);
    });
}

async function reportServerError(kind, err, requestContext = null) {
  const webhook = process.env.CELERITAS_ERROR_REPORT_WEBHOOK?.trim();
  if (!webhook) return;
  try {
    const secretValues = [...sessionSecrets.values()];
    const payload = {
      kind,
      ts: nowIso(),
      requestId: requestContext?.requestId ?? null,
      message: redactText(err?.message || String(err), secretValues),
      code: err?.code || ERROR_CODES.INTERNAL,
      stack: redactText(err?.stack || "", secretValues),
    };
    await fetch(webhook, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    /* no-op */
  }
}

function renderPrometheusMetrics() {
  const lines = [
    "# HELP celeritas_requests_total Total HTTP requests handled by the sidecar.",
    "# TYPE celeritas_requests_total counter",
  ];
  for (const [key, value] of requestMetrics.requestsTotal.entries()) {
    const [method, pathName, status] = key.split("|");
    lines.push(`celeritas_requests_total{method="${method}",path="${pathName}",status="${status}"} ${value}`);
  }
  lines.push("# HELP celeritas_errors_total Total HTTP errors by status code.");
  lines.push("# TYPE celeritas_errors_total counter");
  for (const [key, value] of requestMetrics.errorsTotal.entries()) {
    lines.push(`celeritas_errors_total{status="${key}"} ${value}`);
  }
  lines.push("# HELP celeritas_runs_total Total sidecar run requests.");
  lines.push("# TYPE celeritas_runs_total counter");
  lines.push(`celeritas_runs_total{state="started"} ${requestMetrics.runsStarted}`);
  lines.push(`celeritas_runs_total{state="completed"} ${requestMetrics.runsCompleted}`);
  lines.push("# HELP celeritas_registry_installs_total Total registry installs.");
  lines.push("# TYPE celeritas_registry_installs_total counter");
  lines.push(`celeritas_registry_installs_total{state="started"} ${requestMetrics.installsStarted}`);
  lines.push(`celeritas_registry_installs_total{state="completed"} ${requestMetrics.installsCompleted}`);
  lines.push("# HELP celeritas_request_duration_ms Request duration histogram.");
  lines.push("# TYPE celeritas_request_duration_ms histogram");
  const buckets = [50, 100, 250, 500, 1000, 5000, 30000];
  const counts = new Map(buckets.map((bucket) => [bucket, 0]));
  let sum = 0;
  for (const sample of requestMetrics.requestDurationMs) {
    sum += sample.durationMs;
    for (const bucket of buckets) {
      if (sample.durationMs <= bucket) {
        counts.set(bucket, counts.get(bucket) + 1);
      }
    }
  }
  for (const bucket of buckets) {
    lines.push(`celeritas_request_duration_ms_bucket{le="${bucket}"} ${counts.get(bucket)}`);
  }
  lines.push(`celeritas_request_duration_ms_bucket{le="+Inf"} ${requestMetrics.requestDurationMs.length}`);
  lines.push(`celeritas_request_duration_ms_sum ${sum}`);
  lines.push(`celeritas_request_duration_ms_count ${requestMetrics.requestDurationMs.length}`);
  return `${lines.join("\n")}\n`;
}

function nowIso() {
  return new Date().toISOString();
}

function storeLog(...args) {
  writeLog("warn", args.join(" "));
}

function httpError(statusCode, message, extra = {}) {
  const err = appError(extra.code || ERROR_CODES.INTERNAL, statusCode, message, extra.details);
  if (extra.field) err.field = extra.field;
  return err;
}

function isSecretsKeeperRef(value) {
  return typeof value === "string" && value.startsWith("secrets-keeper://");
}

function isSessionSecretRef(value) {
  return typeof value === "string" && value.startsWith(VOLATILE_SECRET_PREFIX);
}

function makeSessionSecretRef(instanceId, paramName) {
  return `${VOLATILE_SECRET_PREFIX}${instanceId}/${paramName}`;
}

function activeEnvironmentName() {
  const prefs = loadPrefs();
  const value = prefs[ACTIVE_ENVIRONMENT_PREF];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function parseJsonObject(text) {
  const raw = String(text || "");
  const start = raw.indexOf("{");
  if (start < 0) return null;
  try {
    return JSON.parse(raw.slice(start));
  } catch {
    return null;
  }
}

function coerceConfigValue(value) {
  if (typeof value !== "string") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  try {
    const parsed = JSON.parse(value);
    if (
      typeof parsed === "number"
      || typeof parsed === "boolean"
      || Array.isArray(parsed)
      || (parsed && typeof parsed === "object")
    ) {
      return parsed;
    }
  } catch {
    /* fall back to string */
  }
  return value;
}

function compareSemver(a, b) {
  const parse = (value) => {
    const match = String(value || "").trim().match(/^(\d+)\.(\d+)\.(\d+)/);
    return match ? match.slice(1).map((part) => Number(part)) : null;
  };
  const left = parse(a);
  const right = parse(b);
  if (!left || !right) return null;
  for (let index = 0; index < 3; index += 1) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return 0;
}

function engineCompatibility(version) {
  const lower = compareSemver(version, MIN_SUPPORTED_ENGINE_VERSION);
  const upper = compareSemver(version, MAX_SUPPORTED_ENGINE_VERSION_EXCLUSIVE);
  const compatible = lower != null && upper != null && lower >= 0 && upper < 0;
  return {
    compatible,
    supportedRange: `>=${MIN_SUPPORTED_ENGINE_VERSION} <${MAX_SUPPORTED_ENGINE_VERSION_EXCLUSIVE}`,
  };
}

function runCliArgs(args, { useActiveEnvironment = true } = {}) {
  const activeEnvironment = useActiveEnvironment ? activeEnvironmentName() : null;
  if (!activeEnvironment) return args;
  if (args[0] === "environment" || args[0] === "init" || args[0] === "--version" || args[0] === "hub") {
    return args;
  }
  return ["--environment", activeEnvironment, ...args];
}

function redactText(text, secretValues = []) {
  if (!text) return text;
  let output = String(text);
  const uniqueSecrets = [...new Set(secretValues.filter((value) =>
    typeof value === "string" && value && !isSecretsKeeperRef(value) && !isSessionSecretRef(value)
  ))].sort((a, b) => b.length - a.length);
  for (const secret of uniqueSecrets) {
    output = output.split(secret).join("[redacted]");
  }
  return output;
}

function readJsonFile(file, fallback) {
  try {
    if (!fs.existsSync(file)) return fallback;
    const raw = fs.readFileSync(file, "utf8");
    if (!raw.trim()) return fallback;
    return JSON.parse(raw);
  } catch (err) {
    writeLog("warn", "readJsonFile failed", { file, message: err.message });
    return fallback;
  }
}

function createRequestContext(req, pathname) {
  return {
    active: true,
    cliCommands: [],
    method: req.method || "GET",
    path: pathname || req.url || "/",
    requestId: randomUUID(),
    startedAt: Date.now(),
  };
}

function ensureOperationStream(id) {
  let stream = operationStreams.get(id);
  if (!stream) {
    stream = {
      clients: new Set(),
      closed: false,
      history: [],
      purgeTimer: null,
    };
    operationStreams.set(id, stream);
  }
  if (stream.purgeTimer) {
    clearTimeout(stream.purgeTimer);
    stream.purgeTimer = null;
  }
  return stream;
}

function scheduleOperationStreamPurge(id) {
  const stream = operationStreams.get(id);
  if (!stream || stream.purgeTimer) return;
  stream.purgeTimer = setTimeout(() => {
    operationStreams.delete(id);
  }, 10 * 60 * 1000);
}

function publishOperationStream(id, event, data) {
  if (!id) return;
  const stream = ensureOperationStream(id);
  const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  stream.history.push(frame);
  if (stream.history.length > 200) {
    stream.history.shift();
  }
  for (const res of stream.clients) {
    try {
      res.write(frame);
    } catch {
      /* noop */
    }
  }
}

function closeOperationStream(id, finalData) {
  if (!id) return;
  const stream = ensureOperationStream(id);
  if (finalData) {
    publishOperationStream(id, "status", finalData);
  }
  stream.closed = true;
  for (const res of stream.clients) {
    try {
      res.end();
    } catch {
      /* noop */
    }
  }
  stream.clients.clear();
  scheduleOperationStreamPurge(id);
}

function attachOperationStream(id, res) {
  const stream = ensureOperationStream(id);
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.write("retry: 1000\n\n");
  for (const frame of stream.history) {
    res.write(frame);
  }
  if (stream.closed) {
    res.end();
    return;
  }
  stream.clients.add(res);
  res.on("close", () => {
    stream.clients.delete(res);
    if (stream.closed && stream.clients.size === 0) {
      scheduleOperationStreamPurge(id);
    }
  });
}

function finishShutdown() {
  if (!shutdownState || shutdownState.completed) return;
  shutdownState.completed = true;
  clearTimeout(shutdownState.timer);
  writeLog("info", "shutdown complete", {
    active_requests: activeRequests.size,
    signal: shutdownState.signal,
  });
  process.exit(0);
}

function beginShutdown(signal) {
  if (shutdownState) return;
  shutdownState = {
    completed: false,
    signal,
    timer: setTimeout(() => {
      writeLog("warn", "shutdown timeout exceeded; forcing exit", {
        active_requests: activeRequests.size,
        signal,
      });
      process.exit(0);
    }, REQUEST_DRAIN_TIMEOUT_MS),
  };
  writeLog("info", "shutdown initiated", {
    active_requests: activeRequests.size,
    signal,
    timeout_ms: REQUEST_DRAIN_TIMEOUT_MS,
  });
  server.close((err) => {
    if (err) {
      writeLog("error", "shutdown close error", { error: serializeError(err) });
    }
    if (activeRequests.size === 0) {
      finishShutdown();
    }
  });
  if (typeof server.closeIdleConnections === "function") {
    server.closeIdleConnections();
  }
  if (activeRequests.size === 0) {
    finishShutdown();
  }
}

function endTrackedRequest(ctx) {
  if (!ctx?.active) return;
  ctx.active = false;
  activeRequests.delete(ctx);
  if (shutdownState && activeRequests.size === 0) {
    finishShutdown();
  }
}

function attachRequestLifecycle(req, res, pathname) {
  const ctx = createRequestContext(req, pathname);
  req.requestContext = ctx;
  activeRequests.add(ctx);
  res.setHeader("x-request-id", ctx.requestId);
  let completed = false;
  const finalize = () => {
    if (completed) return;
    completed = true;
    const duration = Date.now() - ctx.startedAt;
    const cliCommands = ctx.cliCommands.length === 0
      ? undefined
      : ctx.cliCommands.length === 1
        ? ctx.cliCommands[0]
        : ctx.cliCommands;
    observeRequestMetrics(ctx.method, responsePathLabel(ctx.path), res.statusCode, duration);
    requestLog(ctx, "info", "request complete", {
      duration_ms: duration,
      method: ctx.method,
      path: ctx.path,
      status: res.statusCode,
      ...(cliCommands ? { celeritas_command: cliCommands } : {}),
    });
    endTrackedRequest(ctx);
  };
  res.once("finish", finalize);
  res.once("close", finalize);
  return ctx;
}

// ---------------------------------------------------------------------------
// CLI runner
// ---------------------------------------------------------------------------

let projectInitialized = false;

function ensureProjectDir() {
  fs.mkdirSync(PROJECT_PARENT, { recursive: true });
}

/** Idempotently `celeritas init project` under ~/.celeritas-ui. */
async function ensureProject(requestContext) {
  if (projectInitialized && fs.existsSync(path.join(PROJECT_DIR, "celeritas.yml"))) {
    return;
  }
  ensureProjectDir();
  if (!fs.existsSync(path.join(PROJECT_DIR, "celeritas.yml"))) {
    // init creates <cwd>/project
    const res = await runCliRaw(["init", "project"], {
      cwd: PROJECT_PARENT,
      requestContext,
      timeoutMs: 60000,
    });
    throwCliResultError(res, "celeritas init failed");
    if (res.code !== 0 && !fs.existsSync(path.join(PROJECT_DIR, "celeritas.yml"))) {
      throw appError(ERROR_CODES.ENGINE_FAILED, 502, `celeritas init failed: ${tail(res.stderr || res.stdout)}`);
    }
  }
  projectInitialized = true;
}

function engineAvailability() {
  if (!fs.existsSync(CELERITAS_BIN)) {
    return {
      ok: false,
      error: appError(
        ERROR_CODES.ENGINE_MISSING,
        503,
        `Celeritas binary not found at ${CELERITAS_BIN}. Build it or set CELERITAS_BIN to a valid executable.`,
      ),
    };
  }
  return { ok: true };
}

function createLineEmitter(streamId, event) {
  let buffer = "";
  return {
    push(chunk) {
      if (!streamId) return;
      buffer += chunk;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        publishOperationStream(streamId, event, { line });
      }
    },
    flush() {
      if (streamId && buffer) {
        publishOperationStream(streamId, event, { line: buffer });
      }
      buffer = "";
    },
  };
}

/** Spawn the binary with explicit cwd. Resolves with {code, stdout, stderr}. */
function runCliRaw(args, {
  cwd,
  timeoutMs = 60000,
  requestContext,
  displayArgs,
  streamId,
  envOverrides,
} = {}) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    let child;
    const stdoutEmitter = createLineEmitter(streamId, "stdout");
    const stderrEmitter = createLineEmitter(streamId, "stderr");
    const availability = engineAvailability();
    if (!availability.ok) {
      resolve({
        code: -1,
        codeName: availability.error.code,
        stdout: "",
        stderr: availability.error.message,
      });
      return;
    }
    const visibleArgs = Array.isArray(displayArgs) ? displayArgs : args;
    const displayCommand = `celeritas ${visibleArgs.join(" ")}`;
    trackCliCommand(requestContext, visibleArgs);
    publishOperationStream(streamId, "command", { command: displayCommand });
    try {
      child = spawn(CELERITAS_BIN, args, {
        cwd: cwd || PROJECT_DIR,
        env: { ...process.env, ...(envOverrides || {}) },
      });
    } catch (err) {
      requestLog(requestContext, "error", "failed to spawn celeritas", {
        command: displayCommand,
        error: serializeError(err),
      });
      const codeName = err?.code === "ENOENT" ? ERROR_CODES.ENGINE_MISSING : ERROR_CODES.ENGINE_FAILED;
      resolve({ code: -1, codeName, stdout: "", stderr: String(err && err.message) });
      return;
    }

    const timer = setTimeout(() => {
      if (settled) return;
      try {
        child.kill("SIGKILL");
      } catch {
        /* noop */
      }
      settled = true;
      stdoutEmitter.flush();
      stderrEmitter.flush();
      requestLog(requestContext, "warn", "celeritas command timed out", {
        command: displayCommand,
        timeout_ms: timeoutMs,
      });
      resolve({
        code: -1,
        codeName: ERROR_CODES.TIMEOUT,
        stdout,
        stderr: `${stderr}\n[timed out after ${timeoutMs}ms]`,
      });
    }, timeoutMs);

    child.stdout.on("data", (d) => {
      const chunk = d.toString();
      stdout += chunk;
      stdoutEmitter.push(chunk);
    });
    child.stderr.on("data", (d) => {
      const chunk = d.toString();
      stderr += chunk;
      stderrEmitter.push(chunk);
    });
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stdoutEmitter.flush();
      stderrEmitter.flush();
      requestLog(requestContext, "error", "celeritas process error", {
        command: displayCommand,
        error: serializeError(err),
      });
      const codeName = err?.code === "ENOENT" ? ERROR_CODES.ENGINE_MISSING : ERROR_CODES.ENGINE_FAILED;
      resolve({ code: -1, codeName, stdout, stderr: `${stderr}${err.message}` });
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stdoutEmitter.flush();
      stderrEmitter.flush();
      if ((code == null ? -1 : code) !== 0) {
        requestLog(requestContext, "warn", "celeritas command failed", {
          code: code == null ? -1 : code,
          command: displayCommand,
        });
      }
      resolve({
        code: code == null ? -1 : code,
        codeName: (code == null ? -1 : code) === 0 ? null : ERROR_CODES.ENGINE_FAILED,
        stdout,
        stderr,
      });
    });
  });
}

/** Runner that ensures the project exists first; default cwd = project dir. */
async function runCli(args, opts = {}) {
  await ensureProject(opts.requestContext);
  return runCliRaw(runCliArgs(args, opts), { cwd: PROJECT_DIR, ...opts });
}

// ---------------------------------------------------------------------------
// Catalog (connectors)
// ---------------------------------------------------------------------------

// Map catalog setting kind → ConnectorParam.kind (src/lib/types.ts).
function mapParamKind(kind) {
  switch ((kind || "string").toLowerCase()) {
    case "secret":
      return "secret";
    case "enum":
      return "enum";
    case "integer":
    case "number":
    case "float":
      return "number";
    case "boolean":
    case "bool":
      return "bool";
    case "url":
      return "url";
    case "path":
      return "path";
    case "object":
    case "json":
    case "string":
    default:
      return "string";
  }
}

function settingToParam(s) {
  const kind = mapParamKind(s.kind);
  return {
    name: s.name,
    label: s.title ?? null,
    kind,
    input: kind === "secret" ? "secret_ref" : null,
    group: null,
    advanced: false,
    sensitive: kind === "secret" || s.kind === "secret",
    dependsOn: null,
    min: null,
    max: null,
    unit: null,
    required: Boolean(s.required),
    default: s.default == null ? null : String(s.default),
    placeholder: null,
    help: s.description ?? null,
    options: Array.isArray(s.options) ? s.options : [],
  };
}

function catalogEntryToSpec(entry) {
  const kind = entry.type === "sink" || entry.type === "loader" || entry.type === "target"
    ? "target"
    : "source";
  const params = [];
  const jobParams = [];
  for (const s of entry.settings || []) {
    const param = settingToParam(s);
    // scope: connection → params, run → jobParams; null → connection.
    if (s.scope === "run") jobParams.push(param);
    else params.push(param);
  }
  const actions = kind === "source"
    ? [{ name: "preview", label: "Preview", flag: "--discover", description: "Discover the source catalog." }]
    : [];
  return {
    name: entry.name,
    description: entry.summary || entry.title || "Native connector.",
    kind,
    driver: entry.name,
    icon: null,
    category: entry.category || "files",
    params,
    jobParams,
    actions,
    example: {},
    exampleUnresolved: false,
    available: true,
    path: entry.repository ?? null,
    source: "hub",
  };
}

// Fallback: build catalog directly from `celeritas hub list` + the per-connector
// plugin.celeritas.yml manifests (used only if connectors.json is missing).
async function buildCatalogFromHub() {
  const res = await runCliRaw(["hub", "list"], { cwd: PROJECT_PARENT, timeoutMs: 60000 });
  if (res.code !== 0) return [];
  const specs = [];
  const lines = res.stdout.split("\n").map((l) => l.trim()).filter(Boolean);
  for (const line of lines) {
    const m = line.match(/^(\S+)\s+(\S+)\s+local:(\S+)\s*(.*)$/);
    if (!m) continue;
    const [, name, role, extPath, desc] = m;
    const kind = role === "loader" ? "target" : "source";
    // category from the path segment after source/sink.
    const segs = extPath.split(path.sep);
    let category = "files";
    const idx = segs.findIndex((s) => s === "source" || s === "sink");
    if (idx >= 0 && segs[idx + 1]) category = segs[idx + 1];
    const params = [];
    const jobParams = [];
    try {
      const yml = fs.readFileSync(path.join(extPath, "plugin.celeritas.yml"), "utf8");
      for (const s of parseSettingsFromYml(yml)) {
        const param = settingToParam(s);
        if (s.scope === "run") jobParams.push(param);
        else params.push(param);
      }
    } catch {
      /* manifest unreadable — leave params empty */
    }
    specs.push({
      name,
      description: desc || "Native connector.",
      kind,
      driver: name,
      icon: null,
      category,
      params,
      jobParams,
      actions: kind === "source"
        ? [{ name: "preview", label: "Preview", flag: "--discover", description: "Discover the source catalog." }]
        : [],
      example: {},
      exampleUnresolved: false,
      available: true,
      path: extPath,
      source: "hub",
    });
  }
  return specs;
}

// Minimal YAML settings extractor for plugin.celeritas.yml `settings:` blocks.
// Avoids a YAML dep; handles the simple list-of-maps the manifests use.
function parseSettingsFromYml(yml) {
  const lines = yml.split("\n");
  const settings = [];
  let inSettings = false;
  let settingsIndent = -1;
  let current = null;
  for (const raw of lines) {
    const line = raw.replace(/\t/g, "  ");
    if (!line.trim()) continue;
    const indent = line.length - line.trimStart().length;
    const trimmed = line.trim();
    if (/^settings:\s*$/.test(trimmed)) {
      inSettings = true;
      settingsIndent = indent;
      continue;
    }
    if (inSettings && indent <= settingsIndent && !trimmed.startsWith("-")) {
      // dedented out of settings block
      inSettings = false;
    }
    if (!inSettings) continue;
    if (trimmed.startsWith("- ")) {
      if (current) settings.push(current);
      current = {};
      const rest = trimmed.slice(2).trim();
      addYmlKv(current, rest);
    } else if (current) {
      addYmlKv(current, trimmed);
    }
  }
  if (current) settings.push(current);
  return settings.filter((s) => s.name);
}

function addYmlKv(obj, kv) {
  const m = kv.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
  if (!m) return;
  const key = m[1];
  let val = m[2].trim();
  if (val === "") return;
  // strip quotes
  if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
    val = val.slice(1, -1);
  }
  if (key === "options") {
    // ["a","b"]
    try {
      obj.options = JSON.parse(val.replace(/'/g, '"'));
    } catch {
      obj.options = val.replace(/[[\]]/g, "").split(",").map((s) => s.trim().replace(/['"]/g, "")).filter(Boolean);
    }
    return;
  }
  if (key === "required") obj.required = val === "true";
  else if (key === "default") obj.default = val;
  else obj[key] = val;
}

async function listConnectors() {
  // Primary: authoritative pre-generated catalog.
  const catalog = readJsonFile(CATALOG_JSON, null);
  if (Array.isArray(catalog) && catalog.length) {
    return catalog.map(catalogEntryToSpec);
  }
  // Fallback: live hub + manifests.
  return buildCatalogFromHub();
}

let CONNECTOR_INDEX = null;
async function connectorIndex() {
  if (CONNECTOR_INDEX) return CONNECTOR_INDEX;
  const specs = await listConnectors();
  CONNECTOR_INDEX = new Map(specs.map((s) => [s.name, s]));
  return CONNECTOR_INDEX;
}

async function secretParamNamesForDriver(driver) {
  const index = await connectorIndex();
  const spec = index.get(driver);
  if (!spec) return new Set();
  return new Set(
    [...(spec.params || []), ...(spec.jobParams || [])]
      .filter((param) => param.sensitive === true || param.input === "secret_ref" || param.kind === "secret")
      .map((param) => param.name),
  );
}

async function sanitizePersistedSecrets(driver, params, instanceId) {
  const secretNames = await secretParamNamesForDriver(driver);
  const persisted = {};
  for (const [key, value] of Object.entries(params || {})) {
    if (!secretNames.has(key) || typeof value !== "string" || !value) {
      persisted[key] = value;
      continue;
    }
    if (isSecretsKeeperRef(value) || isSessionSecretRef(value)) {
      persisted[key] = value;
      continue;
    }
    const ref = makeSessionSecretRef(instanceId, key);
    sessionSecrets.set(ref, value);
    persisted[key] = ref;
  }
  return persisted;
}

async function resolveConnectorSecrets(driver, params) {
  const secretNames = await secretParamNamesForDriver(driver);
  const resolved = {};
  const secretValues = [];
  for (const [key, value] of Object.entries(params || {})) {
    if (!secretNames.has(key) || typeof value !== "string" || !value) {
      resolved[key] = value;
      continue;
    }
    if (isSessionSecretRef(value)) {
      const secret = sessionSecrets.get(value);
      if (!secret) {
        throw httpError(400, `secret value for '${key}' is unavailable; re-enter it before running`, {
          code: ERROR_CODES.VALIDATION,
          field: key,
        });
      }
      resolved[key] = secret;
      secretValues.push(secret);
      continue;
    }
    resolved[key] = value;
    if (!isSecretsKeeperRef(value)) {
      secretValues.push(value);
    }
  }
  return { resolved, secretValues };
}

function connectorEnvKey(driver, key) {
  return `${String(driver || "").replace(/-/g, "_").toUpperCase()}__${String(key || "").toUpperCase()}`;
}

async function materializeConnectorSecrets(driver, params, secretNames, secretPrefix) {
  const persisted = {};
  const envOverrides = {};
  const secretValues = [];
  for (const [key, value] of Object.entries(params || {})) {
    if (!secretNames.has(key) || typeof value !== "string" || !value) {
      persisted[key] = value;
      continue;
    }
    if (isSecretsKeeperRef(value)) {
      persisted[key] = value;
      continue;
    }
    let ref = value;
    let secret = value;
    if (isSessionSecretRef(value)) {
      secret = sessionSecrets.get(value);
      if (!secret) {
        throw httpError(400, `secret value for '${key}' is unavailable; re-enter it before running`, {
          code: ERROR_CODES.VALIDATION,
          field: key,
        });
      }
    } else {
      ref = makeSessionSecretRef(secretPrefix, key);
      sessionSecrets.set(ref, value);
    }
    persisted[key] = ref;
    envOverrides[connectorEnvKey(driver, key)] = secret;
    secretValues.push(secret);
  }
  return { persisted, envOverrides, secretValues };
}

async function loadConnectorSchema(driver, requestContext, streamId) {
  const result = await runCli(["describe", driver, "--json-schema"], {
    requestContext,
    streamId,
    timeoutMs: 30000,
  });
  if (result.codeName === ERROR_CODES.ENGINE_MISSING) {
    throw engineAvailability().error;
  }
  if (result.codeName === ERROR_CODES.TIMEOUT) {
    throw httpError(504, "describe timed out", { code: ERROR_CODES.TIMEOUT });
  }
  const schema = result.code === 0 ? parseJsonObject(result.stdout) : null;
  if (schema && typeof schema === "object" && !Array.isArray(schema)) {
    return schema;
  }
  const index = await connectorIndex();
  const spec = index.get(driver);
  if (!spec) return null;
  const allParams = [...(spec.params || []), ...(spec.jobParams || [])];
  if (allParams.length === 0) return null;
  const properties = {};
  for (const param of allParams) {
    const property = {};
    switch (param.kind) {
      case "bool":
        property.type = "boolean";
        break;
      case "number":
        property.type = "number";
        break;
      case "enum":
        property.type = "string";
        property.enum = Array.isArray(param.options) ? param.options.map((option) => option.value) : [];
        break;
      default:
        property.type = "string";
        break;
    }
    properties[param.name] = property;
  }
  return {
    type: "object",
    properties,
  };
}

function collectSchemaIssues(schema, values, path = "") {
  const issues = [];
  const type = Array.isArray(schema?.type)
    ? schema.type.find((entry) => entry !== "null") || schema.type[0]
    : schema?.type;
  if (type === "object") {
    const objectValue = values && typeof values === "object" && !Array.isArray(values) ? values : {};
    const properties = schema?.properties && typeof schema.properties === "object" ? schema.properties : {};
    const required = Array.isArray(schema?.required) ? schema.required : [];
    for (const key of required) {
      const value = objectValue[key];
      if (value === undefined || value === null || value === "") {
        issues.push({
          field: path ? `${path}.${key}` : key,
          message: `${path ? `${path}.` : ""}${key} is required`,
        });
      }
    }
    for (const [key, childSchema] of Object.entries(properties)) {
      if (!(key in objectValue) || objectValue[key] === "" || objectValue[key] == null) continue;
      issues.push(...collectSchemaIssues(childSchema, objectValue[key], path ? `${path}.${key}` : key));
    }
    return issues;
  }
  if (values === undefined || values === null || values === "") {
    return issues;
  }
  if (Array.isArray(schema?.enum) && !schema.enum.includes(values)) {
    issues.push({
      field: path,
      message: `${path || "value"} must be one of ${schema.enum.join(", ")}`,
    });
    return issues;
  }
  const actualType = Array.isArray(values) ? "array" : typeof values;
  const normalizedType = actualType === "boolean"
    ? "boolean"
    : actualType === "number"
      ? Number.isInteger(values) ? "integer" : "number"
      : actualType === "string"
        ? "string"
        : actualType === "object"
          ? "object"
          : actualType;
  if (type === "integer" && !Number.isInteger(values)) {
    issues.push({ field: path, message: `${path || "value"} must be an integer` });
  } else if (type === "number" && typeof values !== "number") {
    issues.push({ field: path, message: `${path || "value"} must be a number` });
  } else if (type === "boolean" && typeof values !== "boolean") {
    issues.push({ field: path, message: `${path || "value"} must be a boolean` });
  } else if (type === "array" && !Array.isArray(values)) {
    issues.push({ field: path, message: `${path || "value"} must be an array` });
  } else if (type === "string" && typeof values !== "string") {
    issues.push({ field: path, message: `${path || "value"} must be a string` });
  } else if (type && type !== "number" && type !== "integer" && type !== "boolean" && type !== "array" && type !== "string" && type !== "object" && normalizedType !== type) {
    issues.push({ field: path, message: `${path || "value"} must be a ${type}` });
  }
  return issues;
}

async function validateConnectorConfig(driver, params, jobParams, requestContext, streamId) {
  const schema = await loadConnectorSchema(driver, requestContext, streamId);
  if (!schema) return;
  const candidate = {};
  for (const [key, value] of Object.entries({ ...(params || {}), ...(jobParams || {}) })) {
    if (value == null || value === "") continue;
    candidate[key] = coerceConfigValue(value);
  }
  const issues = collectSchemaIssues(schema, candidate);
  if (issues.length === 0) return;
  const err = appError(ERROR_CODES.VALIDATION, 400, issues[0].message, issues);
  err.field = issues[0].field;
  throw err;
}

function parseSelectionRules(stdout) {
  return String(stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => line.slice(2).trim())
    .filter((line) => !line.includes(":"));
}

function selectionCommandArgs(driver, rule, { remove = false } = {}) {
  const trimmed = String(rule || "").trim();
  if (!trimmed) return null;
  if (remove) {
    return ["select", driver, "--remove", trimmed];
  }
  const exclude = trimmed.startsWith("!");
  const body = exclude ? trimmed.slice(1) : trimmed;
  const lastDot = body.lastIndexOf(".");
  const stream = lastDot >= 0 ? body.slice(0, lastDot) : body;
  const property = lastDot >= 0 ? body.slice(lastDot + 1) : "*";
  const args = ["select", driver, stream];
  if (property && property !== "*") args.push(property);
  if (exclude) args.push("--exclude");
  return args;
}

async function syncSelectionRules(driver, selection = [], requestContext, streamId) {
  const desired = [...new Set((selection || []).map((rule) => String(rule || "").trim()).filter(Boolean))];
  const currentResult = await runCli(["select", driver], {
    requestContext,
    streamId,
    timeoutMs: 30000,
  });
  throwCliResultError(currentResult, "selection list failed", { classifyNonZero: true });
  const current = parseSelectionRules(currentResult.stdout);
  for (const rule of current) {
    if (desired.includes(rule)) continue;
    const args = selectionCommandArgs(driver, rule, { remove: true });
    const result = await runCli(args, { requestContext, streamId, timeoutMs: 30000 });
    throwCliResultError(result, "selection remove failed", { classifyNonZero: true });
  }
  for (const rule of desired) {
    if (current.includes(rule)) continue;
    const args = selectionCommandArgs(driver, rule);
    const result = await runCli(args, { requestContext, streamId, timeoutMs: 30000 });
    throwCliResultError(result, "selection add failed", { classifyNonZero: true });
  }
}

// ---------------------------------------------------------------------------
// Materializing connectors into the project
// ---------------------------------------------------------------------------

/** True if a plugin is already declared in the manifest. */
function manifestHasPlugin(name) {
  try {
    const yml = fs.readFileSync(path.join(PROJECT_DIR, "celeritas.yml"), "utf8");
    return new RegExp(`name:\\s*${name.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\b`).test(yml);
  } catch {
    return false;
  }
}

async function pluginKindForDriver(driver) {
  const index = await connectorIndex();
  const spec = index.get(driver);
  return spec?.kind === "target" ? "loader" : spec?.kind === "source" ? "extractor" : null;
}

async function lockFilePath(driver) {
  const pluginKind = await pluginKindForDriver(driver);
  if (!pluginKind) return null;
  return path.join(PROJECT_DIR, ".celeritas", "locks", pluginKind, `${driver}.lock.json`);
}

async function parseLockCheckResult(result) {
  const combined = `${result?.stdout || ""}\n${result?.stderr || ""}`;
  const lines = combined
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const statuses = [];
  for (const line of lines) {
    const match = line.match(/^(ok|stale)\s+(\S+)\s+`([^`]+)`$/);
    if (!match) continue;
    const [, rawState, pluginKind, driver] = match;
    const lockPath = await lockFilePath(driver);
    const hasLock = lockPath ? fs.existsSync(lockPath) : false;
    const status = rawState === "ok"
      ? { driver, pluginKind, status: "locked", message: null }
      : {
        driver,
        pluginKind,
        status: hasLock ? "stale" : "missing",
        message: hasLock ? "lockfile drift detected" : "lockfile missing",
      };
    statuses.push(status);
  }
  return statuses;
}

async function connectorLockStatus(driver, requestContext) {
  if (!manifestHasPlugin(driver)) {
    return { status: "untracked", message: "connector is not declared in the project manifest" };
  }
  const result = await runCli(["lock", "--check", driver], {
    requestContext,
    timeoutMs: 30000,
  });
  if (result.codeName === ERROR_CODES.ENGINE_MISSING) throw engineAvailability().error;
  if (result.codeName === ERROR_CODES.TIMEOUT) {
    return { status: "error", message: "lock check timed out" };
  }
  const statuses = await parseLockCheckResult(result);
  const match = statuses.find((entry) => entry.driver === driver);
  if (match) {
    return { status: match.status, message: match.message };
  }
  if (result.code !== 0) {
    const lockPath = await lockFilePath(driver);
    return {
      status: lockPath && fs.existsSync(lockPath) ? "stale" : "missing",
      message: lockPath && fs.existsSync(lockPath) ? "lockfile drift detected" : "lockfile missing",
    };
  }
  return result.code === 0
    ? { status: "locked", message: null }
    : { status: "error", message: tail(result.stderr || result.stdout) || "lock check failed" };
}

async function healthLockSummary(requestContext) {
  if (!fs.existsSync(path.join(PROJECT_DIR, "celeritas.yml"))) {
    return { healthy: true, stale: [], missing: [] };
  }
  const result = await runCli(["lock", "--check"], {
    requestContext,
    timeoutMs: 30000,
  });
  if (result.codeName === ERROR_CODES.ENGINE_MISSING) throw engineAvailability().error;
  if (result.codeName === ERROR_CODES.TIMEOUT) {
    return { healthy: false, stale: [], missing: [], message: "lock check timed out" };
  }
  const statuses = await parseLockCheckResult(result);
  if (result.code !== 0 && statuses.length === 0) {
    const missing = await Promise.all(
      loadInstances()
        .map((record) => record.driver)
        .filter((driver, index, all) => all.indexOf(driver) === index)
        .filter((driver) => manifestHasPlugin(driver))
        .map(async (driver) => {
          const lockPath = await lockFilePath(driver);
          return !lockPath || !fs.existsSync(lockPath) ? driver : null;
        }),
    );
    return {
      healthy: false,
      stale: [],
      missing: missing.filter(Boolean),
    };
  }
  return {
    healthy: statuses.every((entry) => entry.status === "locked"),
    stale: statuses.filter((entry) => entry.status === "stale").map((entry) => entry.driver),
    missing: statuses.filter((entry) => entry.status === "missing").map((entry) => entry.driver),
  };
}

async function refreshConnectorLock(driver, requestContext, streamId) {
  const result = await runCli(["lock", driver], {
    requestContext,
    streamId,
    timeoutMs: 30000,
  });
  throwCliResultError(result, "lock write failed", { classifyNonZero: true });
  return connectorLockStatus(driver, requestContext);
}

/** Serialize a param value for `config set` — objects/arrays as JSON. */
function configValue(v) {
  if (v == null) return "";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

/**
 * Ensure a connector is added to the project and its params configured.
 * Returns {ok, log}. Honest: surfaces add/config failures.
 */
async function materializeConnector(driver, params = {}, jobParams = {}, requestContext, streamId, autoInstall = false, selection = []) {
  assertSafeIdentifier("driver", driver);
  validateFilesystemLikeParams(params, "params");
  validateFilesystemLikeParams(jobParams, "jobParams");
  const resolvedParams = await resolveConnectorSecrets(driver, params);
  const resolvedJobParams = await resolveConnectorSecrets(driver, jobParams);
  await validateConnectorConfig(
    driver,
    resolvedParams.resolved || {},
    resolvedJobParams.resolved || {},
    requestContext,
    streamId,
  );
  const secretNames = await secretParamNamesForDriver(driver);
  const configSecretPrefix = requestContext?.requestId ? `${requestContext.requestId}/${driver}` : `runtime/${driver}`;
  const persistedParams = await materializeConnectorSecrets(driver, params, secretNames, configSecretPrefix);
  const persistedJobParams = await materializeConnectorSecrets(driver, jobParams, secretNames, `${configSecretPrefix}/job`);
  const secretValues = [...resolvedParams.secretValues, ...resolvedJobParams.secretValues];
  const envOverrides = {
    ...persistedParams.envOverrides,
    ...persistedJobParams.envOverrides,
  };
  const logs = [];
  if (!manifestHasPlugin(driver)) {
    if (!autoInstall) {
      return {
        ok: false,
        log: "",
        secretValues,
        error: appError(
          ERROR_CODES.NEEDS_INSTALL,
          409,
          `connector ${driver} is not installed; rerun with { autoInstall: true } or install it first`,
        ),
      };
    }
    const add = await runCli(["add", driver], { requestContext, streamId, timeoutMs: 120000 });
    logs.push(`$ celeritas add ${driver}\n${redactText(tail(add.stdout), secretValues)}${redactText(tail(add.stderr), secretValues)}`);
    if (add.codeName === ERROR_CODES.ENGINE_MISSING) {
      return { ok: false, log: logs.join("\n"), secretValues, error: engineAvailability().error };
    }
    if (add.codeName === ERROR_CODES.TIMEOUT) {
      return {
        ok: false,
        log: logs.join("\n"),
        secretValues,
        error: httpError(504, tail(add.stderr || add.stdout) || "celeritas add timed out", { code: ERROR_CODES.TIMEOUT }),
      };
    }
    if (add.code !== 0 && !manifestHasPlugin(driver)) {
      return { ok: false, log: logs.join("\n"), secretValues };
    }
  }
  await syncSelectionRules(driver, selection, requestContext, streamId);
  const all = { ...(persistedParams.persisted || {}), ...(persistedJobParams.persisted || {}) };
  for (const [k, v] of Object.entries(all)) {
    if (v == null || v === "") continue;
    const displayArgs = secretNames.has(k)
      ? ["config", driver, "set", k, "[redacted]"]
      : undefined;
    const set = await runCli(["config", driver, "set", k, configValue(v)], {
      displayArgs,
      requestContext,
      streamId,
      timeoutMs: 30000,
    });
    if (set.code !== 0) {
      logs.push(`config set ${k} failed: ${redactText(tail(set.stderr), secretValues)}`);
    }
  }
  const lock = await refreshConnectorLock(driver, requestContext, streamId);
  return { ok: true, log: logs.join("\n"), secretValues, envOverrides, lock };
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function setCors(res) {
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Max-Age", "86400");
}

function sendJson(res, status, body) {
  res.statusCode = status;
  setSecurityHeaders(res);
  if (!res.hasHeader("Cache-Control")) {
    res.setHeader("Cache-Control", "no-store");
  }
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

function sendError(res, status, message, extra = {}) {
  const err = appError(extra.code || ERROR_CODES.INTERNAL, status, String(message), extra.details);
  if (extra.field) err.field = extra.field;
  const body = errorBody(err);
  if (extra.field) body.field = extra.field;
  if (extra.retryAfter) {
    res.setHeader("Retry-After", String(extra.retryAfter));
  }
  sendJson(res, status, body);
}

function throwCliResultError(result, fallbackMessage, { classifyNonZero = false } = {}) {
  if (!result?.codeName || result.code === 0) return;
  if (result.codeName === ERROR_CODES.ENGINE_MISSING) {
    throw engineAvailability().error;
  }
  if (result.codeName === ERROR_CODES.TIMEOUT) {
    throw httpError(504, fallbackMessage || tail(result.stderr || result.stdout) || "celeritas command timed out", {
      code: ERROR_CODES.TIMEOUT,
    });
  }
  if (classifyNonZero) {
    throw classifyEngineFailure(result, fallbackMessage);
  }
  if (result.code !== -1) return;
  throw httpError(502, fallbackMessage || tail(result.stderr || result.stdout) || "celeritas command failed", {
    code: ERROR_CODES.ENGINE_FAILED,
  });
}

function sendCacheableJson(req, res, body, { maxAgeSeconds = 60 } = {}) {
  const payload = JSON.stringify(body);
  const etag = `"${createHash("sha1").update(payload).digest("hex")}"`;
  setSecurityHeaders(res);
  res.setHeader("Cache-Control", `private, max-age=${maxAgeSeconds}`);
  res.setHeader("ETag", etag);
  if (req.headers["if-none-match"] === etag) {
    res.statusCode = 304;
    res.end();
    return;
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(payload);
}

function applyCors(req, res) {
  const origin = req.headers.origin;
  if (!origin) return true;
  if (!isOriginAllowed(origin, CORS_ORIGINS)) return false;
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Vary", "Origin");
  setCors(res);
  return true;
}

function isAuthorized(req, pathname) {
  if (
    !API_TOKEN
    || pathname === "/api/health"
    || pathname === "/api/livez"
    || pathname === "/api/readyz"
    || pathname === "/health"
  ) return true;
  const auth = req.headers.authorization;
  return auth === `Bearer ${API_TOKEN}`;
}

async function readValidatedBody(req, routeName) {
  return validateBody(routeName, await readBody(req));
}

async function withEngineLock(task) {
  return engineQueue.run(PROJECT_DIR, task);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    let size = 0;
    let overflow = false;
    req.on("data", (chunk) => {
      if (overflow) return;
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        overflow = true;
        data = "";
        req.resume();
        return;
      }
      data += chunk;
    });
    req.on("end", () => {
      if (overflow) {
        return reject(httpError(413, `request body too large; max ${MAX_BODY_BYTES} bytes`, {
          code: ERROR_CODES.VALIDATION,
        }));
      }
      if (!data.trim()) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch (err) {
        reject(httpError(400, `invalid JSON body: ${err.message}`, { code: ERROR_CODES.VALIDATION }));
      }
    });
    req.on("error", reject);
  });
}

function genId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Engine info
// ---------------------------------------------------------------------------

async function engineVersion(requestContext) {
  if (ENGINE_VERSION_CACHE.value) return ENGINE_VERSION_CACHE.value;
  ensureProjectDir();
  const res = await runCliRaw(["--version"], {
    cwd: PROJECT_PARENT,
    requestContext,
    timeoutMs: 15000,
    useActiveEnvironment: false,
  });
  const out = `${res.stdout} ${res.stderr}`.trim();
  const m = out.match(/(\d+\.\d+\.\d+\S*)/);
  if (m) {
    ENGINE_VERSION_CACHE.value = m[1];
    return ENGINE_VERSION_CACHE.value;
  }
  // Don't cache a transient failure — retry on the next health probe.
  return out || "unknown";
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async function handleHealth(_req, res) {
  // Must respond even before the project is initialized.
  let version = "unknown";
  const availability = engineAvailability();
  try {
    if (availability.ok) {
      version = await engineVersion(_req.requestContext);
    }
  } catch {
    /* keep unknown */
  }
  let locks = { healthy: true, stale: [], missing: [] };
  if (availability.ok) {
    try {
      locks = await healthLockSummary(_req.requestContext);
    } catch {
      locks = { healthy: false, stale: [], missing: [] };
    }
  }
  const compatibility = engineCompatibility(version);
  sendJson(res, 200, {
    ok: true,
    engine: availability.ok ? "celeritas" : false,
    ...(availability.ok ? {} : { code: ERROR_CODES.ENGINE_MISSING, message: availability.error.message }),
    version,
    buildVersion: BUILD_VERSION,
    buildSha: BUILD_SHA,
    compatible: availability.ok ? compatibility.compatible : false,
    supportedRange: compatibility.supportedRange,
    locks,
    project: PROJECT_DIR,
  });
}

async function handleLivez(_req, res) {
  sendJson(res, 200, { ok: true });
}

async function handleReadyz(req, res) {
  const availability = engineAvailability();
  if (!availability.ok) {
    return sendError(res, 503, availability.error.message, {
      code: availability.error.code,
    });
  }
  try {
    await ensureProject(req.requestContext);
    loadInstances();
    loadJobs();
    loadTargets();
    loadPrefs();
    await engineVersion(req.requestContext);
    sendJson(res, 200, { ok: true });
  } catch (err) {
    sendError(res, 503, err?.message || "server not ready", {
      code: err?.code || ERROR_CODES.ENGINE_FAILED,
    });
  }
}

async function handleEngine(_req, res) {
  const maxCores = os.cpus()?.length || 1;
  const envOverride = Boolean(process.env.CELERITAS_CORES);
  const cores = envOverride ? Number(process.env.CELERITAS_CORES) || maxCores : maxCores;
  /** @type {import('../src/lib/types').EmbeddedConfig} */
  const cfg = {
    cores,
    maxCores,
    running: 1,
    envOverride,
  };
  sendJson(res, 200, cfg);
}

async function handleConnectors(_req, res) {
  const specs = await listConnectors();
  sendCacheableJson(_req, res, specs);
}

async function handleConnectorTest(req, res) {
  const body = await readValidatedBody(req, "connectorTest");
  const driver = body.driver || body.spec;
  const kind = body.kind;
  if (!driver) return sendError(res, 400, "missing driver/spec", { code: ERROR_CODES.VALIDATION });
  const started = Date.now();
  const out = await withEngineLock(async () => {
    const mat = await materializeConnector(
      driver,
    body.params,
    body.jobParams,
    req.requestContext,
    null,
    body.autoInstall === true,
    body.selection || [],
  );
    if (mat.error) throw mat.error;
    if (!mat.ok) {
      return {
        status: "fail",
        message: redactText(tail(mat.log), mat.secretValues) || "failed to add connector to project",
        elapsedMs: Date.now() - started,
      };
    }
    // source → discover (success ⇒ connects); sink → test.
    const cmd = kind === "target" ? ["test", driver] : ["discover", driver, "--refresh"];
    const result = await runCli(cmd, {
      requestContext: req.requestContext,
      timeoutMs: 600000,
      envOverrides: mat.envOverrides,
    });
    throwCliResultError(result, "connector command failed");
    const pass = result.code === 0;
    const classified = pass ? null : classifyEngineFailure(result, "connector command failed");
    return {
      status: pass ? "pass" : "fail",
      ...(classified ? { code: classified.code } : {}),
      message: pass
        ? redactText(tail(result.stdout), mat.secretValues) || "Connected."
        : redactText(tail(result.stderr), mat.secretValues) || redactText(tail(result.stdout), mat.secretValues) || "Connection failed.",
      elapsedMs: Date.now() - started,
    };
  });
  sendJson(res, 200, out);
}

async function handleConnectorInspect(req, res) {
  const body = await readValidatedBody(req, "connectorInspect");
  const driver = body.driver || body.spec;
  if (!driver) return sendError(res, 400, "missing driver/spec", { code: ERROR_CODES.VALIDATION });
  const schema = await withEngineLock(async () => {
    const mat = await materializeConnector(
      driver,
      body.params,
      body.jobParams,
      req.requestContext,
      null,
      body.autoInstall === true,
      body.selection || [],
    );
    if (mat.error) throw mat.error;
    if (!mat.ok) {
      throw httpError(502, redactText(tail(mat.log), mat.secretValues) || "failed to materialize connector", {
        code: ERROR_CODES.ENGINE_FAILED,
      });
    }
    const result = await runCli(["discover", driver, "--refresh"], {
      requestContext: req.requestContext,
      timeoutMs: 600000,
      envOverrides: mat.envOverrides,
    });
    throwCliResultError(result, "discover failed", { classifyNonZero: true });
    if (result.code !== 0) {
      throw httpError(
        502,
        redactText(tail(result.stderr), mat.secretValues) || redactText(tail(result.stdout), mat.secretValues) || "discover failed",
        { code: ERROR_CODES.ENGINE_FAILED },
      );
    }
    return parseDiscoverCatalog(result.stdout, driver);
  });
  sendJson(res, 200, schema);
}

/** Build a SourceSchema from discover stdout or the cached catalog.json. */
function parseDiscoverCatalog(stdout, driver) {
  /** @type {{columns:{name:string,dataType:string}[],preview:Record<string,unknown>[]}} */
  const out = { columns: [], preview: [] };
  let catalog = null;
  // Discover prints JSON to stdout; also caches under .celeritas/run/<tap>/catalog.json.
  const jsonStart = stdout.indexOf("{");
  if (jsonStart >= 0) {
    try {
      catalog = JSON.parse(stdout.slice(jsonStart));
    } catch {
      /* fall through to cache */
    }
  }
  if (!catalog) {
    const cached = path.join(PROJECT_DIR, ".celeritas", "run", driver, "catalog.json");
    catalog = readJsonFile(cached, null);
  }
  if (!catalog) return out;
  const streams = Array.isArray(catalog.streams || catalog.Streams)
    ? (catalog.streams || catalog.Streams)
    : [];
  const columnMap = new Map();
  for (const stream of streams) {
    const streamObj = stream?.stream && typeof stream.stream === "object" ? stream.stream : stream;
    const props = streamObj?.schema?.properties || streamObj?.schema?.Properties || {};
    for (const [name, def] of Object.entries(props)) {
      if (columnMap.has(name)) continue;
      let dataType = "string";
      if (def && typeof def === "object") {
        const t = def.type;
        dataType = Array.isArray(t) ? t.find((x) => x !== "null") || "string" : t || "string";
      }
      columnMap.set(name, { name, dataType: String(dataType) });
    }
    const preview = streamObj?.preview || streamObj?.sample || streamObj?.rows || stream?.preview || stream?.sample || stream?.rows;
    if (out.preview.length === 0 && Array.isArray(preview)) {
      out.preview = preview
        .filter((row) => row && typeof row === "object" && !Array.isArray(row))
        .slice(0, PREVIEW_ROW_LIMIT);
    }
  }
  out.columns = [...columnMap.values()];
  return out;
}

function parseValidationResult(result) {
  if (result.code === 0) {
    return { ok: true, issues: [] };
  }
  const issues = [];
  const lines = String(result.stderr || result.stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    const match = line.match(/^([^:]+):\s+(.+)$/);
    if (match && match[1].startsWith("/")) {
      issues.push({ pointer: match[1], message: match[2] });
    }
  }
  return { ok: result.code === 0, issues };
}

function parseScheduleList(stdout) {
  const lines = String(stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0 || /No schedules defined\./i.test(lines[0])) return [];
  return lines.map((line) => {
    const match = line.match(/^(\S+)\s+(\S+)\s+(.+?)\s+(enabled|disabled)\s+(\S+)$/);
    if (!match) {
      return { name: line, raw: line, drift: "unparsed" };
    }
    return {
      name: match[1],
      job: match[2],
      interval: match[3],
      enabled: match[4] === "enabled",
      startDate: match[5] === "-" ? null : match[5],
      raw: line,
    };
  });
}

function isValidCronExpression(value) {
  const fields = String(value || "").trim().split(/\s+/);
  if (fields.length !== 5) return false;
  return fields.every((field) =>
    field === "*"
      || /^(\*|\d+)(-\d+)?(\/\d+)?(,(\*|\d+)(-\d+)?(\/\d+)?)*$/.test(field)
  );
}

function parseJobList(stdout) {
  const lines = String(stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0 || /No runs recorded yet\./i.test(lines[0])) return [];
  return lines.map((line) => {
    const match = line.match(/^(\S+)\s+(\S+)\s+(.+?)\s+\(([^)]+)\)$/);
    if (!match) return null;
    const [, status, duration, name, runId] = match;
    const epochMatch = runId.match(/-(\d{9,})$/);
    const startedAt = epochMatch ? new Date(Number(epochMatch[1]) * 1000).toISOString() : null;
    return {
      id: runId,
      status: status === "success" ? "complete" : status,
      duration,
      name,
      startedAt,
      runId,
    };
  }).filter(Boolean);
}

async function loadEngineStateMap(requestContext) {
  const result = await runCli(["state", "list"], { requestContext, timeoutMs: 30000 });
  throwCliResultError(result, "state list failed", { classifyNonZero: true });
  const lines = String(result.stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/No state recorded yet\./i.test(line));
  const entries = await Promise.all(lines.map(async (stateId) => {
    const stateResult = await runCli(["state", "get", stateId], {
      requestContext,
      timeoutMs: 30000,
    });
    if (stateResult.code !== 0) return [stateId, null];
    try {
      return [stateId, JSON.parse(stateResult.stdout)];
    } catch {
      return [stateId, null];
    }
  }));
  return new Map(entries);
}

async function handleConnectorAction(req, res) {
  const body = await readValidatedBody(req, "connectorAction");
  const driver = body.driver || body.spec;
  const action = body.action || "preview";
  if (!driver) return sendError(res, 400, "missing driver/spec", { code: ERROR_CODES.VALIDATION });
  const started = Date.now();
  const out = await withEngineLock(async () => {
    const mat = await materializeConnector(
      driver,
      body.params,
      body.jobParams,
      req.requestContext,
      null,
      body.autoInstall === true,
      body.selection || [],
    );
    if (mat.error) throw mat.error;
    if (!mat.ok) {
      return {
        status: "failed",
        message: redactText(tail(mat.log), mat.secretValues) || "failed to materialize connector",
        rows: [],
        elapsedMs: Date.now() - started,
      };
    }
    // preview / discover are the only safe in-place actions for sources.
    const result = await runCli(["discover", driver, "--refresh"], {
      requestContext: req.requestContext,
      timeoutMs: 600000,
      envOverrides: mat.envOverrides,
    });
    throwCliResultError(result, "connector action failed");
    const ok = result.code === 0;
    const classified = ok ? null : classifyEngineFailure(result, "connector action failed");
    const schema = ok ? parseDiscoverCatalog(result.stdout, driver) : { columns: [], preview: [] };
    return {
      status: ok ? "complete" : "failed",
      ...(classified ? { code: classified.code } : {}),
      message: ok
        ? `Action '${action}' completed (${schema.columns.length} columns).`
        : redactText(tail(result.stderr), mat.secretValues) || "Action failed.",
      rows: schema.preview || [],
      elapsedMs: Date.now() - started,
    };
  });
  sendJson(res, 200, out);
}

// ---- Connector instances ----

function loadInstances() {
  return readStoreFile(STORE.instances(), [], { log: storeLog });
}

async function handleInstancesList(_req, res) {
  const list = await Promise.all(loadInstances().map(async (record) => ({
    ...record,
    lock: await connectorLockStatus(record.driver, _req.requestContext),
  })));
  sendJson(res, 200, list);
}

async function handleInstancesSave(req, res) {
  const body = await readBody(req);
  if (!body || !body.driver) return sendError(res, 400, "ConnectorInstanceRecord requires driver", { code: ERROR_CODES.VALIDATION });
  const instanceId = body.id || genId("ci");
  const lock = await connectorLockStatus(body.driver, req.requestContext);
  const record = {
    id: instanceId,
    driver: body.driver,
    name: body.name || body.driver,
    params: await sanitizePersistedSecrets(body.driver, body.params || {}, instanceId),
    jobParamDefaults: await sanitizePersistedSecrets(body.driver, body.jobParamDefaults || {}, instanceId),
    selection: Array.isArray(body.selection) ? body.selection.filter((rule) => typeof rule === "string" && rule.trim()) : [],
    lock,
  };
  const list = loadInstances();
  const idx = list.findIndex((x) => x.id === record.id);
  if (idx >= 0) list[idx] = record;
  else list.push(record);
  writeStoreFile(STORE.instances(), list);
  sendJson(res, 200, record);
}

async function handleInstancesDelete(id, _req, res) {
  const list = loadInstances().filter((x) => x.id !== id);
  writeStoreFile(STORE.instances(), list);
  auditAction(_req.requestContext, "connector.delete", id, "ok");
  sendJson(res, 200, { ok: true });
}

// ---- Targets ----

const EMBEDDED_TARGET = { id: "embedded", name: "Local engine", kind: "embedded", url: "" };

function loadTargets() {
  const stored = readStoreFile(STORE.targets(), [], { log: storeLog });
  const out = [EMBEDDED_TARGET, ...stored.filter((t) => t.id !== "embedded" && t.kind === "embedded")];
  return out;
}

async function handleTargetsList(_req, res) {
  sendJson(res, 200, loadTargets());
}

async function handleTargetsSave(req, res) {
  const body = await readValidatedBody(req, "targetSave");
  if (!body || !body.name) return sendError(res, 400, "Target requires a name", { code: ERROR_CODES.VALIDATION });
  if (body.id === "embedded") return sendJson(res, 200, EMBEDDED_TARGET);
  if ((body.kind || "remote") !== "embedded") {
    return sendError(res, 501, "remote targets are not available yet; use the embedded local engine", {
      code: ERROR_CODES.CONFIG_INVALID,
    });
  }
  const target = {
    id: body.id || genId("target"),
    name: body.name,
    kind: "embedded",
    url: body.url || "",
    ...(body.token ? { token: body.token } : {}),
  };
  const stored = readStoreFile(STORE.targets(), [], { log: storeLog }).filter((t) => t.id !== "embedded");
  const idx = stored.findIndex((t) => t.id === target.id);
  if (idx >= 0) stored[idx] = target;
  else stored.push(target);
  writeStoreFile(STORE.targets(), stored);
  sendJson(res, 200, target);
}

async function handleTargetsDelete(id, _req, res) {
  if (id === "embedded") return sendError(res, 400, "cannot delete the embedded target", { code: ERROR_CODES.VALIDATION });
  const stored = readStoreFile(STORE.targets(), [], { log: storeLog }).filter((t) => t.id !== id && t.id !== "embedded");
  writeStoreFile(STORE.targets(), stored);
  auditAction(_req.requestContext, "target.delete", id, "ok");
  sendJson(res, 200, { ok: true });
}

// ---- Jobs ----

function loadJobs() {
  return readStoreFile(STORE.jobs(), [], { log: storeLog });
}

async function handleJobsList(_req, res) {
  sendJson(res, 200, loadJobs());
}

async function handleJobsSave(req, res) {
  const body = await readValidatedBody(req, "jobSave");
  if (!body || !body.name) return sendError(res, 400, "Job requires a name", { code: ERROR_CODES.VALIDATION });
  const job = {
    id: body.id || genId("job"),
    name: body.name,
    description: body.description || "",
    definition: body.definition || { source: "", steps: [] },
    enabled: body.enabled !== false,
  };
  const list = loadJobs();
  const idx = list.findIndex((x) => x.id === job.id);
  if (idx >= 0) list[idx] = job;
  else list.push(job);
  writeStoreFile(STORE.jobs(), list);
  sendJson(res, 200, job);
}

async function handleJobsDelete(id, _req, res) {
  const list = loadJobs().filter((x) => x.id !== id);
  writeStoreFile(STORE.jobs(), list);
  auditAction(_req.requestContext, "job.delete", id, "ok");
  sendJson(res, 200, { ok: true });
}

// ---- Run helpers ----

function loadRuns() {
  return readStoreFile(STORE.runs(), [], { log: storeLog });
}

function saveRun(run) {
  const list = loadRuns();
  list.unshift(run);
  const kept = list.slice(0, RUN_HISTORY_LIMIT);
  writeStoreFile(STORE.runs(), kept);
}

function saveRunSteps(runId, steps) {
  const map = readStoreFile(RUN_STEPS_FILE, {}, { log: storeLog });
  map[runId] = steps;
  for (const staleRunId of Object.keys(map)) {
    if (!loadRuns().some((run) => run.id === staleRunId)) {
      delete map[staleRunId];
    }
  }
  writeStoreFile(RUN_STEPS_FILE, map);
}

function loadRunSteps(runId) {
  const map = readStoreFile(RUN_STEPS_FILE, {}, { log: storeLog });
  return map[runId] || [];
}

/** Resolve a job's source + sink driver from the UI's job model. */
function resolveJobConnectors(job) {
  const def = job.definition || {};
  const instances = loadInstances();
  const byId = new Map(instances.map((i) => [i.id, i]));
  // Source: explicit driver name, or via a bound source instance.
  let sourceDriver = def.source;
  let sourceParams = {};
  let sourceJobParams = def.sourceConnectorParams || {};
  const srcInst = def.sourceConnectorId ? byId.get(def.sourceConnectorId) : null;
  let sourceSelection = [];
  if (srcInst) {
    sourceDriver = srcInst.driver;
    sourceParams = srcInst.params || {};
    sourceJobParams = { ...(srcInst.jobParamDefaults || {}), ...sourceJobParams };
    sourceSelection = Array.isArray(srcInst.selection) ? srcInst.selection : [];
  }
  // Sink: from steps[0].template, or via a bound target instance.
  let sinkDriver = def.steps?.[0]?.template;
  let sinkParams = {};
  let sinkJobParams = def.targetConnectorParams || {};
  const tgtInst = def.targetConnectorId ? byId.get(def.targetConnectorId) : null;
  if (tgtInst) {
    sinkDriver = tgtInst.driver;
    sinkParams = tgtInst.params || {};
    sinkJobParams = { ...(tgtInst.jobParamDefaults || {}), ...sinkJobParams };
  }
  return { sourceDriver, sourceParams, sourceJobParams, sourceSelection, sinkDriver, sinkParams, sinkJobParams };
}

async function executeJob(job, {
  autoInstall = false,
  dryRun,
  fullRefresh = false,
  requestContext,
  runId = genId("run"),
  streamId,
} = {}) {
  const startedAt = nowIso();
  const resolved = resolveJobConnectors(job);
  const { sourceDriver, sinkDriver } = resolved;

  if (!sourceDriver || (!sinkDriver && !dryRun)) {
    const run = {
      id: runId,
      jobId: job.id,
      status: "failed",
      trigger: "manual",
      startedAt,
      finishedAt: nowIso(),
    };
    const step = {
      runId,
      stepIdx: 0,
      template: sinkDriver || sourceDriver || "?",
      args: {},
      status: "failed",
      rowCount: null,
      log: "Job is missing a source or sink connector.",
    };
    saveRun(run);
    saveRunSteps(runId, [step]);
    return { run, steps: [step] };
  }
  assertSafeIdentifier("sourceDriver", sourceDriver);
  if (sinkDriver) {
    assertSafeIdentifier("sinkDriver", sinkDriver);
  }

  const logs = [];
  // Materialize both connectors.
  const matSrc = await materializeConnector(
    sourceDriver,
    resolved.sourceParams,
    resolved.sourceJobParams,
    requestContext,
    streamId,
    autoInstall,
    resolved.sourceSelection,
  );
  if (matSrc.error) throw matSrc.error;
  logs.push(matSrc.log);
  if (!matSrc.ok) {
    throw httpError(502, redactText(tail(matSrc.log), matSrc.secretValues) || "failed to materialize source connector", {
      code: ERROR_CODES.ENGINE_FAILED,
    });
  }
  let matSink = { ok: true, log: "", secretValues: [] };
  if (!dryRun) {
    matSink = await materializeConnector(
      sinkDriver,
      resolved.sinkParams,
      resolved.sinkJobParams,
      requestContext,
      streamId,
      autoInstall,
    );
    if (matSink.error) throw matSink.error;
    logs.push(matSink.log);
    if (!matSink.ok) {
      throw httpError(502, redactText(tail(matSink.log), matSink.secretValues) || "failed to materialize target connector", {
        code: ERROR_CODES.ENGINE_FAILED,
      });
    }
  }
  const secretValues = [...matSrc.secretValues, ...(matSink.secretValues || [])];
  const envOverrides = {
    ...(matSrc.envOverrides || {}),
    ...(matSink.envOverrides || {}),
  };

  let result;
  if (dryRun) {
    const validation = await runCli(["validate"], {
      requestContext,
      streamId,
      timeoutMs: 30000,
      envOverrides,
    });
    const validationResult = parseValidationResult(validation);
    if (!validationResult.ok) {
      throw classifyEngineFailure({
        code: validation.code,
        stderr: validation.stderr,
        stdout: validation.stdout,
      }, "validation failed");
    }
    result = await runCli(["discover", sourceDriver, "--refresh"], {
      requestContext,
      streamId,
      timeoutMs: 600000,
      envOverrides,
    });
    result.validation = validationResult;
  } else {
    const args = ["run", sourceDriver, sinkDriver];
    if (fullRefresh) args.push("--full-refresh");
    result = await runCli(args, {
      requestContext,
      streamId,
      timeoutMs: 600000,
      envOverrides,
    });
  }
  logs.push(redactText(tail(result.stdout), secretValues));
  logs.push(redactText(tail(result.stderr), secretValues));
  throwCliResultError(result, dryRun ? "dry run failed" : "job run failed");

  const ok = result.code === 0;
  const finishedAt = nowIso();
  const run = {
    id: runId,
    jobId: job.id,
    status: ok ? "complete" : "failed",
    trigger: "manual",
    startedAt,
    finishedAt,
    ...(dryRun
      ? {
        metadata: {
          diagnostics: {
            fullRefresh,
            validation: result.validation || { ok: true, issues: [] },
          },
        },
      }
      : fullRefresh
        ? { metadata: { diagnostics: { fullRefresh: true } } }
        : {}),
  };

  let steps = [];
  if (dryRun) {
    const schema = ok ? parseDiscoverCatalog(result.stdout, sourceDriver) : { columns: [] };
    steps = [
      {
        runId,
        stepIdx: 0,
        template: sinkDriver || "dry-run",
        args: {
          source: sourceDriver,
          columns: schema.columns.map((column) => column.name),
        },
        status: ok ? "complete" : "failed",
        rowCount: ok ? schema.columns.length : null,
        log: ok ? `Dry run: discovered ${schema.columns.length} columns.` : redactText(tail(result.stderr), secretValues),
      },
    ];
    run.metadata.diagnostics.schema = schema;
  } else {
    const { streams, total } = parseRunSummary(result.stdout);
    if (streams.length) {
      steps = streams.map((s, i) => ({
        runId,
        stepIdx: i,
        template: sinkDriver,
        args: { source: sourceDriver, stream: s.stream },
          status: ok ? "complete" : "failed",
          rowCount: s.count,
          log: i === 0 ? redactText(tail(logs.filter(Boolean).join("\n")), secretValues) : null,
        }));
    } else {
      steps = [
        {
          runId,
          stepIdx: 0,
          template: sinkDriver,
          args: { source: sourceDriver },
          status: ok ? "complete" : "failed",
          rowCount: ok ? total : null,
          log: redactText(tail(ok ? result.stdout : result.stderr || result.stdout), secretValues),
        },
      ];
    }
  }

  saveRun(run);
  saveRunSteps(runId, steps);
  return { run, steps };
}

async function handleJobRun(id, req, res, { dryRun } = {}) {
  const body = await readBody(req).catch(() => ({}));
  const job = loadJobs().find((j) => j.id === id);
  if (!job) return sendError(res, 404, `job ${id} not found`, { code: ERROR_CODES.NOT_FOUND });
  const runId = typeof body?.runId === "string" && body.runId.trim() ? assertSafeIdentifier("runId", body.runId.trim()) : genId("run");
  publishOperationStream(runId, "status", { state: "started", runId });
  requestMetrics.runsStarted += 1;
  try {
    const out = await withRateLimit("run", RUN_RATE_LIMIT_COUNT, RUN_MAX_INFLIGHT, () =>
      withEngineLock(() => executeJob(job, {
        autoInstall: body?.autoInstall === true,
        dryRun,
        fullRefresh: body?.fullRefresh === true,
        requestContext: req.requestContext,
        runId,
        streamId: runId,
      }))
    );
    requestMetrics.runsCompleted += 1;
    auditAction(req.requestContext, dryRun ? "job.dry_run" : "job.run", id, out.run.status);
    closeOperationStream(runId, { state: out.run.status, runId });
    sendJson(res, 200, out);
  } catch (err) {
    if (err?.statusCode === 429) {
      return sendError(res, 429, err.message, {
        code: err.code,
        details: err.details,
        retryAfter: err.retryAfter,
      });
    }
    auditAction(req.requestContext, dryRun ? "job.dry_run" : "job.run", id, "failed", {
      code: err?.code || ERROR_CODES.INTERNAL,
    });
    closeOperationStream(runId, {
      state: "failed",
      runId,
      code: err?.code || ERROR_CODES.INTERNAL,
      message: err?.message || "run failed",
    });
    throw err;
  }
}

async function handleDescribeSteps(req, res) {
  const body = await readBody(req);
  const job = body.job || body;
  const resolved = resolveJobConnectors(job);
  const step = {
    runId: "",
    stepIdx: 0,
    template: resolved.sinkDriver || "(sink)",
    args: { source: resolved.sourceDriver || "(source)" },
    status: "pending",
    rowCount: null,
    log: null,
  };
  sendJson(res, 200, [step]);
}

async function handleJobDryRun(req, res) {
  const body = await readValidatedBody(req, "jobDryRun");
  const job = body.job || {
    id: genId("job"),
    name: "dry-run",
    definition: {
      source: body.source || "",
      steps: Array.isArray(body.steps) ? body.steps : [],
    },
    enabled: true,
  };
  const runId = genId("run");
  requestMetrics.runsStarted += 1;
  try {
    const out = await withRateLimit("run", RUN_RATE_LIMIT_COUNT, RUN_MAX_INFLIGHT, () =>
      withEngineLock(() => executeJob(job, {
        autoInstall: body.autoInstall === true,
        dryRun: true,
        fullRefresh: body.fullRefresh === true,
        requestContext: req.requestContext,
        runId,
        streamId: runId,
      }))
    );
    requestMetrics.runsCompleted += 1;
    auditAction(req.requestContext, "job.dry_run", job.id, out.run.status);
    closeOperationStream(runId, { state: out.run.status, runId });
    sendJson(res, 200, out);
  } catch (err) {
    if (err?.statusCode === 429) {
      return sendError(res, 429, err.message, {
        code: err.code,
        details: err.details,
        retryAfter: err.retryAfter,
      });
    }
    throw err;
  }
}

// ---- Runs ----

async function engineRuns(requestContext) {
  const jobsResult = await runCli(["job", "list"], { requestContext, timeoutMs: 30000 });
  throwCliResultError(jobsResult, "job list failed", { classifyNonZero: true });
  const jobs = parseJobList(jobsResult.stdout);
  const stateMap = await loadEngineStateMap(requestContext);
  return jobs.map((job) => {
    const state = stateMap.get(job.name) ?? null;
    return {
      id: job.id,
      jobId: null,
      status: job.status,
      trigger: "engine",
      startedAt: job.startedAt || nowIso(),
      finishedAt: null,
      metadata: {
        diagnostics: {
          duration: job.duration,
          engineStateId: job.name,
          state,
        },
      },
    };
  });
}

async function handleRunsList(_req, res) {
  const combined = [...loadRuns(), ...await engineRuns(_req.requestContext)];
  const deduped = new Map();
  for (const run of combined) {
    if (!deduped.has(run.id)) deduped.set(run.id, run);
  }
  let runs = [...deduped.values()].sort((a, b) =>
    String(b.startedAt || "").localeCompare(String(a.startedAt || ""))
  );
  const url = new URL(_req.url, `http://localhost:${PORT}`);
  const limit = Number(url.searchParams.get("limit")) > 0 ? Number(url.searchParams.get("limit")) : 50;
  const offset = Number(url.searchParams.get("offset")) > 0 ? Number(url.searchParams.get("offset")) : 0;
  const jobId = url.searchParams.get("jobId");
  const status = url.searchParams.get("status");
  const since = url.searchParams.get("since");
  if (jobId) runs = runs.filter((run) => run.jobId === jobId);
  if (status) runs = runs.filter((run) => run.status === status);
  if (since) runs = runs.filter((run) => String(run.startedAt || "") >= since);
  const total = runs.length;
  sendJson(res, 200, {
    runs: runs.slice(offset, offset + limit),
    total,
    limit,
    offset,
  });
}

async function handleRunSteps(id, _req, res) {
  sendJson(res, 200, loadRunSteps(id));
}

async function handleRunStream(id, _req, res) {
  attachOperationStream(id, res);
}

// ---- Schedules ----

async function handleSchedulesList(_req, res) {
  const result = await runCli(["schedule", "list"], {
    requestContext: _req.requestContext,
    timeoutMs: 30000,
  });
  throwCliResultError(result, "schedule list failed", { classifyNonZero: true });
  if (result.code !== 0) {
    throw httpError(502, tail(result.stderr || result.stdout) || "schedule list failed", {
      code: ERROR_CODES.ENGINE_FAILED,
    });
  }
  const engine = parseScheduleList(result.stdout);
  const jobs = loadJobs();
  const byName = new Map(engine.map((schedule) => [schedule.name, schedule]));
  const schedules = jobs
    .filter((job) => job.definition?.schedule)
    .map((job) => {
      const name = jobScheduleName(job.id);
      const engineSchedule = byName.get(name);
      return {
        name,
        job: job.name,
        jobId: job.id,
        interval: job.definition.schedule,
        enabled: true,
        drift: engineSchedule && engineSchedule.interval === job.definition.schedule ? null : "engine_mismatch",
        raw: engineSchedule?.raw || null,
      };
    });
  for (const schedule of engine) {
    if (!schedules.find((item) => item.name === schedule.name)) {
      schedules.push({ ...schedule, drift: "ui_missing" });
    }
  }
  sendJson(res, 200, schedules);
}

function jobScheduleName(jobId) {
  return `ui-${jobId}`;
}

async function handleJobScheduleAdd(id, req, res) {
  const body = await readBody(req);
  const cron = body.cron || body.interval;
  if (!cron) return sendError(res, 400, "missing cron/interval", { code: ERROR_CODES.VALIDATION });
  if (!isValidCronExpression(cron)) {
    return sendError(res, 400, "invalid cron expression", { code: ERROR_CODES.VALIDATION, field: "cron" });
  }
  const job = loadJobs().find((j) => j.id === id);
  if (!job) return sendError(res, 404, `job ${id} not found`, { code: ERROR_CODES.NOT_FOUND });
  assertSafeIdentifier("job", job.name);
  // schedule add requires a *manifest* job. Map the UI job to the source→sink
  // run by name; if no manifest job exists, persist the schedule in the UI job.
  const name = jobScheduleName(id);
  const result = await withEngineLock(() => runCli(
    ["schedule", "add", name, "--job", job.name, "--interval", cron],
    { requestContext: req.requestContext, timeoutMs: 30000 },
  ));
  throwCliResultError(result, "schedule add failed", { classifyNonZero: true });
  if (result.code !== 0) {
    throw classifyEngineFailure(result, "schedule add failed");
  }
  // Persist on the job definition regardless (UI source of truth).
  job.definition = { ...(job.definition || {}), schedule: cron };
  const list = loadJobs();
  const idx = list.findIndex((j) => j.id === id);
  if (idx >= 0) list[idx] = job;
  writeStoreFile(STORE.jobs(), list);
  auditAction(req.requestContext, "schedule.add", id, "ok", { cron });
  sendJson(res, 200, { ok: true });
}

async function handleJobScheduleRemove(id, _req, res) {
  const name = jobScheduleName(id);
  const result = await withEngineLock(() => runCli(["schedule", "remove", name], {
    requestContext: _req.requestContext,
    timeoutMs: 30000,
  }));
  throwCliResultError(result, "schedule remove failed", { classifyNonZero: true });
  if (result.code !== 0) {
    throw httpError(502, tail(result.stderr || result.stdout) || "schedule remove failed", {
      code: ERROR_CODES.ENGINE_FAILED,
    });
  }
  const list = loadJobs();
  const idx = list.findIndex((j) => j.id === id);
  if (idx >= 0 && list[idx].definition) {
    delete list[idx].definition.schedule;
    writeStoreFile(STORE.jobs(), list);
  }
  auditAction(_req.requestContext, "schedule.remove", id, "ok");
  sendJson(res, 200, { ok: true });
}

async function handleEnvironmentsList(req, res) {
  await ensureProject(req.requestContext);
  const result = await runCliRaw(["environment", "list"], {
    cwd: PROJECT_DIR,
    requestContext: req.requestContext,
    timeoutMs: 30000,
    useActiveEnvironment: false,
  });
  throwCliResultError(result, "environment list failed", { classifyNonZero: true });
  const active = activeEnvironmentName();
  const lines = String(result.stdout || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/No environments defined\./i.test(line));
  const environments = lines.map((line) => {
    const match = line.match(/^(.+?)(?:\s+\(default\))?$/);
    const name = match ? match[1] : line;
    return {
      name,
      default: /\(default\)$/.test(line),
      active: active === name,
    };
  });
  sendJson(res, 200, { active, environments });
}

async function handleActiveEnvironmentPut(req, res) {
  const body = await readValidatedBody(req, "activeEnvironmentPut");
  await ensureProject(req.requestContext);
  const prefs = loadPrefs();
  const name = typeof body.name === "string" && body.name.trim() ? body.name.trim() : null;
  if (name) {
    const result = await runCliRaw(["environment", "list"], {
      cwd: PROJECT_DIR,
      requestContext: req.requestContext,
      timeoutMs: 30000,
      useActiveEnvironment: false,
    });
    throwCliResultError(result, "environment list failed", { classifyNonZero: true });
    const available = String(result.stdout || "")
      .split("\n")
      .map((line) => line.trim().replace(/\s+\(default\)$/, ""))
      .filter(Boolean)
      .filter((line) => !/No environments defined\./i.test(line));
    if (!available.includes(name)) {
      return sendError(res, 404, `environment ${name} not found`, { code: ERROR_CODES.NOT_FOUND });
    }
    prefs[ACTIVE_ENVIRONMENT_PREF] = name;
  } else {
    delete prefs[ACTIVE_ENVIRONMENT_PREF];
  }
  savePrefs(prefs);
  sendJson(res, 200, { ok: true, value: name });
}

// ---- Prefs ----

function loadPrefs() {
  return readStoreFile(STORE.prefs(), {}, { log: storeLog });
}

function savePrefs(prefs) {
  writeStoreFile(STORE.prefs(), prefs);
}

async function handlePrefGet(key, _req, res) {
  const prefs = loadPrefs();
  sendJson(res, 200, { value: key in prefs ? prefs[key] : null });
}

async function handlePrefPut(key, req, res) {
  const body = await readValidatedBody(req, "prefPut");
  const prefs = loadPrefs();
  prefs[key] = body.value;
  writeStoreFile(STORE.prefs(), prefs);
  sendJson(res, 200, { ok: true });
}

// ---- Secrets keeper ----

async function handleSecretsKeeperTest(req, res) {
  const body = await readBody(req);
  const url = body.url;
  if (!url) return sendJson(res, 200, { status: "fail", message: "missing url" });
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 5000);
    const headers = body.token ? { Authorization: `Bearer ${body.token}` } : {};
    const resp = await fetch(url, { headers, signal: controller.signal }).catch((e) => {
      throw e;
    });
    clearTimeout(t);
    sendJson(res, 200, {
      status: resp.ok ? "pass" : "fail",
      message: `HTTP ${resp.status} ${resp.statusText}`,
    });
  } catch (err) {
    sendJson(res, 200, { status: "fail", message: String(err && err.message) });
  }
}

// ---- FS browser ----

async function handleFsList(query, _req, res) {
  let dir = query.get("path");
  if (!dir || dir === "~") dir = os.homedir();
  try {
    const safePath = normalizePathInsideRoots(dir);
    const stat = fs.statSync(safePath);
    const base = stat.isDirectory() ? safePath : path.dirname(safePath);
    const entries = fs.readdirSync(base, { withFileTypes: true })
      .filter((e) => !e.name.startsWith("."))
      .map((e) => ({
        name: e.name,
        path: path.join(base, e.name),
        dir: e.isDirectory(),
      }))
      .sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1));
    sendJson(res, 200, { entries, base });
  } catch (err) {
    sendError(res, 400, String(err && err.message), { code: ERROR_CODES.VALIDATION });
  }
}

// ---------------------------------------------------------------------------
// Registry / Store (ADR-0016) — wired from ./registry.mjs
// ---------------------------------------------------------------------------

initRegistry({
  attachOperationStream,
  auditAction,
  closeOperationStream,
  ensureOperationStream,
  publishOperationStream,
  requestMetrics,
  runCli,
  runCliRaw,
  loadPrefs,
  savePrefs,
  tail,
  withInstallRateLimit: (task) => withRateLimit("install", INSTALL_RATE_LIMIT_COUNT, INSTALL_MAX_INFLIGHT, task),
  withEngineLock,
  celeritasRoot: CELERITAS_ROOT,
  projectParent: PROJECT_PARENT,
  log,
});

const REGISTRY_HELPERS = { sendJson, sendError, sendCacheableJson };

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const ROUTES = [
  { m: "GET", re: /^\/health$/, h: (req, res) => handleHealth(req, res) },
  { m: "GET", re: /^\/api\/livez$/, h: (req, res) => handleLivez(req, res) },
  { m: "GET", re: /^\/api\/readyz$/, h: (req, res) => handleReadyz(req, res) },
  { m: "GET", re: /^\/api\/health$/, h: (req, res) => handleHealth(req, res) },
  { m: "GET", re: /^\/metrics$/, h: (_req, res) => {
    res.statusCode = 200;
    setSecurityHeaders(res);
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
    res.end(renderPrometheusMetrics());
  } },
  { m: "GET", re: /^\/api\/engine$/, h: (req, res) => handleEngine(req, res) },
  { m: "GET", re: /^\/api\/connectors$/, h: (req, res) => handleConnectors(req, res) },
  { m: "POST", re: /^\/api\/connectors\/test$/, h: (req, res) => handleConnectorTest(req, res) },
  { m: "POST", re: /^\/api\/connectors\/inspect$/, h: (req, res) => handleConnectorInspect(req, res) },
  { m: "POST", re: /^\/api\/connectors\/action$/, h: (req, res) => handleConnectorAction(req, res) },

  { m: "GET", re: /^\/api\/connector-instances$/, h: (req, res) => handleInstancesList(req, res) },
  { m: "POST", re: /^\/api\/connector-instances$/, h: (req, res) => handleInstancesSave(req, res) },
  { m: "DELETE", re: /^\/api\/connector-instances\/([^/]+)$/, h: (req, res, mt) => handleInstancesDelete(decodeURIComponent(mt[1]), req, res) },

  { m: "GET", re: /^\/api\/targets$/, h: (req, res) => handleTargetsList(req, res) },
  { m: "POST", re: /^\/api\/targets$/, h: (req, res) => handleTargetsSave(req, res) },
  { m: "DELETE", re: /^\/api\/targets\/([^/]+)$/, h: (req, res, mt) => handleTargetsDelete(decodeURIComponent(mt[1]), req, res) },

  { m: "GET", re: /^\/api\/jobs$/, h: (req, res) => handleJobsList(req, res) },
  { m: "POST", re: /^\/api\/jobs$/, h: (req, res) => handleJobsSave(req, res) },
  { m: "POST", re: /^\/api\/jobs\/dry-run$/, h: (req, res) => handleJobDryRun(req, res) },
  // job sub-routes must precede the bare /jobs/:id delete.
  { m: "POST", re: /^\/api\/jobs\/([^/]+)\/run$/, h: (req, res, mt) => handleJobRun(decodeURIComponent(mt[1]), req, res, { dryRun: false }) },
  { m: "POST", re: /^\/api\/jobs\/([^/]+)\/dry-run$/, h: (req, res, mt) => handleJobRun(decodeURIComponent(mt[1]), req, res, { dryRun: true }) },
  { m: "POST", re: /^\/api\/jobs\/describe-steps$/, h: (req, res) => handleDescribeSteps(req, res) },
  { m: "POST", re: /^\/api\/jobs\/([^/]+)\/schedule$/, h: (req, res, mt) => handleJobScheduleAdd(decodeURIComponent(mt[1]), req, res) },
  { m: "DELETE", re: /^\/api\/jobs\/([^/]+)\/schedule$/, h: (req, res, mt) => handleJobScheduleRemove(decodeURIComponent(mt[1]), req, res) },
  { m: "DELETE", re: /^\/api\/jobs\/([^/]+)$/, h: (req, res, mt) => handleJobsDelete(decodeURIComponent(mt[1]), req, res) },

  { m: "GET", re: /^\/api\/runs$/, h: (req, res) => handleRunsList(req, res) },
  { m: "GET", re: /^\/api\/runs\/([^/]+)\/steps$/, h: (req, res, mt) => handleRunSteps(decodeURIComponent(mt[1]), req, res) },
  { m: "GET", re: /^\/api\/runs\/([^/]+)\/stream$/, h: (req, res, mt) => handleRunStream(decodeURIComponent(mt[1]), req, res) },

  { m: "GET", re: /^\/api\/schedules$/, h: (req, res) => handleSchedulesList(req, res) },

  { m: "GET", re: /^\/api\/prefs\/([^/]+)$/, h: (req, res, mt) => handlePrefGet(decodeURIComponent(mt[1]), req, res) },
  { m: "PUT", re: /^\/api\/prefs\/([^/]+)$/, h: (req, res, mt) => handlePrefPut(decodeURIComponent(mt[1]), req, res) },
  { m: "GET", re: /^\/api\/environments$/, h: (req, res) => handleEnvironmentsList(req, res) },
  { m: "PUT", re: /^\/api\/environments\/active$/, h: (req, res) => handleActiveEnvironmentPut(req, res) },

  { m: "POST", re: /^\/api\/secrets-keeper\/test$/, h: (req, res) => handleSecretsKeeperTest(req, res) },
];

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;
  attachRequestLifecycle(req, res, pathname);
  try {
    const corsAllowed = applyCors(req, res);
    if (!corsAllowed && (req.method === "OPTIONS" || isMutatingMethod(req.method))) {
      return sendError(res, 403, "origin not allowed", { code: ERROR_CODES.ORIGIN_FORBIDDEN });
    }
    if (shutdownState) {
      res.setHeader("Connection", "close");
      return sendError(res, 503, "server is shutting down", { code: ERROR_CODES.SHUTTING_DOWN });
    }
    if (req.method === "OPTIONS") {
      res.statusCode = 204;
      res.end();
      return;
    }
    if (isMutatingMethod(req.method) && !isWriteOriginAllowed(req)) {
      return sendError(res, 403, "mutating requests require an allowed Origin or Referer", {
        code: ERROR_CODES.ORIGIN_FORBIDDEN,
      });
    }
    if (pathname.startsWith("/api/") && !isAuthorized(req, pathname)) {
      res.setHeader("WWW-Authenticate", "Bearer");
      return sendError(res, 401, "authorization required", { code: ERROR_CODES.AUTH_REQUIRED });
    }

    // /fs/list with query string.
    if (req.method === "GET" && pathname === "/api/fs/list") {
      return await handleFsList(url.searchParams, req, res);
    }

    // Registry / Store (ADR-0016) — handlers need query params and/or the body.
    if (req.method === "GET" && pathname === "/api/registry") {
      return await handleRegistryList(url.searchParams, req, res, REGISTRY_HELPERS);
    }
    if (req.method === "GET" && pathname === "/api/registry/package") {
      return await handleRegistryPackage(url.searchParams, req, res, REGISTRY_HELPERS);
    }
    if (req.method === "GET" && pathname.match(/^\/api\/registry\/install\/[^/]+\/stream$/)) {
      const match = pathname.match(/^\/api\/registry\/install\/([^/]+)\/stream$/);
      return await handleRegistryInstallStream(decodeURIComponent(match[1]), req, res);
    }
    if (req.method === "GET" && pathname === "/api/registry/urls") {
      return await handleRegistryUrlsGet(url.searchParams, req, res, REGISTRY_HELPERS);
    }
    if (req.method === "POST" && pathname === "/api/registry/install") {
      const body = await readValidatedBody(req, "registryInstall");
      return await handleRegistryInstall(body, req, res, REGISTRY_HELPERS);
    }
    if (req.method === "PUT" && pathname === "/api/registry/urls") {
      const body = await readValidatedBody(req, "registryUrlsPut");
      return await handleRegistryUrlsPut(body, req, res, REGISTRY_HELPERS);
    }

    for (const route of ROUTES) {
      if (route.m !== req.method) continue;
      const mt = pathname.match(route.re);
      if (mt) {
        return await route.h(req, res, mt);
      }
    }
    sendError(res, 404, `no route for ${req.method} ${pathname}`, { code: ERROR_CODES.NOT_FOUND });
  } catch (err) {
    requestLog(req.requestContext, "error", "handler error", { error: serializeError(err) });
    await reportServerError("handler", err, req.requestContext);
    if (!res.headersSent) {
      sendError(
        res,
        err?.statusCode && Number.isInteger(err.statusCode) ? err.statusCode : 500,
        err && err.message ? err.message : String(err),
        {
          ...(err?.code ? { code: err.code } : {}),
          ...(err?.details !== undefined ? { details: err.details } : {}),
          ...(err?.field ? { field: err.field } : {}),
          ...(err?.retryAfter ? { retryAfter: err.retryAfter } : {}),
        },
      );
    }
    else {
      try {
        res.end();
      } catch {
        /* noop */
      }
    }
  }
});

server.on("clientError", (_err, socket) => {
  try {
    socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
  } catch {
    /* noop */
  }
});

process.on("SIGINT", () => beginShutdown("SIGINT"));
process.on("SIGTERM", () => beginShutdown("SIGTERM"));
process.on("uncaughtException", (err) => {
  writeLog("error", "uncaughtException", { error: serializeError(err) });
  void reportServerError("uncaughtException", err);
});
process.on("unhandledRejection", (err) => {
  writeLog("error", "unhandledRejection", { error: serializeError(err) });
  void reportServerError("unhandledRejection", err);
});

if (!isLoopbackHost(HOST) && !API_TOKEN) {
  // eslint-disable-next-line no-console
  console.error(`Refusing to bind to ${HOST} without CELERITAS_API_TOKEN.`);
  process.exit(1);
}
if (!isLoopbackHost(HOST)) {
  writeLog("warn", "binding sidecar to a non-loopback interface", { host: HOST, token_auth: true });
}

server.on("error", (err) => {
  if (err?.code === "EADDRINUSE") {
    // eslint-disable-next-line no-console
    console.error(`Port ${PORT} is already in use on ${HOST}. Stop the other Celeritas server or set PORT to a free port.`);
    process.exit(1);
  }
  writeLog("error", "server error", { error: serializeError(err) });
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  writeLog("info", "server listening", {
    build_sha: BUILD_SHA,
    build_version: BUILD_VERSION,
    host: HOST,
    port: PORT,
    project_dir: PROJECT_DIR,
  });
  writeLog("debug", "server config", {
    celeritas_bin: CELERITAS_BIN,
    cors_origins: CORS_ORIGINS,
    host: HOST,
    project_dir: PROJECT_DIR,
    token_auth: Boolean(API_TOKEN),
  });
  if (!fs.existsSync(CELERITAS_BIN)) {
    writeLog("warn", "celeritas binary not found", { celeritas_bin: CELERITAS_BIN });
  }
});
