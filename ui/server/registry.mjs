// Celeritas UI ⇄ Engine sidecar — Registry / Store (ADR-0016).
//
// Looks up a registry manifest from online (the ADR-0013 `registry.manifest.json`
// publication format), normalizes packages into a searchable list of additional
// sources & targets, surfaces full package detail, and installs into the local
// engine. See ui/docs/WIRING.md §"Registry / Store (ADR-0016)".
//
// This module is imported by celeritas-server.mjs. It reuses that server's
// `runCli` (project-aware CLI runner), prefs store, JSON helpers, and `tail`,
// which are injected via init() to keep a single source of truth.

import { createHash } from "node:crypto";

// ---------------------------------------------------------------------------
// Injected deps (from celeritas-server.mjs)
// ---------------------------------------------------------------------------

let DEPS = null;

/**
 * Wire in the host server's helpers.
 * @param {{
 *   runCli: (args:string[], opts?:object)=>Promise<{code:number,stdout:string,stderr:string}>,
 *   runCliRaw: (args:string[], opts?:object)=>Promise<{code:number,stdout:string,stderr:string}>,
 *   loadPrefs: ()=>Record<string,unknown>,
 *   savePrefs: (prefs:Record<string,unknown>)=>void,
 *   tail: (text:string, max?:number)=>string,
 *   attachOperationStream: (id:string, res: import('node:http').ServerResponse)=>void,
 *   closeOperationStream: (id:string, finalData?:object)=>void,
 *   ensureOperationStream: (id:string)=>unknown,
 *   publishOperationStream: (id:string, event:string, data:object)=>void,
 *   withEngineLock: <T>(task: ()=>Promise<T>)=>Promise<T>,
 *   celeritasRoot: string,
 *   projectParent: string,
 *   log: (...args:unknown[])=>void,
 * }} deps
 */
export function initRegistry(deps) {
  DEPS = deps;
}

// ---------------------------------------------------------------------------
// Default registry URL resolution
// ---------------------------------------------------------------------------

const FALLBACK_URLS = [
  "http://127.0.0.1:4321/registry.manifest.json",
  "https://celeritas.dev/registry.manifest.json",
];
export const TRUSTED_PUBLISHERS = new Set(["cargo", "local", "npm", "pip"]);

const REGISTRY_URLS_PREF = "registry-urls";
const REGISTRY_PINS_PREF = "registry-pins";

/** Configured registry URLs from the prefs store (array), if any. */
function configuredUrls() {
  if (!DEPS?.loadPrefs) return [];
  const prefs = DEPS.loadPrefs();
  const v = prefs[REGISTRY_URLS_PREF];
  if (Array.isArray(v))
    return v.filter((u) => typeof u === "string" && u.trim());
  return [];
}

/**
 * Ordered list of candidate manifest URLs to try.
 * Explicit `url` wins; otherwise: CELERITAS_REGISTRY_URL env → registry-urls
 * pref array → local website fallback → prod fallback.
 */
function resolveUrls(explicit) {
  if (explicit && explicit.trim()) return [explicit.trim()];
  const out = [];
  if (
    process.env.CELERITAS_REGISTRY_URL &&
    process.env.CELERITAS_REGISTRY_URL.trim()
  ) {
    out.push(process.env.CELERITAS_REGISTRY_URL.trim());
  }
  for (const u of configuredUrls()) out.push(u);
  for (const u of FALLBACK_URLS) out.push(u);
  // de-dupe, preserve order
  return [...new Set(out)];
}

export function trustedRegistryUrls() {
  return resolveUrls();
}

export function isTrustedRegistryUrl(url) {
  return trustedRegistryUrls().includes(url);
}

function configuredPins() {
  const pins = {};
  const envPins = process.env.CELERITAS_REGISTRY_PINS?.trim();
  if (envPins) {
    try {
      Object.assign(pins, JSON.parse(envPins));
    } catch {
      /* ignore invalid env pins */
    }
  }
  if (DEPS?.loadPrefs) {
    const prefs = DEPS.loadPrefs();
    if (prefs[REGISTRY_PINS_PREF] && typeof prefs[REGISTRY_PINS_PREF] === "object") {
      Object.assign(pins, prefs[REGISTRY_PINS_PREF]);
    }
  }
  return pins;
}

function manifestPinForUrl(url) {
  const pins = configuredPins();
  const pin = pins[url];
  return typeof pin === "string" && pin.trim() ? pin.trim() : null;
}

// ---------------------------------------------------------------------------
// Manifest fetch + per-URL cache (~60s)
// ---------------------------------------------------------------------------

const MANIFEST_CACHE = new Map(); // url -> { at:number, manifest:object, checksum:string, etag:string|null }
const CACHE_TTL_MS =
  Number(process.env.REGISTRY_CACHE_TTL_MS) > 0
    ? Number(process.env.REGISTRY_CACHE_TTL_MS)
    : 60 * 1000;
const REGISTRY_FETCH_TIMEOUT_MS =
  Number(process.env.REGISTRY_FETCH_TIMEOUT_MS) > 0
    ? Number(process.env.REGISTRY_FETCH_TIMEOUT_MS)
    : 10_000;
const REGISTRY_FETCH_RETRY_DELAY_MS = 250;

export function validateManifestShape(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("registry schema error: manifest must be an object");
  }
  if (!("schema_version" in manifest) && !("schemaVersion" in manifest)) {
    throw new Error("registry schema error: missing schema_version");
  }
  if (
    !manifest.registry ||
    typeof manifest.registry !== "object" ||
    Array.isArray(manifest.registry)
  ) {
    throw new Error("registry schema error: missing registry object");
  }
  if (!Array.isArray(manifest.packages)) {
    throw new Error("registry schema error: packages must be an array");
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchManifestFromUrl(url, { timeoutMs = 10000 } = {}) {
  const cached = MANIFEST_CACHE.get(url);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return {
      url,
      manifest: cached.manifest,
      checksum: cached.checksum,
      etag: cached.etag,
      cached: true,
    };
  }
  try {
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const resp = await fetch(url, {
          signal: controller.signal,
          headers: { Accept: "application/json" },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
        const manifest = await resp.json();
        validateManifestShape(manifest);
        const payload = JSON.stringify(manifest);
        const checksum = `sha256:${createHash("sha256").update(payload).digest("hex")}`;
        const etag = resp.headers.get("etag");
        MANIFEST_CACHE.set(url, { at: Date.now(), manifest, checksum, etag });
        return { url, manifest, checksum, etag, cached: false, stale: false };
      } catch (err) {
        lastError = err;
        if (attempt === 0) {
          await sleep(REGISTRY_FETCH_RETRY_DELAY_MS);
        }
      } finally {
        clearTimeout(timer);
      }
    }
    if (cached) {
      return {
        url,
        manifest: cached.manifest,
        checksum: cached.checksum,
        etag: cached.etag,
        cached: true,
        stale: true,
      };
    }
    throw lastError;
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error(`registry fetch timeout after ${timeoutMs}ms`, {
        cause: err,
      });
    }
    throw err;
  }
}

/** Fetch from the first URL that resolves; returns { url, manifest }. */
async function fetchManifest(explicit) {
  const urls = resolveUrls(explicit);
  const errors = [];
  for (const url of urls) {
    try {
      return await fetchManifestFromUrl(url, {
        timeoutMs: REGISTRY_FETCH_TIMEOUT_MS,
      });
    } catch (err) {
      errors.push(`${url}: ${err && err.message ? err.message : String(err)}`);
    }
  }
  throw new Error(
    `failed to fetch registry manifest (tried ${urls.length}): ${errors.join("; ")}`,
  );
}

function verifyManifestIntegrity(url, fetched) {
  const expected = manifestPinForUrl(url);
  if (!expected) {
    return { ok: false, message: `registry install refused: ${url} has no configured integrity pin` };
  }
  if (expected === fetched.checksum) {
    return { ok: true };
  }
  if (fetched.etag && expected === fetched.etag) {
    return { ok: true };
  }
  return {
    ok: false,
    message: `registry install refused: manifest integrity mismatch for ${url}`,
  };
}

// ---------------------------------------------------------------------------
// Installed-set (from `celeritas hub list`)
// ---------------------------------------------------------------------------

const INSTALLED_CACHE = { at: 0, set: null };

/** Set of locally-available connector names from `celeritas hub list`. */
async function installedNames() {
  if (INSTALLED_CACHE.set && Date.now() - INSTALLED_CACHE.at < CACHE_TTL_MS) {
    return INSTALLED_CACHE.set;
  }
  const set = new Set();
  try {
    const res = await DEPS.runCliRaw(["hub", "list"], {
      cwd: DEPS.projectParent,
      timeoutMs: 60000,
    });
    if (res.code === 0) {
      for (const raw of res.stdout.split("\n")) {
        const line = raw.trim();
        if (!line) continue;
        const m = line.match(/^(\S+)\s+/);
        if (m) set.add(m[1]);
      }
    }
  } catch (err) {
    DEPS.log("registry: hub list failed", err && err.message);
  }
  INSTALLED_CACHE.at = Date.now();
  INSTALLED_CACHE.set = set;
  return set;
}

function invalidateInstalledCache() {
  INSTALLED_CACHE.at = 0;
  INSTALLED_CACHE.set = null;
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/** manifest `type` (source|sink|loader|target) → coarse UI kind (source|target). */
export function coarseType(type) {
  const t = String(type || "").toLowerCase();
  if (t === "sink" || t === "loader" || t === "target") return "target";
  return "source";
}

/** `celeritas add <type> <name>` plugin-type word for the coarse kind. */
function addTypeWord(type) {
  return coarseType(type) === "target" ? "loader" : "source";
}

/**
 * Normalize a raw manifest package into a RegistryPackage.
 * Adds: installed, installCommand, packagedBy, packagedAt (+ keeps every
 * original manifest field including manifest.settings / capabilities / tags).
 */
export function normalizePackage(pkg, { schemaVersion, installed }) {
  const installCommand =
    (pkg.artifact && pkg.artifact.install_command) ||
    `celeritas add ${addTypeWord(pkg.type)} ${pkg.name}`;
  const packagedBy = pkg.author || { name: null, email: null, url: null };
  const packagedAt = pkg.publishedAt || schemaVersion || null;
  return {
    ...pkg,
    rawType: pkg.type,
    type: coarseType(pkg.type),
    installed: Boolean(installed),
    installCommand,
    packagedBy,
    packagedAt,
  };
}

/** Build the full normalized package list for a manifest. */
export async function normalizeManifest(manifest) {
  const schemaVersion =
    manifest.schema_version || manifest.schemaVersion || null;
  const installed = await installedNames();
  const packages = (manifest.packages || []).map((pkg) =>
    normalizePackage(pkg, {
      schemaVersion,
      installed: installed.has(pkg.name),
    }),
  );
  return { registry: manifest.registry || {}, packages };
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

export function applyFilters(packages, { type, q, category }) {
  let out = packages;
  if (type) {
    const want = coarseType(type);
    out = out.filter((p) => coarseType(p.type) === want);
  }
  if (category) {
    const c = String(category).toLowerCase();
    out = out.filter((p) => String(p.category || "").toLowerCase() === c);
  }
  if (q) {
    const needle = String(q).toLowerCase();
    out = out.filter((p) => {
      const hay = [
        p.name,
        p.title,
        p.summary,
        p.description,
        p.category,
        ...(Array.isArray(p.tags) ? p.tags : []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(needle);
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// HTTP handlers (mirror the host server's handler signature/style)
// ---------------------------------------------------------------------------

/** GET /api/registry?url=&type=&q=&category= */
export async function handleRegistryList(query, _req, res, helpers) {
  const { sendJson, sendError, sendCacheableJson } = helpers;
  try {
    const { manifest } = await fetchManifest(query.get("url"));
    const { registry, packages } = await normalizeManifest(manifest);
    const filtered = applyFilters(packages, {
      type: query.get("type"),
      q: query.get("q"),
      category: query.get("category"),
    });
    const payload = { registry, packages: filtered };
    if (typeof sendCacheableJson === "function") {
      sendCacheableJson(_req, res, payload);
      return;
    }
    sendJson(res, 200, payload);
  } catch (err) {
    sendError(res, 502, err && err.message ? err.message : String(err), {
      code: /schema error/i.test(err?.message || "")
        ? ERROR_CODES.REGISTRY_SCHEMA
        : ERROR_CODES.REGISTRY_FETCH_FAILED,
    });
  }
}

/** GET /api/registry/package?name=&url= */
export async function handleRegistryPackage(query, _req, res, helpers) {
  const { sendJson, sendError } = helpers;
  const name = query.get("name");
  if (!name)
    return sendError(res, 400, "missing name", {
      code: ERROR_CODES.VALIDATION,
    });
  try {
    const { manifest } = await fetchManifest(query.get("url"));
    const { packages } = await normalizeManifest(manifest);
    const pkg = packages.find((p) => p.name === name);
    if (!pkg)
      return sendError(res, 404, `package ${name} not found in registry`, {
        code: ERROR_CODES.REGISTRY_PACKAGE_NOT_FOUND,
      });
    sendJson(res, 200, pkg);
  } catch (err) {
    sendError(res, 502, err && err.message ? err.message : String(err), {
      code: /schema error/i.test(err?.message || "")
        ? ERROR_CODES.REGISTRY_SCHEMA
        : ERROR_CODES.REGISTRY_FETCH_FAILED,
    });
  }
}

/**
 * POST /api/registry/install body { name, url? }
 * → { status:"installed"|"failed", message, installCommand, log }
 *
 * - publisher "pip": `celeritas hub add <name> --type <ext|loader> --pip <pkg>`,
 *   plus capabilities/setting/description so it is resolvable afterwards.
 * - otherwise: execute `artifact.install_command` from the celeritas repo root
 *   (so `cargo install --path extensions/...` relative paths resolve).
 * Honest: exit 0 ⇒ installed; else failed with stderr tail.
 */
export async function handleRegistryInstall(body, _req, res, helpers) {
  const { sendJson, sendError } = helpers;
  const name = body && body.name;
  if (!name)
    return sendError(res, 400, "missing name", {
      code: ERROR_CODES.VALIDATION,
    });
  const installId =
    typeof body?.installId === "string" && body.installId.trim()
      ? body.installId.trim()
      : null;
  const closeInstallStream = (finalData) => {
    if (installId) {
      DEPS.closeOperationStream(installId, finalData);
    }
  };
  if (installId) {
    DEPS.ensureOperationStream(installId);
    DEPS.publishOperationStream(installId, "status", {
      state: "started",
      installId,
    });
  }

  let pkg;
  let manifestUrl;
  try {
    const fetched = await fetchManifest(body.url);
    manifestUrl = fetched.url;
    const { manifest } = fetched;
    const { packages } = await normalizeManifest(manifest);
    pkg = packages.find((p) => p.name === name);
    body._manifestFetch = fetched;
  } catch (err) {
    closeInstallStream({
      state: "failed",
      installId,
      code: /schema error/i.test(err?.message || "")
        ? ERROR_CODES.REGISTRY_SCHEMA
        : ERROR_CODES.REGISTRY_FETCH_FAILED,
      message: err && err.message ? err.message : String(err),
    });
    return sendError(res, 502, err && err.message ? err.message : String(err), {
      code: /schema error/i.test(err?.message || "")
        ? ERROR_CODES.REGISTRY_SCHEMA
        : ERROR_CODES.REGISTRY_FETCH_FAILED,
    });
  }
  if (!pkg) {
    closeInstallStream({
      state: "failed",
      installId,
      code: ERROR_CODES.REGISTRY_PACKAGE_NOT_FOUND,
      message: `package ${name} not found in registry`,
    });
    return sendError(res, 404, `package ${name} not found in registry`, {
      code: ERROR_CODES.REGISTRY_PACKAGE_NOT_FOUND,
    });
  }

  const installCommand = pkg.installCommand;
  if (!isTrustedRegistryUrl(manifestUrl)) {
    closeInstallStream({
      state: "failed",
      installId,
      code: ERROR_CODES.REGISTRY_UNTRUSTED,
      message: `registry install refused: ${manifestUrl} is not in the trusted registry URL list`,
    });
    return sendError(
      res,
      403,
      `registry install refused: ${manifestUrl} is not in the trusted registry URL list`,
      {
        code: ERROR_CODES.REGISTRY_UNTRUSTED,
        details: { installCommand },
      },
    );
  }
  const integrity = verifyManifestIntegrity(manifestUrl, body._manifestFetch);
  if (!integrity.ok) {
    closeInstallStream({
      state: "failed",
      installId,
      code: ERROR_CODES.REGISTRY_UNTRUSTED,
      message: integrity.message,
    });
    return sendError(res, 403, integrity.message, {
      code: ERROR_CODES.REGISTRY_UNTRUSTED,
      details: {
        checksum: body._manifestFetch?.checksum || null,
        etag: body._manifestFetch?.etag || null,
      },
    });
  }
  const TIMEOUT_MS = 900_000; // cargo --path builds from source
  const logs = [];
  let result;

  const publisher = pkg.artifact && pkg.artifact.publisher;
  if (!TRUSTED_PUBLISHERS.has(publisher)) {
    closeInstallStream({
      state: "failed",
      installId,
      code: ERROR_CODES.REGISTRY_UNTRUSTED,
      message: `registry install refused: publisher '${publisher}' is not allowlisted`,
    });
    return sendError(
      res,
      403,
      `registry install refused: publisher '${publisher}' is not allowlisted`,
      {
        code: ERROR_CODES.REGISTRY_UNTRUSTED,
        details: { installCommand },
      },
    );
  }
  const installPlan = buildInstallPlan(pkg);
  if (!installPlan) {
    closeInstallStream({
      state: "failed",
      installId,
      code: ERROR_CODES.REGISTRY_UNTRUSTED,
      message: `registry install refused: could not derive a safe install command for publisher '${publisher}'`,
    });
    return sendError(
      res,
      403,
      `registry install refused: could not derive a safe install command for publisher '${publisher}'`,
      {
        code: ERROR_CODES.REGISTRY_UNTRUSTED,
        details: { installCommand },
      },
    );
  }

  DEPS.requestMetrics.installsStarted += 1;
  result = await DEPS.withInstallRateLimit(() => DEPS.withEngineLock(async () => {
    if (installPlan.kind === "celeritas") {
      const cliResult = await DEPS.runCli(installPlan.args, {
        cwd: DEPS.projectParent,
        requestContext: _req.requestContext,
        streamId: installId,
        timeoutMs: TIMEOUT_MS,
      });
      logs.push(
        `$ ${installPlan.displayCommand}\n${DEPS.tail(cliResult.stdout)}${DEPS.tail(cliResult.stderr)}`,
      );
      return cliResult;
    }
    const commandResult = await runCommand(
      installPlan.command,
      installPlan.args,
      {
        cwd: DEPS.celeritasRoot,
        streamId: installId,
        timeoutMs: TIMEOUT_MS,
      },
    );
    logs.push(
      `$ ${installPlan.displayCommand}\n${DEPS.tail(commandResult.stdout)}${DEPS.tail(commandResult.stderr)}`,
    );
    return commandResult;
  }));

  // A successful install changes hub state.
  invalidateInstalledCache();

  const ok = result.code === 0;
  DEPS.requestMetrics.installsCompleted += 1;
  const log = logs.join("\n");
  DEPS.auditAction(_req.requestContext, "registry.install", name, ok ? "ok" : "failed", {
    manifestUrl,
    checksum: body._manifestFetch?.checksum || null,
  });
  closeInstallStream({
    state: ok ? "installed" : "failed",
    installId,
    code: ok ? null : result.codeName || ERROR_CODES.ENGINE_FAILED,
    message: ok
      ? `Installed ${name}.`
      : DEPS.tail(result.stderr) ||
        DEPS.tail(result.stdout) ||
        "install failed",
  });
  sendJson(res, 200, {
    ...(installId ? { installId } : {}),
    status: ok ? "installed" : "failed",
    message: ok
      ? `Installed ${name}.`
      : DEPS.tail(result.stderr) ||
        DEPS.tail(result.stdout) ||
        "install failed",
    installCommand,
    log,
  });
}

/** GET /api/registry/urls → string[] */
export async function handleRegistryUrlsGet(_query, _req, res, helpers) {
  helpers.sendJson(res, 200, configuredUrls());
}

/** PUT /api/registry/urls body { urls } → { ok:true } */
export async function handleRegistryUrlsPut(body, _req, res, helpers) {
  const { sendJson, sendError } = helpers;
  const urls = body && body.urls;
  if (!Array.isArray(urls))
    return sendError(res, 400, "urls must be an array", {
      code: ERROR_CODES.VALIDATION,
    });
  const clean = urls
    .filter((u) => typeof u === "string" && u.trim())
    .map((u) => u.trim());
  const prefs = DEPS.loadPrefs();
  prefs[REGISTRY_URLS_PREF] = clean;
  DEPS.savePrefs(prefs);
  MANIFEST_CACHE.clear(); // configured URLs changed
  sendJson(res, 200, { ok: true });
}

export async function handleRegistryInstallStream(id, _req, res) {
  DEPS.attachOperationStream(id, res);
}

// ---------------------------------------------------------------------------
// Local command execution (for non-pip install commands)
// ---------------------------------------------------------------------------

import { spawn } from "node:child_process";
import path from "node:path";
import { ERROR_CODES } from "./errors.mjs";

function manifestInstallPath(pkg) {
  if (typeof pkg?.manifest?.path !== "string" || !pkg.manifest.path.trim())
    return null;
  return path.dirname(pkg.manifest.path);
}

export function buildInstallPlan(pkg) {
  const publisher = pkg?.artifact?.publisher;
  const packageName =
    typeof pkg?.artifact?.package === "string"
      ? pkg.artifact.package.trim()
      : "";
  const manifestPath = manifestInstallPath(pkg);
  if (publisher === "pip") {
    const pipPkg = packageName || pkg.name;
    const typeWord = coarseType(pkg.type) === "target" ? "loader" : "extractor";
    const args = ["hub", "add", pkg.name, "--type", typeWord, "--pip", pipPkg];
    for (const cap of pkg.manifest?.capabilities || [])
      args.push("--capability", cap);
    for (const s of pkg.manifest?.settings || []) {
      if (s && s.name) args.push("--setting", s.name);
    }
    if (pkg.summary) args.push("--description", pkg.summary);
    return {
      kind: "celeritas",
      args,
      displayCommand: `celeritas ${args.join(" ")}`,
    };
  }
  if ((publisher === "cargo" || publisher === "local") && manifestPath) {
    return {
      kind: "command",
      command: "cargo",
      args: ["install", "--path", manifestPath],
      displayCommand: `cargo install --path ${manifestPath}`,
    };
  }
  if (publisher === "npm" && packageName) {
    return {
      kind: "command",
      command: "npm",
      args: ["install", "--global", packageName],
      displayCommand: `npm install --global ${packageName}`,
    };
  }
  return null;
}

/** Run an executable with args, capturing stdout/stderr. */
function createLineEmitter(streamId, event) {
  let buffer = "";
  return {
    push(chunk) {
      if (!streamId) return;
      buffer += chunk;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        DEPS.publishOperationStream(streamId, event, { line });
      }
    },
    flush() {
      if (streamId && buffer) {
        DEPS.publishOperationStream(streamId, event, { line: buffer });
      }
      buffer = "";
    },
  };
}

function runCommand(command, args, { cwd, streamId, timeoutMs = 900000 } = {}) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    let child;
    const stdoutEmitter = createLineEmitter(streamId, "stdout");
    const stderrEmitter = createLineEmitter(streamId, "stderr");
    if (streamId) {
      DEPS.publishOperationStream(streamId, "command", {
        command: `${command} ${args.join(" ")}`,
      });
    }
    try {
      child = spawn(command, args, { cwd, env: { ...process.env } });
    } catch (err) {
      resolve({ code: -1, stdout: "", stderr: String(err && err.message) });
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
      resolve({
        code: -1,
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
      resolve({ code: -1, stdout, stderr: `${stderr}${err.message}` });
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stdoutEmitter.flush();
      stderrEmitter.flush();
      resolve({ code: code == null ? -1 : code, stdout, stderr });
    });
  });
}
