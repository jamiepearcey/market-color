// Local engine sidecar (HTTP) data path. The third "server mode" between the
// Tauri desktop backend and the browser preview mock: when a `celeritas` Node
// sidecar (see `server/celeritas-server.mjs`, ADR-0015 / docs/WIRING.md) is
// reachable on SERVER_BASE, `api.ts` routes the ETL calls here so the browser
// build performs *real* discovery / tests / runs against the engine.
import { isDesktop } from "./api";
import { ServerApiError } from "./ui-errors";

const SERVER_BASE_PREF_KEY = "celeritas.ui.server_base";
const ENV_SERVER_BASE =
  (import.meta.env.VITE_CELERITAS_API as string | undefined) ??
  "http://127.0.0.1:8787";
const ENV_DEFAULT_REGISTRY_URL =
  (import.meta.env.VITE_CELERITAS_REGISTRY_URL as string | undefined)?.trim() ??
  "";
let serverBase = readServerBase();

// ---- Health probe (cached) ----------------------------------------------------

const PROBE_TTL_MS = 3000;
const PROBE_TIMEOUT_MS = 800;

let cachedReady: boolean | null = null;
let cachedAt = 0;
let inFlight: Promise<boolean> | null = null;
let cachedHealth: ServerHealth | null = null;
let inFlightHealth: Promise<ServerHealth | null> | null = null;

interface ServerErrorPayload {
  code?: string;
  message?: string;
  error?: string;
  field?: string | null;
  details?: unknown;
}

function readServerBase(): string {
  if (typeof window === "undefined") return ENV_SERVER_BASE;
  try {
    return (
      localStorage.getItem(SERVER_BASE_PREF_KEY)?.trim() || ENV_SERVER_BASE
    );
  } catch {
    return ENV_SERVER_BASE;
  }
}

function resetServerProbeCache() {
  cachedReady = null;
  cachedAt = 0;
  inFlight = null;
  cachedHealth = null;
  inFlightHealth = null;
}

export function getServerBase(): string {
  return serverBase;
}

export function defaultServerBase(): string {
  return ENV_SERVER_BASE;
}

export function defaultRegistryUrl(): string {
  return ENV_DEFAULT_REGISTRY_URL;
}

export function setServerBase(next: string) {
  serverBase = next.trim() || ENV_SERVER_BASE;
  if (typeof window !== "undefined") {
    try {
      localStorage.setItem(SERVER_BASE_PREF_KEY, serverBase);
    } catch {
      /* ignore */
    }
  }
  resetServerProbeCache();
}

export interface ServerHealth {
  ok: boolean;
  engine: string | false;
  version: string;
  compatible: boolean;
  supportedRange: string;
  project: string;
  buildVersion: string;
  buildSha: string;
  code?: string;
  message?: string;
}

/**
 * Cached probe of `GET {SERVER_BASE}/api/health`. Re-probes at most once every
 * few seconds and uses an ~800ms abort timeout so a missing server fails fast.
 * Never throws.
 */
export async function serverReady(): Promise<boolean> {
  const now = Date.now();
  if (cachedReady !== null && now - cachedAt < PROBE_TTL_MS) return cachedReady;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    let health: ServerHealth | null = null;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), PROBE_TIMEOUT_MS);
      try {
        const res = await fetch(`${serverBase}/api/health`, {
          method: "GET",
          signal: ctrl.signal,
        });
        if (res.ok) {
          health = (await res.json()) as ServerHealth;
        }
      } finally {
        clearTimeout(timer);
      }
    } catch {
      health = null;
    }
    cachedHealth = health;
    cachedReady = Boolean(health?.ok);
    cachedAt = Date.now();
    inFlight = null;
    return cachedReady;
  })();
  return inFlight;
}

export async function serverHealth(): Promise<ServerHealth | null> {
  const now = Date.now();
  if (cachedHealth && now - cachedAt < PROBE_TTL_MS) return cachedHealth;
  if (inFlightHealth) return inFlightHealth;

  inFlightHealth = (async () => {
    const ready = await serverReady();
    inFlightHealth = null;
    return ready ? cachedHealth : null;
  })();
  return inFlightHealth;
}

// ---- Fetch helpers ------------------------------------------------------------

/** Low-level JSON fetch against `{SERVER_BASE}/api{path}`. Throws on non-2xx. */
export async function srv<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${serverBase}/api${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = "";
    let parsed: ServerErrorPayload | null = null;
    try {
      detail = await res.text();
      parsed = detail ? (JSON.parse(detail) as ServerErrorPayload) : null;
    } catch {
      /* ignore */
    }
    const message =
      parsed?.message ??
      parsed?.error ??
      (detail
        ? `server ${res.status} ${res.statusText} for ${path}: ${detail}`
        : `server ${res.status} ${res.statusText} for ${path}`);
    throw new ServerApiError({
      message,
      status: res.status,
      ...(parsed?.code !== undefined ? { code: parsed.code } : {}),
      field: parsed?.field ?? null,
      ...(parsed?.details !== undefined ? { details: parsed.details } : {}),
    });
  }
  return (await res.json()) as T;
}

export function srvGet<T>(path: string): Promise<T> {
  return srv<T>(path, { method: "GET" });
}

export function srvPost<T>(path: string, body?: unknown): Promise<T> {
  return srv<T>(path, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
}

export function srvDelete<T>(path: string): Promise<T> {
  return srv<T>(path, { method: "DELETE" });
}

// ---- Engine mode --------------------------------------------------------------

export type EngineMode = "desktop" | "server" | "preview";

/** Live engine mode for the status indicator: desktop (Tauri) → server (sidecar
 *  reachable) → preview (offline mock). */
export async function engineMode(): Promise<EngineMode> {
  if (isDesktop()) return "desktop";
  if (await serverReady()) return "server";
  return "preview";
}
