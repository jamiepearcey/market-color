// Where the client talks to the API. The UI is a static build (served by Tauri
// or any static host) and the API is a separate server that may live on another
// machine — so the base URL is configurable at runtime, not baked in.
//
// Resolution order: a user override in localStorage (Settings) > the VITE_API_BASE
// build env > same-origin "". Same-origin works in dev (Vite proxies /api + /chat
// to the API server) and in a single-origin deploy (the API also serves dist/).
// For the Tauri desktop client pointing at a remote API, the user sets the base
// in Settings.

const KEY = "market-color:api-base";

function trim(v: string): string {
  return v.replace(/\/+$/, "");
}

export function apiBase(): string {
  try {
    const v = localStorage.getItem(KEY);
    if (v) return trim(v);
  } catch {
    /* ignore */
  }
  const env = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
  return trim(env);
}

export function setApiBase(v: string): void {
  try {
    if (v.trim()) localStorage.setItem(KEY, trim(v.trim()));
    else localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export function getApiBaseRaw(): string {
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}
