// market-color management API client + types.
//
// The chat lives at /chat (NDJSON). Everything else — desks, sources, daily
// runs, and reports, all backed by Postgres — is plain JSON under /api, served
// by the same dev bridge / standalone server. This module is the single
// contract both the UI panels and the backend (server/api.mjs) agree on.

import { apiBase } from "./api-base";

export interface Desk {
  id: number;
  key: string;
  label: string;
  anchor: string;
  created_at: string;
}

export interface Source {
  id: number;
  name: string;
  kind: "rss" | "adhoc";
  url: string | null;
  scope: string;
  tier: number;
  access: string;
  method: string;
  desks: string[];
  enabled: boolean;
  created_at: string;
}

export type RunStatus =
  | "pending"
  | "scraping"
  | "decomposing"
  | "indexing"
  | "reporting"
  | "done"
  | "error";

export interface Run {
  id: number;
  run_date: string;
  kind: string; // 'daily' | 'adhoc-ingest'
  status: RunStatus;
  stage: string | null;
  stats: Record<string, unknown>;
  error: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface ReportFact {
  fact_id?: string;
  claim: string;
  desk?: string;
  direction?: string;
  metric?: string;
  source_name?: string;
  published_date?: string;
  url?: string;
  title?: string;
  score?: number;
}

export interface Report {
  id: number;
  run_id: number | null;
  kind: "daily" | "adhoc";
  run_date: string | null;
  title: string;
  prompt: string | null;
  body: string;
  facts: ReportFact[];
  status: "pending" | "done" | "error";
  error: string | null;
  created_at: string;
}

export interface ApiHealth {
  db: boolean;
  qdrant: boolean;
  anthropic: boolean;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase()}/api${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-json */
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    if (data && typeof data === "object" && "error" in data) {
      msg = String((data as { error: unknown }).error);
    }
    throw new Error(msg);
  }
  return data as T;
}

// --- health ----------------------------------------------------------------
export const getApiHealth = () => json<ApiHealth>("/health");

// --- desks -----------------------------------------------------------------
export const getDesks = () => json<Desk[]>("/desks");
export const addDesk = (d: { key: string; label: string; anchor: string }) =>
  json<Desk>("/desks", { method: "POST", body: JSON.stringify(d) });
export const deleteDesk = (id: number) =>
  json<{ ok: true }>(`/desks/${id}`, { method: "DELETE" });

// --- sources ---------------------------------------------------------------
export const getSources = () => json<Source[]>("/sources");
export const addSource = (s: {
  name: string;
  url: string;
  scope?: string;
  tier?: number;
  access?: string;
  method?: string;
  desks?: string[];
}) => json<Source>("/sources", { method: "POST", body: JSON.stringify(s) });
export const setSourceEnabled = (id: number, enabled: boolean) =>
  json<Source>(`/sources/${id}`, { method: "PATCH", body: JSON.stringify({ enabled }) });
export const deleteSource = (id: number) =>
  json<{ ok: true }>(`/sources/${id}`, { method: "DELETE" });

/** Manually ingest an ad-hoc source / market-color report. Returns the run that
 *  decomposes + indexes it into facts. */
export const uploadAdhoc = (doc: {
  title: string;
  body: string;
  url?: string;
  published_date?: string;
  desks?: string[];
}) =>
  json<{ document: { doc_id: string }; run: Run }>("/sources/upload", {
    method: "POST",
    body: JSON.stringify(doc),
  });

// --- runs ------------------------------------------------------------------
export const getRuns = () => json<Run[]>("/runs");
export const getRun = (id: number) => json<Run>(`/runs/${id}`);
/** Trigger a daily news run for a date (default: today). Returns immediately;
 *  poll getRun for progress. */
export const triggerRun = (date?: string) =>
  json<Run>("/runs", { method: "POST", body: JSON.stringify({ date: date ?? null }) });

// --- graph (transmission map) ------------------------------------------------
export interface GraphNode {
  id: string;
  count: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  /** true = directed cause → effect edge (from `cause_entities`); false = undirected co-occurrence */
  causal: boolean;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** Raw Qdrant fact payload (fields are best-effort — render defensively). */
export interface FactPayload {
  fact_id?: string;
  doc_id?: string;
  claim?: string;
  subject?: string;
  predicate?: string;
  object?: string | null;
  direction?: string;
  magnitude?: string;
  time?: string;
  cause?: string | null;
  entities?: string[];
  cause_entities?: string[];
  confidence?: number;
  desk?: string;
  source_name?: string;
  published_date?: string;
  published_utc?: string;
  url?: string;
  doc_title?: string;
  title?: string;
}

export interface EntityFacts {
  entity: string;
  facts: FactPayload[];
}

export const getGraph = (p: {
  desk?: string;
  since?: string;
  until?: string;
  minWeight?: number;
  maxNodes?: number;
}) => {
  const q = new URLSearchParams();
  if (p.desk) q.set("desk", p.desk);
  if (p.since) q.set("since", p.since);
  if (p.until) q.set("until", p.until);
  if (p.minWeight != null) q.set("min_weight", String(p.minWeight));
  if (p.maxNodes != null) q.set("max_nodes", String(p.maxNodes));
  const qs = q.toString();
  return json<Graph>(`/graph${qs ? `?${qs}` : ""}`);
};

export const getEntityFacts = (name: string, desk?: string) => {
  const q = new URLSearchParams({ name });
  if (desk) q.set("desk", desk);
  return json<EntityFacts>(`/graph/entity?${q.toString()}`);
};

// --- briefs ------------------------------------------------------------------
export interface BriefRef {
  date: string;
  desk: string;
}

export interface BriefMover {
  symbol?: string;
  name?: string;
  close?: number;
  ret_1d?: number;
  zscore_20d?: number;
}

export interface BriefSectionFact {
  claim?: string;
  source_name?: string;
  published_date?: string;
  url?: string;
  corroboration?: number;
  is_new?: boolean;
}

export interface BriefSection {
  kind?: "driver" | "no_driver" | string;
  anchor?: string;
  title?: string;
  chains?: [string, string, string][];
  facts?: BriefSectionFact[];
}

/** Sibling <desk>.json payload — schema owned by the brief generator; every
 *  field is optional so old/fixture briefs still render. */
export interface BriefData {
  fixture?: boolean;
  desk?: string;
  date?: string;
  generated_utc?: string;
  movers?: BriefMover[];
  sections?: BriefSection[];
  new_fact_count?: number;
}

export interface Brief {
  date: string;
  desk: string;
  markdown: string;
  data: BriefData | null;
}

export const getBriefs = () => json<BriefRef[]>("/briefs");
export const getBrief = (date: string, desk: string) =>
  json<Brief>(`/briefs/${encodeURIComponent(date)}/${encodeURIComponent(desk)}`);

// --- reports ---------------------------------------------------------------
export const getReports = () => json<Report[]>("/reports");
export const getReport = (id: number) => json<Report>(`/reports/${id}`);
/** Generate an ad-hoc report from a prompt, grounded in the corpus. Returns
 *  immediately with a pending report; poll getReport for the result. */
export const generateReport = (req: { prompt: string; title?: string; run_date?: string }) =>
  json<Report>("/reports", { method: "POST", body: JSON.stringify(req) });
