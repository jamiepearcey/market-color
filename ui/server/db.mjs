// market-color config + state store (Postgres).
//
// Single owner of the database. Desks, sources (RSS + ad-hoc uploads), runs, and
// reports all live here; the Python pipeline never touches Postgres — the
// orchestrator materializes config to JSON for it. migrate() is idempotent and
// self-seeds the default desk taxonomy and the curated feed list so a fresh DB
// is immediately usable.
//
// Connection: DATABASE_URL, else PG* env vars, else postgres://localhost/market_color.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(HERE, "..", ".."); // research/market-color
const FEEDS_JSON = path.join(PROJECT_ROOT, "config", "feeds.json");

const DEFAULT_URL = "postgres://localhost:5432/market_color";

let pool = null;
let migrated = null;

export function getPool() {
  if (!pool) {
    pool = new pg.Pool({ connectionString: process.env.DATABASE_URL || DEFAULT_URL });
  }
  return pool;
}

async function q(text, params = []) {
  return getPool().query(text, params);
}

const SCHEMA = `
create table if not exists desks (
  id serial primary key,
  key text unique not null,
  label text not null,
  anchor text not null default '',
  created_at timestamptz not null default now()
);
create table if not exists sources (
  id serial primary key,
  name text not null,
  kind text not null default 'rss',
  url text,
  scope text not null default 'global',
  tier int not null default 2,
  access text not null default 'open',
  method text not null default 'rss',
  domain text,
  desks text[] not null default '{}',
  enabled boolean not null default true,
  created_at timestamptz not null default now()
);
create table if not exists adhoc_documents (
  id serial primary key,
  source_id int references sources(id) on delete set null,
  doc_id text unique not null,
  title text,
  body text not null,
  url text,
  published_date date not null,
  desks text[] not null default '{}',
  ingested boolean not null default false,
  created_at timestamptz not null default now()
);
create table if not exists runs (
  id serial primary key,
  run_date date not null,
  kind text not null default 'daily',
  status text not null default 'pending',
  stage text,
  stats jsonb not null default '{}',
  error text,
  started_at timestamptz not null default now(),
  finished_at timestamptz
);
create table if not exists reports (
  id serial primary key,
  run_id int references runs(id) on delete set null,
  kind text not null default 'daily',
  run_date date,
  title text not null,
  prompt text,
  body text not null default '',
  facts jsonb not null default '[]',
  status text not null default 'pending',
  error text,
  created_at timestamptz not null default now()
);
create table if not exists settings (
  key text primary key,
  value text
);
`;

// Default desk taxonomy — mirrors index_corpus.DESK_ANCHORS so a fresh DB
// classifies the same way the original pipeline did.
const DEFAULT_DESKS = [
  ["rates", "Rates", "central bank interest rate policy, Federal Reserve and ECB rate decisions, Treasury and government bond yields, the yield curve, rate hikes and monetary tightening"],
  ["fx", "FX", "foreign exchange currency markets, the US dollar index DXY, euro, yen, sterling and currency pairs, FX trading and central bank intervention"],
  ["energy", "Energy", "crude oil Brent and WTI prices, OPEC supply and production, natural gas and LNG, refineries and oil inventories, energy markets"],
  ["metals", "Metals", "gold and silver precious metals, copper and industrial base metals, mining and smelting, metals prices and demand"],
  ["crypto", "Crypto", "Bitcoin and Ethereum cryptocurrency, digital assets, stablecoins, crypto ETFs and exchanges"],
  ["equities", "Equities", "stock markets, S&P 500 and Nasdaq indices, company earnings and single stocks, equity rally and selloff"],
  ["geopolitics", "Geopolitics", "war conflict and military strikes, sanctions, elections, coups and political instability, geopolitical risk"],
  ["macro", "Macro", "macroeconomic data, inflation GDP and employment, recession risk, the global economy and trade, central bank policy"],
  ["asia", "Asia", "China and Japan economies and markets, the yuan and yen, Chinese property and exports, Asian equities and Asia session"],
];

async function seedDesks() {
  const { rows } = await q("select count(*)::int as n from desks");
  if (rows[0].n > 0) return;
  for (const [key, label, anchor] of DEFAULT_DESKS) {
    await q(
      "insert into desks (key, label, anchor) values ($1,$2,$3) on conflict (key) do nothing",
      [key, label, anchor],
    );
  }
}

async function seedSources() {
  const { rows } = await q("select count(*)::int as n from sources");
  if (rows[0].n > 0) return;
  let feeds = [];
  try {
    const data = JSON.parse(fs.readFileSync(FEEDS_JSON, "utf8"));
    feeds = Array.isArray(data) ? data : data.feeds ?? [];
  } catch {
    feeds = [];
  }
  for (const f of feeds) {
    await q(
      `insert into sources (name, kind, url, scope, tier, access, method, desks, enabled)
       values ($1,'rss',$2,$3,$4,$5,$6,$7,true)`,
      [
        f.name ?? "(unnamed)",
        f.url ?? null,
        f.scope ?? "global",
        Number.isFinite(f.tier) ? f.tier : 2,
        f.access ?? "open",
        f.method ?? "rss",
        Array.isArray(f.desks) ? f.desks : [],
      ],
    );
  }
}

export function migrate() {
  if (!migrated) {
    migrated = (async () => {
      await q(SCHEMA);
      await seedDesks();
      await seedSources();
      await q(
        "insert into settings (key, value) values ('default_model','claude-opus-4-8') on conflict (key) do nothing",
      );
    })();
  }
  return migrated;
}

export async function dbReady() {
  try {
    await migrate();
    await q("select 1");
    return true;
  } catch {
    return false;
  }
}

// --- desks -----------------------------------------------------------------
export async function listDesks() {
  await migrate();
  const { rows } = await q("select * from desks order by key asc");
  return rows;
}
export async function addDesk({ key, label, anchor }) {
  await migrate();
  const { rows } = await q(
    `insert into desks (key, label, anchor) values ($1,$2,$3)
     on conflict (key) do update set label = excluded.label, anchor = excluded.anchor
     returning *`,
    [key, label, anchor ?? ""],
  );
  return rows[0];
}
export async function deleteDesk(id) {
  await migrate();
  await q("delete from desks where id = $1", [id]);
  return { ok: true };
}

// --- sources ---------------------------------------------------------------
export async function listSources() {
  await migrate();
  const { rows } = await q("select * from sources order by kind asc, name asc");
  return rows;
}
export async function addSource(s) {
  await migrate();
  const { rows } = await q(
    `insert into sources (name, kind, url, scope, tier, access, method, desks, enabled)
     values ($1,$2,$3,$4,$5,$6,$7,$8,true) returning *`,
    [
      s.name,
      s.kind ?? "rss",
      s.url ?? null,
      s.scope ?? "global",
      Number.isFinite(s.tier) ? s.tier : 2,
      s.access ?? "open",
      s.method ?? "rss",
      Array.isArray(s.desks) ? s.desks : [],
    ],
  );
  return rows[0];
}
export async function updateSource(id, patch) {
  await migrate();
  const { rows } = await q(
    "update sources set enabled = coalesce($2, enabled) where id = $1 returning *",
    [id, typeof patch.enabled === "boolean" ? patch.enabled : null],
  );
  return rows[0];
}
export async function deleteSource(id) {
  await migrate();
  await q("delete from sources where id = $1", [id]);
  return { ok: true };
}

// --- ad-hoc documents ------------------------------------------------------
export async function addAdhoc(doc) {
  await migrate();
  const src = await addSource({
    name: doc.title || "Ad-hoc upload",
    kind: "adhoc",
    url: doc.url ?? null,
    method: "upload",
    access: "open",
    desks: doc.desks ?? [],
  });
  const { rows } = await q(
    `insert into adhoc_documents (source_id, doc_id, title, body, url, published_date, desks)
     values ($1,$2,$3,$4,$5,$6,$7)
     on conflict (doc_id) do update set body = excluded.body, title = excluded.title
     returning *`,
    [
      src.id,
      doc.doc_id,
      doc.title ?? null,
      doc.body,
      doc.url ?? null,
      doc.published_date,
      Array.isArray(doc.desks) ? doc.desks : [],
    ],
  );
  return rows[0];
}
export async function getAdhoc(docId) {
  await migrate();
  const { rows } = await q("select * from adhoc_documents where doc_id = $1", [docId]);
  return rows[0] ?? null;
}
export async function markAdhocIngested(docId) {
  await q("update adhoc_documents set ingested = true where doc_id = $1", [docId]);
}

// --- runs ------------------------------------------------------------------
export async function createRun({ run_date, kind }) {
  await migrate();
  const { rows } = await q(
    "insert into runs (run_date, kind, status, stage) values ($1,$2,'pending','queued') returning *",
    [run_date, kind ?? "daily"],
  );
  return rows[0];
}
export async function updateRun(id, patch) {
  const sets = [];
  const params = [id];
  let i = 2;
  for (const [k, v] of Object.entries(patch)) {
    if (k === "stats") {
      sets.push(`stats = $${i}::jsonb`);
      params.push(JSON.stringify(v));
    } else {
      sets.push(`${k} = $${i}`);
      params.push(v);
    }
    i++;
  }
  if (!sets.length) return getRun(id);
  const { rows } = await q(`update runs set ${sets.join(", ")} where id = $1 returning *`, params);
  return rows[0];
}
export async function getRun(id) {
  await migrate();
  const { rows } = await q("select * from runs where id = $1", [id]);
  return rows[0] ?? null;
}
export async function listRuns(limit = 50) {
  await migrate();
  const { rows } = await q("select * from runs order by started_at desc limit $1", [limit]);
  return rows;
}

// --- reports ---------------------------------------------------------------
export async function createReport(r) {
  await migrate();
  const { rows } = await q(
    `insert into reports (run_id, kind, run_date, title, prompt, status)
     values ($1,$2,$3,$4,$5,'pending') returning *`,
    [r.run_id ?? null, r.kind ?? "adhoc", r.run_date ?? null, r.title, r.prompt ?? null],
  );
  return rows[0];
}
export async function updateReport(id, patch) {
  const sets = [];
  const params = [id];
  let i = 2;
  for (const [k, v] of Object.entries(patch)) {
    if (k === "facts") {
      sets.push(`facts = $${i}::jsonb`);
      params.push(JSON.stringify(v));
    } else {
      sets.push(`${k} = $${i}`);
      params.push(v);
    }
    i++;
  }
  if (!sets.length) return getReport(id);
  const { rows } = await q(`update reports set ${sets.join(", ")} where id = $1 returning *`, params);
  return rows[0];
}
export async function getReport(id) {
  await migrate();
  const { rows } = await q("select * from reports where id = $1", [id]);
  return rows[0] ?? null;
}
export async function listReports(limit = 50) {
  await migrate();
  const { rows } = await q("select * from reports order by created_at desc limit $1", [limit]);
  return rows;
}

// --- settings --------------------------------------------------------------
export async function getSetting(key, fallback = null) {
  await migrate();
  const { rows } = await q("select value from settings where key = $1", [key]);
  return rows[0]?.value ?? fallback;
}
export async function setSetting(key, value) {
  await migrate();
  await q(
    "insert into settings (key, value) values ($1,$2) on conflict (key) do update set value = excluded.value",
    [key, value],
  );
  return { ok: true };
}
