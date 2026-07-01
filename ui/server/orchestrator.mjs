// market-color run orchestrator.
//
// Owns the daily pipeline and ad-hoc ingest. Postgres is the source of truth for
// config; before a run we materialize the enabled sources + desk taxonomy to JSON
// files and hand them to the Python steps (scrape -> decompose -> index), each
// scoped to a dated partition. After indexing, a daily report is generated. Runs
// are recorded in the `runs` table and progress through statuses the UI polls.

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  createReport,
  createRun,
  getAdhoc,
  listDesks,
  listSources,
  markAdhocIngested,
  updateRun,
} from "./db.mjs";
import { generateReport } from "./reports.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(HERE, "..", ".."); // research/market-color
const CONFIG_DIR = path.join(PROJECT_ROOT, "config");
const TMP_DIR = path.join(PROJECT_ROOT, "data", "tmp");

const PY = {
  scrape: {
    script: path.join(PROJECT_ROOT, "scrape_news_feeds.py"),
    deps: ["feedparser", "httpx", "pyarrow", "python-dateutil", "trafilatura"],
  },
  decompose: {
    script: path.join(PROJECT_ROOT, "decompose_facts.py"),
    deps: ["anthropic>=0.49", "duckdb>=1.0", "pyarrow>=15"],
  },
  index: {
    script: path.join(PROJECT_ROOT, "index_facts.py"),
    deps: ["qdrant-client>=1.15", "duckdb>=1.0", "fastembed"],
  },
};

function today() {
  return new Date().toISOString().slice(0, 10);
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

/** Run a Python step via `uv run --with … python script args…`. Resolves with
 *  the exit code and captured output; stdout carries the step's final JSON. */
function runPy(step, args) {
  return new Promise((resolve) => {
    const cmd = process.env.MARKET_UV || "uv";
    const argv = ["run"];
    for (const d of step.deps) argv.push("--with", d);
    argv.push("python", step.script, ...args);
    const child = spawn(cmd, argv, {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
        MARKET_FACTS_COLLECTION: process.env.MARKET_FACTS_COLLECTION || "market_facts",
      },
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (e) => resolve({ code: -1, stdout, stderr: String(e) }));
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

/** Best-effort parse of the trailing JSON object a step prints to stdout. */
function tailJson(stdout) {
  const a = stdout.indexOf("{");
  const b = stdout.lastIndexOf("}");
  if (a < 0 || b <= a) return {};
  try {
    return JSON.parse(stdout.slice(a, b + 1));
  } catch {
    return {};
  }
}

async function materializeConfig() {
  ensureDir(CONFIG_DIR);
  const sources = await listSources();
  const feeds = sources
    .filter((s) => s.kind === "rss" && s.enabled && s.url)
    .map((s) => ({
      name: s.name,
      url: s.url,
      scope: s.scope,
      tier: s.tier,
      access: s.access,
      method: s.method,
      desks: s.desks,
    }));
  const feedsPath = path.join(CONFIG_DIR, "feeds.generated.json");
  fs.writeFileSync(feedsPath, JSON.stringify({ version: "db", feeds }, null, 2));

  const desks = await listDesks();
  const desksPath = path.join(CONFIG_DIR, "desks.generated.json");
  fs.writeFileSync(
    desksPath,
    JSON.stringify(desks.map((d) => ({ key: d.key, label: d.label, anchor: d.anchor })), null, 2),
  );
  return { feedsPath, desksPath, feedCount: feeds.length };
}

// --- daily run --------------------------------------------------------------
export async function triggerDailyRun(date) {
  const runDate = date || today();
  const run = await createRun({ run_date: runDate, kind: "daily" });
  void _runDaily(run.id, runDate); // fire-and-forget; UI polls status
  return run;
}

async function _runDaily(runId, runDate) {
  const stats = {};
  try {
    const { feedsPath, desksPath, feedCount } = await materializeConfig();
    stats.feeds = feedCount;

    // 1. scrape
    await updateRun(runId, { status: "scraping", stage: `scraping ${feedCount} feeds` });
    const scrape = await runPy(PY.scrape, [
      "--feeds", feedsPath,
      "--since-hours", process.env.MARKET_SINCE_HOURS || "48",
      "--fetch-full-text",
    ]);
    if (scrape.code !== 0) throw new Error(`scrape failed (code ${scrape.code}): ${scrape.stderr.slice(-500)}`);
    Object.assign(stats, { scrape: tailJson(scrape.stdout) });
    await updateRun(runId, { stats });

    // 2. decompose (scoped to the dated partition)
    await updateRun(runId, { status: "decomposing", stage: `decomposing ${runDate}` });
    const dec = await runPy(PY.decompose, [
      "--start-date", runDate, "--end-date", runDate, "--desks-file", desksPath,
    ]);
    if (dec.code !== 0) throw new Error(`decompose failed (code ${dec.code}): ${dec.stderr.slice(-500)}`);
    stats.decompose = tailJson(dec.stdout);
    await updateRun(runId, { stats });

    // 3. index the facts
    await updateRun(runId, { status: "indexing", stage: `indexing ${runDate}` });
    const idx = await runPy(PY.index, ["--start-date", runDate, "--end-date", runDate]);
    if (idx.code !== 0) throw new Error(`index failed (code ${idx.code}): ${idx.stderr.slice(-500)}`);
    stats.index = tailJson(idx.stdout);
    await updateRun(runId, { stats });

    // 4. daily report
    await updateRun(runId, { status: "reporting", stage: "writing report" });
    const report = await createReport({
      run_id: runId,
      kind: "daily",
      run_date: runDate,
      title: `Daily market color — ${runDate}`,
    });
    await generateReport(report.id);
    stats.report_id = report.id;

    await updateRun(runId, {
      status: "done",
      stage: "complete",
      stats,
      finished_at: new Date().toISOString(),
    });
  } catch (e) {
    await updateRun(runId, {
      status: "error",
      error: String(e?.message || e),
      stats,
      finished_at: new Date().toISOString(),
    });
  }
}

// --- ad-hoc ingest ----------------------------------------------------------
export async function ingestAdhoc(docId) {
  const doc = await getAdhoc(docId);
  if (!doc) throw new Error(`no ad-hoc document ${docId}`);
  const runDate = String(doc.published_date).slice(0, 10);
  const run = await createRun({ run_date: runDate, kind: "adhoc-ingest" });
  void _runAdhoc(run.id, doc, runDate);
  return run;
}

async function _runAdhoc(runId, doc, runDate) {
  const stats = { doc_id: doc.doc_id };
  try {
    ensureDir(TMP_DIR);
    const { desksPath } = await materializeConfig();
    // One corpus-shaped row for the uploaded document.
    const row = {
      doc_id: doc.doc_id,
      title: doc.title,
      body_text: doc.body,
      source_name: doc.title || "Ad-hoc upload",
      source_domain: null,
      source_access: "open",
      url: doc.url,
      published_date: runDate,
      published_utc: `${runDate}T00:00:00+00:00`,
      lang: "en",
      word_count: String(doc.body || "").split(/\s+/).length,
    };
    const inputPath = path.join(TMP_DIR, `adhoc-${doc.doc_id}.json`);
    fs.writeFileSync(inputPath, JSON.stringify([row], null, 2));

    await updateRun(runId, { status: "decomposing", stage: "decomposing upload" });
    const dec = await runPy(PY.decompose, ["--input-json", inputPath, "--desks-file", desksPath]);
    if (dec.code !== 0) throw new Error(`decompose failed (code ${dec.code}): ${dec.stderr.slice(-500)}`);
    stats.decompose = tailJson(dec.stdout);

    await updateRun(runId, { status: "indexing", stage: "indexing upload" });
    const idx = await runPy(PY.index, ["--start-date", runDate, "--end-date", runDate]);
    if (idx.code !== 0) throw new Error(`index failed (code ${idx.code}): ${idx.stderr.slice(-500)}`);
    stats.index = tailJson(idx.stdout);

    await markAdhocIngested(doc.doc_id);
    await updateRun(runId, {
      status: "done",
      stage: "complete",
      stats,
      finished_at: new Date().toISOString(),
    });
  } catch (e) {
    await updateRun(runId, {
      status: "error",
      error: String(e?.message || e),
      stats,
      finished_at: new Date().toISOString(),
    });
  }
}
