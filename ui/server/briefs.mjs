// Desk briefs — read-only file surface over data/briefs/dt=YYYY-MM-DD/<desk>.md
// (+ optional sibling <desk>.json with movers/sections). Another workstream
// writes the briefs; we only list and read them. Node built-ins only.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const DESK_RE = /^[a-z0-9][a-z0-9_-]*$/i;

export function briefsDir() {
  return process.env.MARKET_BRIEFS_DIR || path.resolve(HERE, "..", "..", "data", "briefs");
}

/** All available briefs, newest date first: [{date, desk}]. */
export function listBriefs() {
  const dir = briefsDir();
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return []; // no briefs generated yet
  }
  const briefs = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const m = /^dt=(\d{4}-\d{2}-\d{2})$/.exec(e.name);
    if (!m) continue;
    let files;
    try {
      files = fs.readdirSync(path.join(dir, e.name));
    } catch {
      continue;
    }
    for (const f of files) {
      if (!f.endsWith(".md")) continue;
      const desk = f.slice(0, -3);
      if (DESK_RE.test(desk)) briefs.push({ date: m[1], desk });
    }
  }
  briefs.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.desk < b.desk ? -1 : 1));
  return briefs;
}

/** One brief, or null. Date/desk are validated (no path traversal) and the
 *  resolved path is double-checked to stay inside the briefs dir. */
export function readBrief(date, desk) {
  if (!DATE_RE.test(String(date)) || !DESK_RE.test(String(desk))) return null;
  const dir = briefsDir();
  const mdFile = path.resolve(dir, `dt=${date}`, `${desk}.md`);
  if (!mdFile.startsWith(path.resolve(dir) + path.sep)) return null;
  let markdown;
  try {
    markdown = fs.readFileSync(mdFile, "utf8");
  } catch {
    return null;
  }
  let data = null;
  try {
    data = JSON.parse(fs.readFileSync(mdFile.slice(0, -3) + ".json", "utf8"));
  } catch {
    /* no sibling json (or unparseable) — markdown-only brief */
  }
  return { date, desk, markdown, data };
}
