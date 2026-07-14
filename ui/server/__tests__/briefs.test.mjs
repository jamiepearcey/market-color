import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { briefsDir, listBriefs, readBrief } from "../briefs.mjs";

function withBriefsDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "market-color-briefs-"));
  const prev = process.env.MARKET_BRIEFS_DIR;
  process.env.MARKET_BRIEFS_DIR = dir;
  t.after(() => {
    if (prev === undefined) delete process.env.MARKET_BRIEFS_DIR;
    else process.env.MARKET_BRIEFS_DIR = prev;
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return dir;
}

test("briefsDir honors MARKET_BRIEFS_DIR", (t) => {
  const dir = withBriefsDir(t);
  assert.equal(briefsDir(), dir);
});

test("listBriefs returns [] when the directory does not exist", (t) => {
  const dir = withBriefsDir(t);
  process.env.MARKET_BRIEFS_DIR = path.join(dir, "missing");
  assert.deepEqual(listBriefs(), []);
});

test("listBriefs lists dt=<date>/<desk>.md, newest date first then desk asc", (t) => {
  const dir = withBriefsDir(t);
  fs.mkdirSync(path.join(dir, "dt=2026-06-30"));
  fs.mkdirSync(path.join(dir, "dt=2026-07-01"));
  fs.mkdirSync(path.join(dir, "not-a-partition"));
  fs.writeFileSync(path.join(dir, "dt=2026-06-30", "energy.md"), "# old");
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "macro.md"), "# m");
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "energy.md"), "# e");
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "energy.json"), "{}"); // sibling data, not a brief
  assert.deepEqual(listBriefs(), [
    { date: "2026-07-01", desk: "energy" },
    { date: "2026-07-01", desk: "macro" },
    { date: "2026-06-30", desk: "energy" },
  ]);
});

test("readBrief returns markdown plus parsed sibling json when present", (t) => {
  const dir = withBriefsDir(t);
  fs.mkdirSync(path.join(dir, "dt=2026-07-01"));
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "energy.md"), "# Energy\n\nHormuz reopened.");
  fs.writeFileSync(
    path.join(dir, "dt=2026-07-01", "energy.json"),
    JSON.stringify({ fixture: true, movers: [{ symbol: "CL=F", ret_1d: -0.031 }] }),
  );
  const brief = readBrief("2026-07-01", "energy");
  assert.equal(brief.date, "2026-07-01");
  assert.equal(brief.desk, "energy");
  assert.match(brief.markdown, /Hormuz reopened/);
  assert.equal(brief.data.fixture, true);
  assert.equal(brief.data.movers[0].symbol, "CL=F");
});

test("readBrief tolerates a missing or corrupt sibling json", (t) => {
  const dir = withBriefsDir(t);
  fs.mkdirSync(path.join(dir, "dt=2026-07-01"));
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "macro.md"), "# Macro");
  assert.equal(readBrief("2026-07-01", "macro").data, null);
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "macro.json"), "{broken");
  assert.equal(readBrief("2026-07-01", "macro").data, null);
});

test("readBrief rejects traversal and malformed identifiers", (t) => {
  const dir = withBriefsDir(t);
  fs.mkdirSync(path.join(dir, "dt=2026-07-01"));
  fs.writeFileSync(path.join(dir, "dt=2026-07-01", "energy.md"), "# e");
  fs.writeFileSync(path.join(dir, "secret.md"), "top secret");
  assert.equal(readBrief("2026-07-01", "../secret"), null);
  assert.equal(readBrief("2026-07-01", "..%2Fsecret"), null);
  assert.equal(readBrief("../dt=2026-07-01", "energy"), null);
  assert.equal(readBrief("2026-07-01", "nope"), null); // missing file
});
