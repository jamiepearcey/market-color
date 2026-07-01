import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readStoreFile, writeStoreFile, STORE_VERSION } from "../store-files.mjs";

test("store files round-trip through the versioned envelope", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "celeritas-store-"));
  const file = path.join(dir, "jobs.json");
  const value = [{ id: "job-1", name: "daily" }];
  writeStoreFile(file, value);
  assert.deepEqual(readStoreFile(file, []), value);
  const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(parsed.version, STORE_VERSION);
  assert.deepEqual(parsed.data, value);
});

test("corrupt store files are quarantined and recreated", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "celeritas-store-"));
  const file = path.join(dir, "prefs.json");
  fs.writeFileSync(file, "{broken");
  const fallback = { theme: "amber" };
  assert.deepEqual(readStoreFile(file, fallback), fallback);
  const backups = fs.readdirSync(dir).filter((entry) => entry.includes(".bak"));
  assert.equal(backups.length, 1);
  const recreated = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(recreated.version, STORE_VERSION);
  assert.deepEqual(recreated.data, fallback);
});
