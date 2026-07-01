import test from "node:test";
import assert from "node:assert/strict";
import {
  applyFilters,
  buildInstallPlan,
  isTrustedRegistryUrl,
  normalizePackage,
  validateManifestShape,
} from "../registry.mjs";

test("registry normalization fills derived fields", () => {
  const normalized = normalizePackage(
    {
      name: "source-demo",
      type: "source",
      author: { name: "Ops" },
      artifact: { publisher: "cargo", package: "source-demo", install_command: "" },
    },
    { schemaVersion: "2026-06-14", installed: true },
  );
  assert.equal(normalized.type, "source");
  assert.equal(normalized.installed, true);
  assert.equal(normalized.installCommand, "celeritas add source source-demo");
  assert.equal(normalized.packagedAt, "2026-06-14");
  assert.deepEqual(normalized.packagedBy, { name: "Ops" });
});

test("registry filters collapse source and target types", () => {
  const packages = [
    { name: "source-a", type: "source", category: "files", summary: "csv" },
    { name: "sink-b", type: "sink", category: "cache", summary: "redis" },
  ];
  assert.deepEqual(applyFilters(packages, { type: "target" }).map((pkg) => pkg.name), ["sink-b"]);
  assert.deepEqual(applyFilters(packages, { q: "csv" }).map((pkg) => pkg.name), ["source-a"]);
});

test("trusted registry URL checks use the allowlist", () => {
  assert.equal(isTrustedRegistryUrl("http://127.0.0.1:4321/registry.manifest.json"), true);
  assert.equal(isTrustedRegistryUrl("https://untrusted.example/registry.manifest.json"), false);
});

test("safe install plans are reconstructed from structured fields", () => {
  const cargoPlan = buildInstallPlan({
    name: "sink-redis-ipc",
    type: "sink",
    summary: "Redis",
    artifact: { publisher: "cargo", package: "sink-redis-ipc" },
    manifest: { path: "extensions/sink/cache/redis-ipc/plugin.celeritas.yml" },
  });
  assert.deepEqual(cargoPlan, {
    kind: "command",
    command: "cargo",
    args: ["install", "--path", "extensions/sink/cache/redis-ipc"],
    displayCommand: "cargo install --path extensions/sink/cache/redis-ipc",
  });

  const pipPlan = buildInstallPlan({
    name: "source-stripe",
    type: "source",
    summary: "Stripe",
    artifact: { publisher: "pip", package: "celeritas-source-stripe" },
    manifest: { capabilities: ["discover"], settings: [{ name: "api_key" }] },
  });
  assert.equal(pipPlan.kind, "celeritas");
  assert.deepEqual(pipPlan.args.slice(0, 6), [
    "hub",
    "add",
    "source-stripe",
    "--type",
    "extractor",
    "--pip",
  ]);
  assert.equal(buildInstallPlan({
    name: "source-evil",
    type: "source",
    artifact: { publisher: "evil", package: "payload" },
    manifest: {},
  }), null);
});

test("registry manifest schema validation rejects malformed payloads", () => {
  assert.throws(() => validateManifestShape({ packages: [] }), /missing schema_version/);
  assert.throws(() => validateManifestShape({ schema_version: "1", registry: {}, packages: {} }), /packages must be an array/);
});
