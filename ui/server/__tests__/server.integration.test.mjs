import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import net from "node:net";
import { once } from "node:events";
import { spawn } from "node:child_process";
import http from "node:http";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import fs from "node:fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UI_DIR = path.resolve(__dirname, "../..");
const SERVER_ENTRY = path.join(UI_DIR, "server", "celeritas-server.mjs");

async function getFreePort() {
  const server = net.createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

function nonLoopbackIpv4() {
  for (const addresses of Object.values(os.networkInterfaces())) {
    for (const address of addresses || []) {
      if (address.family === "IPv4" && !address.internal) {
        return address.address;
      }
    }
  }
  return null;
}

async function startServer(env = {}) {
  const homeDir =
    env.HOME ?? fs.mkdtempSync(path.join(os.tmpdir(), "celeritas-home-"));
  const port = env.PORT ?? (await getFreePort());
  const child = spawn(process.execPath, [SERVER_ENTRY], {
    cwd: UI_DIR,
    env: {
      ...process.env,
      HOME: homeDir,
      HOST: "127.0.0.1",
      LOG_FORMAT: "json",
      LOG_LEVEL: "info",
      PORT: String(port),
      ...env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(
          new Error(
            `server did not start\nstdout:\n${stdout}\nstderr:\n${stderr}`,
          ),
        ),
      5000,
    );
    child.stdout.on("data", () => {
      if (stdout.includes('"msg":"server listening"')) {
        clearTimeout(timer);
        resolve();
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(
        new Error(
          `server exited early with ${code}\nstdout:\n${stdout}\nstderr:\n${stderr}`,
        ),
      );
    });
  });

  return {
    port,
    homeDir,
    async fetch(pathname, init = {}) {
      const method = init.method || "GET";
      const headers = new Headers(init.headers || {});
      if (["POST", "PUT", "DELETE", "PATCH"].includes(method) && !headers.has("Origin")) {
        headers.set("Origin", "http://127.0.0.1:5173");
      }
      return fetch(`http://127.0.0.1:${port}${pathname}`, {
        ...init,
        headers,
      });
    },
    output() {
      return { stderr, stdout };
    },
    async stop() {
      if (child.exitCode !== null) return;
      child.kill("SIGTERM");
      await once(child, "exit");
    },
  };
}

async function spawnAndWaitForExit(env = {}) {
  const port = env.PORT ?? (await getFreePort());
  const child = spawn(process.execPath, [SERVER_ENTRY], {
    cwd: UI_DIR,
    env: {
      ...process.env,
      LOG_FORMAT: "json",
      LOG_LEVEL: "warn",
      PORT: String(port),
      ...env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });
  const [code] = await once(child, "exit");
  return { code, stdout, stderr };
}

async function startRegistryManifestServer(handler) {
  const server = http.createServer(async (req, res) => {
    const payload =
      typeof handler === "function" ? await handler(req, res) : handler;
    if (res.writableEnded) return;
    res.statusCode = 200;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify(payload));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}/registry.manifest.json`,
    async stop() {
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

function openEventStream(port, pathname) {
  let body = "";
  let readyResolve;
  const ready = new Promise((resolve) => {
    readyResolve = resolve;
  });
  const done = new Promise((resolve, reject) => {
    const req = http.get(`http://127.0.0.1:${port}${pathname}`, (res) => {
      readyResolve(res);
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => resolve(body));
      res.on("error", reject);
    });
    req.on("error", reject);
  });
  return { ready, done };
}

function manifestChecksum(manifest) {
  return `sha256:${createHash("sha256").update(JSON.stringify(manifest)).digest("hex")}`;
}

test("validation errors return 400 with the offending field", async () => {
  const server = await startServer();
  try {
    const missing = await server.fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(missing.status, 400);
    assert.deepEqual(await missing.json(), {
      code: "VALIDATION",
      message: "name must be a non-empty string",
      error: "name must be a non-empty string",
      field: "name",
    });

    const extra = await server.fetch("/api/targets", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "Local", extra: true }),
    });
    assert.equal(extra.status, 400);
    assert.deepEqual(await extra.json(), {
      code: "VALIDATION",
      message: "unexpected field 'extra'",
      error: "unexpected field 'extra'",
      field: "extra",
    });

    const malicious = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ driver: "bad;rm -rf /" }),
    });
    assert.equal(malicious.status, 400);
    assert.deepEqual(await malicious.json(), {
      code: "VALIDATION",
      message: "driver must match /^[a-z0-9][a-z0-9._/-]*$/",
      error: "driver must match /^[a-z0-9][a-z0-9._/-]*$/",
      field: "driver",
    });
  } finally {
    await server.stop();
  }
});

test("CORS allows configured local origins and rejects disallowed mutating origins", async () => {
  const server = await startServer();
  try {
    const allowed = await server.fetch("/api/engine", {
      headers: { Origin: "http://127.0.0.1:5173" },
    });
    assert.equal(allowed.status, 200);
    assert.equal(
      allowed.headers.get("access-control-allow-origin"),
      "http://127.0.0.1:5173",
    );

    const denied = await server.fetch("/api/jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        Origin: "https://evil.example",
      },
      body: JSON.stringify({ name: "job" }),
    });
    assert.equal(denied.status, 403);
    assert.deepEqual(await denied.json(), {
      code: "ORIGIN_FORBIDDEN",
      message: "origin not allowed",
      error: "origin not allowed",
    });
  } finally {
    await server.stop();
  }
});

test("default loopback bind is not reachable via a non-loopback interface when available", async (t) => {
  const externalIp = nonLoopbackIpv4();
  if (!externalIp) {
    t.skip("no non-loopback IPv4 interface available");
    return;
  }
  const server = await startServer();
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1000);
    await assert.rejects(
      fetch(`http://${externalIp}:${server.port}/api/engine`, {
        signal: controller.signal,
      }),
    );
    clearTimeout(timer);
  } finally {
    await server.stop();
  }
});

test("non-loopback bind without a token refuses to boot", async () => {
  const result = await spawnAndWaitForExit({ HOST: "0.0.0.0" });
  assert.equal(result.code, 1);
  assert.match(
    result.stderr,
    /Refusing to bind to 0\.0\.0\.0 without CELERITAS_API_TOKEN/,
  );
});

test("token auth protects API routes while leaving health open", async () => {
  const server = await startServer({
    HOST: "0.0.0.0",
    CELERITAS_API_TOKEN: "secret-token",
  });
  try {
    const unauthorized = await server.fetch("/api/engine");
    assert.equal(unauthorized.status, 401);
    assert.equal(unauthorized.headers.get("www-authenticate"), "Bearer");
    assert.deepEqual(await unauthorized.json(), {
      code: "AUTH_REQUIRED",
      message: "authorization required",
      error: "authorization required",
    });

    const authorized = await server.fetch("/api/engine", {
      headers: { Authorization: "Bearer secret-token" },
    });
    assert.equal(authorized.status, 200);

    const health = await server.fetch("/api/health");
    assert.equal(health.status, 200);

    const probe = await server.fetch("/health");
    assert.equal(probe.status, 200);
  } finally {
    await server.stop();
  }
});

test("livez stays up while readyz reflects engine availability", async () => {
  const missingBin = path.join(os.tmpdir(), `celeritas-readyz-missing-${process.pid}.sh`);
  fs.rmSync(missingBin, { force: true });
  const missing = await startServer({ CELERITAS_BIN: missingBin });
  const livez = await missing.fetch("/api/livez");
  assert.equal(livez.status, 200);
  const readyzMissing = await missing.fetch("/api/readyz");
  assert.equal(readyzMissing.status, 503);
  await missing.stop();

  const goodBin = path.join(os.tmpdir(), `celeritas-readyz-good-${process.pid}.sh`);
  writeStubCeleritas(
    goodBin,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.1.0"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
exit 0
`,
  );
  const ready = await startServer({ CELERITAS_BIN: goodBin });
  try {
    const readyz = await ready.fetch("/api/readyz");
    assert.equal(readyz.status, 200);
    assert.deepEqual(await readyz.json(), { ok: true });
  } finally {
    await ready.stop();
    fs.rmSync(goodBin, { force: true });
  }
});

test("mutating requests without an allowed origin or token are rejected", async () => {
  const server = await startServer();
  try {
    const response = await fetch(`http://127.0.0.1:${server.port}/api/jobs`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "job" }),
    });
    assert.equal(response.status, 403);
    assert.deepEqual(await response.json(), {
      code: "ORIGIN_FORBIDDEN",
      message: "mutating requests require an allowed Origin or Referer",
      error: "mutating requests require an allowed Origin or Referer",
    });
  } finally {
    await server.stop();
  }
});

test("remote targets are explicitly gated off in server mode", async () => {
  const server = await startServer();
  try {
    const response = await server.fetch("/api/targets", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: "Remote sidecar",
        kind: "remote",
        url: "http://127.0.0.1:8788",
      }),
    });
    assert.equal(response.status, 501);
    assert.deepEqual(await response.json(), {
      code: "CONFIG_INVALID",
      message:
        "remote targets are not available yet; use the embedded local engine",
      error:
        "remote targets are not available yet; use the embedded local engine",
    });
  } finally {
    await server.stop();
  }
});

test("path traversal is rejected for fs listing and filesystem-like connector params", async () => {
  const server = await startServer();
  try {
    const fsList = await server.fetch("/api/fs/list?path=/etc");
    assert.equal(fsList.status, 400);
    const fsListBody = await fsList.json();
    assert.equal(fsListBody.code, "VALIDATION");

    const connector = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        driver: "source-demo",
        kind: "source",
        params: { file_path: "/etc/passwd" },
      }),
    });
    assert.equal(connector.status, 400);
    const connectorBody = await connector.json();
    assert.equal(connectorBody.code, "VALIDATION");
  } finally {
    await server.stop();
  }
});

test("metrics endpoint exposes Prometheus counters and histogram text", async () => {
  const server = await startServer();
  try {
    await server.fetch("/api/engine");
    const metrics = await server.fetch("/metrics");
    assert.equal(metrics.status, 200);
    const body = await metrics.text();
    assert.match(body, /# TYPE celeritas_requests_total counter/);
    assert.match(body, /celeritas_requests_total\{method="GET",path="\/api\/engine",status="200"\}/);
    assert.match(body, /# TYPE celeritas_request_duration_ms histogram/);
  } finally {
    await server.stop();
  }
});

test("run requests are rate-limited with 429 and Retry-After", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-rate-limit-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.1.0"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "run" ]; then
  sleep 0.2
  echo "Total records: 1"
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({
    CELERITAS_BIN: bin,
    RUN_MAX_INFLIGHT: "1",
    RUN_RATE_LIMIT_COUNT: "10",
  });
  try {
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "limited-run",
          definition: {
            source: "source-demo",
            steps: [{ template: "sink-demo", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const first = server.fetch(`/api/jobs/${job.id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    const second = server.fetch(`/api/jobs/${job.id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    const [, throttled] = await Promise.all([first, second]);
    assert.equal(throttled.status, 429);
    assert.equal(throttled.headers.get("retry-after"), "60");
    const throttledBody = await throttled.json();
    assert.equal(throttledBody.code, "RATE_LIMITED");
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("registry install refuses untrusted registries and non-allowlisted publishers", async () => {
  const untrustedManifestBody = {
    schema_version: "1",
    registry: { name: "Untrusted" },
    packages: [
      {
        name: "source-rogue",
        type: "source",
        artifact: {
          publisher: "cargo",
          package: "source-rogue",
          install_command: "cargo install --path rogue",
        },
        manifest: { path: "extensions/source/rogue/plugin.celeritas.yml" },
      },
    ],
  };
  const untrustedManifest = await startRegistryManifestServer(untrustedManifestBody);
  const trustedManifestBody = {
    schema_version: "1",
    registry: { name: "Trusted" },
    packages: [
      {
        name: "source-rogue",
        type: "source",
        artifact: {
          publisher: "evil",
          package: "source-rogue",
          install_command: "evil install source-rogue",
        },
        manifest: { path: "extensions/source/rogue/plugin.celeritas.yml" },
      },
    ],
  };
  const trustedManifest = await startRegistryManifestServer(trustedManifestBody);
  const untrustedServer = await startServer();
  const trustedServer = await startServer({
    CELERITAS_REGISTRY_URL: trustedManifest.url,
    CELERITAS_REGISTRY_PINS: JSON.stringify({
      [trustedManifest.url]: manifestChecksum(trustedManifestBody),
    }),
  });
  try {
    const untrusted = await untrustedServer.fetch("/api/registry/install", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: "source-rogue",
        url: untrustedManifest.url,
      }),
    });
    assert.equal(untrusted.status, 403);
    const untrustedBody = await untrusted.json();
    assert.equal(untrustedBody.code, "REGISTRY_UNTRUSTED");
    assert.match(untrustedBody.message, /trusted registry URL list/);

    const publisherRefused = await trustedServer.fetch(
      "/api/registry/install",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "source-rogue",
          url: trustedManifest.url,
        }),
      },
    );
    assert.equal(publisherRefused.status, 403);
    const refusedBody = await publisherRefused.json();
    assert.equal(refusedBody.code, "REGISTRY_UNTRUSTED");
    assert.match(refusedBody.message, /publisher 'evil' is not allowlisted/);
  } finally {
    await untrustedServer.stop();
    await trustedServer.stop();
    await untrustedManifest.stop();
    await trustedManifest.stop();
  }
});

test("registry fetch fails fast on timeout, rejects malformed manifests, and serves stale cache", async () => {
  let mode = "ok";
  const manifestServer = await startRegistryManifestServer(
    async (_req, _res) => {
      if (mode === "hang") {
        await new Promise((resolve) => setTimeout(resolve, 200));
        return {
          schema_version: "1",
          registry: {},
          packages: [],
        };
      }
      if (mode === "malformed") {
        return { nope: true };
      }
      return {
        schema_version: "1",
        registry: { name: "Local" },
        packages: [
          {
            name: "source-demo",
            type: "source",
            artifact: {
              publisher: "cargo",
              package: "source-demo",
              install_command: "",
            },
            manifest: { path: "extensions/source/demo/plugin.celeritas.yml" },
          },
        ],
      };
    },
  );
  const server = await startServer({
    CELERITAS_REGISTRY_URL: manifestServer.url,
    REGISTRY_CACHE_TTL_MS: "10",
    REGISTRY_FETCH_TIMEOUT_MS: "50",
  });
  try {
    const first = await server.fetch("/api/registry");
    assert.equal(first.status, 200);
    const firstBody = await first.json();
    assert.equal(firstBody.packages.length, 1);

    mode = "malformed";
    await new Promise((resolve) => setTimeout(resolve, 20));
    const stale = await server.fetch("/api/registry");
    assert.equal(stale.status, 200);
    const staleBody = await stale.json();
    assert.equal(staleBody.packages.length, 1);

    mode = "hang";
    const timed = await server.fetch(
      "/api/registry?url=http://127.0.0.1:9/registry.manifest.json",
    );
    assert.equal(timed.status, 502);
    const timedBody = await timed.json();
    assert.equal(timedBody.code, "REGISTRY_FETCH_FAILED");
    assert.match(timedBody.error, /failed to fetch registry manifest/);
  } finally {
    await server.stop();
    await manifestServer.stop();
  }
});

test("health and CLI routes report ENGINE_MISSING and recover when the binary appears", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-missing-${process.pid}.sh`);
  fs.rmSync(bin, { force: true });
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const healthMissing = await server.fetch("/api/health");
    assert.equal(healthMissing.status, 200);
    assert.deepEqual(await healthMissing.json(), {
      ok: true,
      engine: false,
      code: "ENGINE_MISSING",
      message: `Celeritas binary not found at ${bin}. Build it or set CELERITAS_BIN to a valid executable.`,
      version: "unknown",
      buildVersion: "0.1.0",
      buildSha: "dev",
      compatible: false,
      supportedRange: ">=0.1.0 <0.2.0",
      locks: { healthy: true, stale: [], missing: [] },
      project: path.join(server.homeDir, ".celeritas-ui", "project"),
    });

    const schedulesMissing = await server.fetch("/api/schedules");
    assert.equal(schedulesMissing.status, 503);
    assert.deepEqual(await schedulesMissing.json(), {
      code: "ENGINE_MISSING",
      message: `Celeritas binary not found at ${bin}. Build it or set CELERITAS_BIN to a valid executable.`,
      error: `Celeritas binary not found at ${bin}. Build it or set CELERITAS_BIN to a valid executable.`,
    });

    writeStubCeleritas(
      bin,
      `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.0.1"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "schedule" ] && [ "$2" = "list" ]; then
  exit 0
fi
exit 0
`,
    );

    const healthRecovered = await server.fetch("/api/health");
    assert.equal(healthRecovered.status, 200);
    const healthRecoveredBody = await healthRecovered.json();
    assert.equal(healthRecoveredBody.engine, "celeritas");
    assert.equal(healthRecoveredBody.version, "0.0.1");

    const schedulesRecovered = await server.fetch("/api/schedules");
    assert.equal(schedulesRecovered.status, 200);
    assert.deepEqual(await schedulesRecovered.json(), []);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("health reports engine version compatibility", async () => {
  const incompatibleBin = path.join(
    os.tmpdir(),
    `celeritas-version-bad-${process.pid}.sh`,
  );
  const compatibleBin = path.join(
    os.tmpdir(),
    `celeritas-version-good-${process.pid}.sh`,
  );
  writeStubCeleritas(
    incompatibleBin,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.0.1"
  exit 0
fi
exit 0
`,
  );
  writeStubCeleritas(
    compatibleBin,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.1.0"
  exit 0
fi
exit 0
`,
  );

  const incompatible = await startServer({ CELERITAS_BIN: incompatibleBin });
  const compatible = await startServer({ CELERITAS_BIN: compatibleBin });
  try {
    const badHealth = await incompatible.fetch("/api/health");
    assert.equal(badHealth.status, 200);
    assert.deepEqual(await badHealth.json(), {
      ok: true,
      engine: "celeritas",
      version: "0.0.1",
      buildVersion: "0.1.0",
      buildSha: "dev",
      compatible: false,
      supportedRange: ">=0.1.0 <0.2.0",
      locks: { healthy: true, stale: [], missing: [] },
      project: path.join(incompatible.homeDir, ".celeritas-ui", "project"),
    });

    const goodHealth = await compatible.fetch("/api/health");
    assert.equal(goodHealth.status, 200);
    assert.deepEqual(await goodHealth.json(), {
      ok: true,
      engine: "celeritas",
      version: "0.1.0",
      buildVersion: "0.1.0",
      buildSha: "dev",
      compatible: true,
      supportedRange: ">=0.1.0 <0.2.0",
      locks: { healthy: true, stale: [], missing: [] },
      project: path.join(compatible.homeDir, ".celeritas-ui", "project"),
    });
  } finally {
    await incompatible.stop();
    await compatible.stop();
    fs.rmSync(incompatibleBin, { force: true });
    fs.rmSync(compatibleBin, { force: true });
  }
});

test("materialization writes lockfiles and health surfaces lock drift", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-lock-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "lock" ] && [ "$2" = "--check" ] && [ -n "$3" ]; then
  if [ -f "$PWD/.celeritas/locks/extractor/$3.lock.json" ]; then
    echo "ok extractor \`$3\`"
    exit 0
  fi
  echo "stale extractor \`$3\`"
  exit 1
fi
if [ "$1" = "lock" ] && [ "$2" = "--check" ]; then
  for file in "$PWD"/.celeritas/locks/extractor/*.lock.json; do
    if [ -f "$file" ]; then
      name="$(basename "$file" .lock.json)"
      echo "ok extractor \`$name\`"
    fi
  done
  if [ -f "$PWD/celeritas.yml" ] && grep -q "name: source-demo" "$PWD/celeritas.yml" && [ ! -f "$PWD/.celeritas/locks/extractor/source-demo.lock.json" ]; then
    echo "stale extractor \`source-demo\`"
    exit 1
  fi
  exit 0
fi
if [ "$1" = "lock" ]; then
  mkdir -p "$PWD/.celeritas/locks/extractor"
  printf '{"locked":true}\\n' > "$PWD/.celeritas/locks/extractor/$2.lock.json"
  exit 0
fi
if [ "$1" = "discover" ]; then
  echo '{"streams":[]}'
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const saved = await server.fetch("/api/connector-instances", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        driver: "source-demo",
        name: "source",
        params: {},
      }),
    });
    assert.equal(saved.status, 200);
    assert.deepEqual((await saved.json()).lock, {
      status: "untracked",
      message: "connector is not declared in the project manifest",
    });

    const testResponse = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        autoInstall: true,
        driver: "source-demo",
        params: {},
      }),
    });
    assert.equal(testResponse.status, 200);
    const lockPath = path.join(
      server.homeDir,
      ".celeritas-ui",
      "project",
      ".celeritas",
      "locks",
      "extractor",
      "source-demo.lock.json",
    );
    assert.equal(fs.existsSync(lockPath), true);

    const instancesLocked = await server.fetch("/api/connector-instances");
    assert.equal(instancesLocked.status, 200);
    const lockedBody = await instancesLocked.json();
    assert.deepEqual(lockedBody[0].lock, {
      status: "locked",
      message: null,
    });

    fs.rmSync(lockPath, { force: true });
    const health = await server.fetch("/api/health");
    assert.equal(health.status, 200);
    const healthBody = await health.json();
    assert.deepEqual(healthBody.locks, {
      healthy: false,
      stale: [],
      missing: ["source-demo"],
    });
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("run stream emits incremental output and a terminal status event", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-sse-run-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "run" ]; then
  echo "starting run"
  echo "loader warmup" >&2
  sleep 0.1
  echo "Records:"
  echo "  users: 2"
  echo "Total records: 2"
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const source = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "source-postgres-ipc",
          name: "source",
          params: {},
        }),
      })
      .then((res) => res.json());
    const target = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "sink-redis-ipc",
          name: "target",
          params: {},
        }),
      })
      .then((res) => res.json());
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "stream-job",
          definition: {
            sourceConnectorId: source.id,
            targetConnectorId: target.id,
            steps: [{ template: "sink-redis-ipc", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const runId = "run-stream-1";
    const stream = openEventStream(server.port, `/api/runs/${runId}/stream`);
    await stream.ready;
    const runResponsePromise = server.fetch(`/api/jobs/${job.id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ autoInstall: true, runId }),
    });

    const [streamBody, runResponse] = await Promise.all([
      stream.done,
      runResponsePromise,
    ]);
    assert.equal(runResponse.status, 200);
    const runBody = await runResponse.json();
    assert.equal(runBody.run.id, runId);
    assert.match(streamBody, /event: status/);
    assert.match(streamBody, /"state":"started"/);
    assert.match(streamBody, /event: command/);
    assert.match(
      streamBody,
      /celeritas run source-postgres-ipc sink-redis-ipc/,
    );
    assert.match(streamBody, /event: stdout/);
    assert.match(streamBody, /"line":"starting run"/);
    assert.match(streamBody, /"line":"Records:"/);
    assert.match(streamBody, /event: stderr/);
    assert.match(streamBody, /"line":"loader warmup"/);
    assert.match(streamBody, /"state":"complete"/);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("registry install stream emits command output and final status", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-sse-install-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "hub" ] && [ "$2" = "add" ]; then
  echo "installing $3"
  echo "done" >&2
  sleep 0.1
  exit 0
fi
exit 0
`,
  );
  const manifestBody = {
    schema_version: "1",
    registry: { name: "Local" },
    packages: [
      {
        name: "source-demo",
        type: "source",
        artifact: { publisher: "pip", package: "source-demo" },
        manifest: {
          path: "extensions/source/demo/plugin.celeritas.yml",
          capabilities: [],
          settings: [],
        },
      },
    ],
  };
  const manifestServer = await startRegistryManifestServer(manifestBody);
  const server = await startServer({
    CELERITAS_BIN: bin,
    CELERITAS_REGISTRY_URL: manifestServer.url,
    CELERITAS_REGISTRY_PINS: JSON.stringify({
      [manifestServer.url]: manifestChecksum(manifestBody),
    }),
  });
  try {
    const installId = "install-stream-1";
    const stream = openEventStream(
      server.port,
      `/api/registry/install/${installId}/stream`,
    );
    await stream.ready;
    const installResponsePromise = server.fetch("/api/registry/install", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "source-demo", installId }),
    });

    const [streamBody, installResponse] = await Promise.all([
      stream.done,
      installResponsePromise,
    ]);
    assert.equal(installResponse.status, 200);
    const installBody = await installResponse.json();
    assert.equal(installBody.installId, installId);
    assert.equal(installBody.status, "installed");
    assert.match(streamBody, /event: status/);
    assert.match(streamBody, /"state":"started"/);
    assert.match(streamBody, /event: command/);
    assert.match(streamBody, /celeritas hub add source-demo/);
    assert.match(streamBody, /event: stdout/);
    assert.match(streamBody, /"line":"installing source-demo"/);
    assert.match(streamBody, /event: stderr/);
    assert.match(streamBody, /"line":"done"/);
    assert.match(streamBody, /"state":"installed"/);
  } finally {
    await server.stop();
    await manifestServer.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("connectors and registry reads honor ETag/If-None-Match", async () => {
  const manifestServer = await startRegistryManifestServer({
    schema_version: "1",
    registry: { name: "Local" },
    packages: [
      {
        name: "source-demo",
        type: "source",
        artifact: {
          publisher: "cargo",
          package: "source-demo",
          install_command: "",
        },
        manifest: { path: "extensions/source/demo/plugin.celeritas.yml" },
      },
    ],
  });
  const server = await startServer({
    CELERITAS_REGISTRY_URL: manifestServer.url,
  });
  try {
    const connectors = await server.fetch("/api/connectors");
    const connectorsEtag = connectors.headers.get("etag");
    assert.ok(connectorsEtag);
    assert.match(connectors.headers.get("cache-control"), /max-age=60/);
    const connectors304 = await server.fetch("/api/connectors", {
      headers: { "If-None-Match": connectorsEtag },
    });
    assert.equal(connectors304.status, 304);

    const registry = await server.fetch("/api/registry");
    const registryEtag = registry.headers.get("etag");
    assert.ok(registryEtag);
    const registry304 = await server.fetch("/api/registry", {
      headers: { "If-None-Match": registryEtag },
    });
    assert.equal(registry304.status, 304);

    const filtered = await server.fetch("/api/registry?q=missing");
    assert.equal(filtered.status, 200);
    assert.notEqual(filtered.headers.get("etag"), registryEtag);
  } finally {
    await server.stop();
    await manifestServer.stop();
  }
});

function writeStubCeleritas(file, script) {
  fs.writeFileSync(file, script, { mode: 0o755 });
}

function assertRecord(value) {
  assert.equal(typeof value, "object");
  assert.notEqual(value, null);
}

function assertOptionalString(value) {
  assert.ok(
    value === undefined ||
      value === null ||
      typeof value === "string",
    "expected optional string",
  );
}

function assertConnectorSpecShape(value) {
  assertRecord(value);
  assert.equal(typeof value.name, "string");
  assert.equal(typeof value.description, "string");
  assert.ok(value.kind === "source" || value.kind === "target");
  assert.equal(typeof value.driver, "string");
  assert.equal(typeof value.category, "string");
  assert.ok(Array.isArray(value.params));
  assert.ok(Array.isArray(value.jobParams));
  assert.ok(Array.isArray(value.actions));
  assertRecord(value.example);
  assert.equal(typeof value.exampleUnresolved, "boolean");
  assert.equal(typeof value.available, "boolean");
  assertOptionalString(value.icon);
  assertOptionalString(value.path);
  assertOptionalString(value.source);
}

function assertRegistryPackageShape(value) {
  assertRecord(value);
  assert.equal(typeof value.name, "string");
  assert.equal(typeof value.title, "string");
  assert.equal(typeof value.summary, "string");
  assert.equal(typeof value.description, "string");
  assert.ok(value.type === "source" || value.type === "target");
  assert.equal(typeof value.version, "string");
  assertRecord(value.author);
  assert.equal(typeof value.author.name, "string");
  assert.equal(typeof value.repository, "string");
  assert.equal(typeof value.license, "string");
  assertRecord(value.artifact);
  assert.equal(typeof value.artifact.publisher, "string");
  assert.equal(typeof value.artifact.package, "string");
  assert.equal(typeof value.artifact.install_command, "string");
  assertRecord(value.manifest);
  assert.ok(Array.isArray(value.tags));
  assert.equal(typeof value.category, "string");
  assert.equal(typeof value.installed, "boolean");
  assert.equal(typeof value.installCommand, "string");
  assertRecord(value.packagedBy);
  assert.equal(typeof value.packagedBy.name, "string");
  assert.equal(typeof value.packagedAt, "string");
}

function assertHealthShape(value) {
  assertRecord(value);
  assert.equal(value.ok, true);
  assert.ok(value.engine === false || value.engine === "celeritas");
  assert.equal(typeof value.version, "string");
  assert.equal(typeof value.buildVersion, "string");
  assert.equal(typeof value.buildSha, "string");
  assert.equal(typeof value.compatible, "boolean");
  assert.equal(typeof value.supportedRange, "string");
  assertRecord(value.locks);
  assert.equal(typeof value.locks.healthy, "boolean");
  assert.ok(Array.isArray(value.locks.stale));
  assert.ok(Array.isArray(value.locks.missing));
  assert.equal(typeof value.project, "string");
}

function assertJobRunShape(value) {
  assertRecord(value);
  assert.equal(typeof value.id, "string");
  assert.equal(typeof value.jobId, "string");
  assertOptionalString(value.targetId);
  assert.equal(typeof value.status, "string");
  assert.equal(typeof value.trigger, "string");
  assert.equal(typeof value.startedAt, "string");
  assertOptionalString(value.finishedAt);
  if (value.metadata !== undefined) {
    assertRecord(value.metadata);
  }
}

function assertJobRunStepShape(value) {
  assertRecord(value);
  assert.equal(typeof value.runId, "string");
  assert.equal(typeof value.stepIdx, "number");
  assert.equal(typeof value.template, "string");
  assertRecord(value.args);
  assert.equal(typeof value.status, "string");
  assert.ok(
    value.rowCount === undefined ||
      value.rowCount === null ||
      typeof value.rowCount === "number",
  );
  assertOptionalString(value.log);
}

test("HTTP contract exposes typed health, catalog, registry, and run payloads against a stub engine", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-contract-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "celeritas 0.1.5"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "lock" ]; then
  exit 0
fi
if [ "$1" = "select" ]; then
  exit 0
fi
if [ "$1" = "run" ]; then
  echo "Records:"
  echo "  users: 2"
  echo "Total records: 2"
  exit 0
fi
echo "unexpected args: $*" >&2
exit 1
`,
  );
  const manifestServer = await startRegistryManifestServer({
    name: "Contract test registry",
    homepage: "https://registry.example.test",
    packages: [
      {
        name: "source-contract",
        title: "Source Contract",
        summary: "Contract source package",
        description: "A stub package used by the sidecar contract test.",
        type: "source",
        version: "1.2.3",
        author: { name: "Celeritas QA", email: "qa@example.test" },
        repository: "https://example.test/source-contract",
        license: "MIT",
        artifact: {
          publisher: "qa",
          package: "source-contract",
          install_command: "celeritas hub install qa/source-contract",
        },
        manifest: {
          capabilities: ["discover"],
          settings: [],
        },
        tags: ["contract", "test"],
        category: "Files",
        packaged_by: { name: "Celeritas QA", email: "qa@example.test" },
        packaged_at: "2026-01-01T00:00:00.000Z",
        install_command: "celeritas hub install qa/source-contract",
      },
    ],
  });
  const server = await startServer({
    CELERITAS_BIN: bin,
    CELERITAS_REGISTRY_URL: manifestServer.url,
  });
  try {
    const healthResponse = await server.fetch("/api/health");
    assert.equal(healthResponse.status, 200);
    assert.equal(typeof healthResponse.headers.get("x-request-id"), "string");
    const healthBody = await healthResponse.json();
    assertHealthShape(healthBody);
    assert.equal(healthBody.engine, "celeritas");
    assert.equal(healthBody.version, "0.1.5");
    assert.equal(healthBody.compatible, true);

    const connectorsResponse = await server.fetch("/api/connectors");
    assert.equal(connectorsResponse.status, 200);
    const connectorsBody = await connectorsResponse.json();
    assert.ok(Array.isArray(connectorsBody));
    assert.ok(connectorsBody.length > 0);
    assertConnectorSpecShape(connectorsBody[0]);

    const registryResponse = await server.fetch("/api/registry");
    assert.equal(registryResponse.status, 200);
    const registryBody = await registryResponse.json();
    assertRecord(registryBody);
    assertRecord(registryBody.registry);
    assert.ok(Array.isArray(registryBody.packages));
    assert.equal(typeof registryBody.registry.name, "string");
    assert.ok(registryBody.packages.length > 0);
    assertRegistryPackageShape(registryBody.packages[0]);

    const source = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "source-demo",
          name: "source",
          params: {},
        }),
      })
      .then((res) => res.json());
    const target = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "sink-demo",
          name: "target",
          params: {},
        }),
      })
      .then((res) => res.json());
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "contract-job",
          definition: {
            sourceConnectorId: source.id,
            targetConnectorId: target.id,
            steps: [{ template: "sink-demo", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const runResponse = await server.fetch(`/api/jobs/${job.id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ autoInstall: true, fullRefresh: true }),
    });
    assert.equal(runResponse.status, 200);
    const runBody = await runResponse.json();
    assertRecord(runBody);
    assertJobRunShape(runBody.run);
    assert.ok(Array.isArray(runBody.steps));
    assert.ok(runBody.steps.length > 0);
    assertJobRunStepShape(runBody.steps[0]);
    assert.equal(runBody.run.jobId, job.id);
    assert.equal(runBody.steps[0].runId, runBody.run.id);
  } finally {
    await server.stop();
    await manifestServer.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("inline secrets are redacted from connector errors and request logs", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-secret-${process.pid}.sh`);
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "discover" ]; then
  echo "authentication failed for password supersecret" >&2
  exit 1
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const response = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        autoInstall: true,
        driver: "source-postgres-ipc",
        params: { password: "supersecret" },
      }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.status, "fail");
    assert.equal(body.code, "AUTH_FAILED");
    assert.doesNotMatch(body.message, /supersecret/);
    assert.match(body.message, /\[redacted\]/);
    const logs = server.output().stdout;
    assert.doesNotMatch(logs, /supersecret/);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("saved connector instances persist only secret references and dry-run resolves them in-session", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-session-${process.pid}.sh`);
  const envLog = path.join(
    os.tmpdir(),
    `celeritas-session-env-${process.pid}.log`,
  );
  fs.rmSync(envLog, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  printf '%s=%s\\n' "$4" "$5" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "discover" ]; then
  printf '%s' "$SOURCE_POSTGRES_IPC__PASSWORD" > "${envLog}"
  echo '{"streams":[]}'
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const source = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "source-postgres-ipc",
          name: "source",
          params: { password: "supersecret" },
        }),
      })
      .then((res) => res.json());
    const target = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "sink-redis-ipc",
          name: "target",
          params: {},
        }),
      })
      .then((res) => res.json());
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "daily-job",
          definition: {
            sourceConnectorId: source.id,
            targetConnectorId: target.id,
            steps: [{ template: "sink-redis-ipc", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const instancesFile = path.join(
      server.homeDir,
      ".celeritas-ui",
      "project",
      "instances.json",
    );
    const persisted = fs.readFileSync(instancesFile, "utf8");
    assert.doesNotMatch(persisted, /supersecret/);
    assert.match(persisted, /secret-session:\/\//);
    const manifestFile = path.join(
      server.homeDir,
      ".celeritas-ui",
      "project",
      "celeritas.yml",
    );

    const dryRun = await server.fetch(`/api/jobs/${job.id}/dry-run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ autoInstall: true }),
    });
    assert.equal(dryRun.status, 200);
    const body = await dryRun.json();
    assert.equal(body.run.status, "complete");
    const manifest = fs.readFileSync(manifestFile, "utf8");
    assert.doesNotMatch(manifest, /supersecret/);
    assert.match(manifest, /password=secret-session:\/\//);
    assert.equal(fs.readFileSync(envLog, "utf8"), "supersecret");
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(envLog, { force: true });
  }
});

test("connector config is validated against describe schema before discover runs", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-schema-${process.pid}.sh`);
  const argsLog = path.join(
    os.tmpdir(),
    `celeritas-schema-args-${process.pid}.log`,
  );
  fs.rmSync(argsLog, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
echo "$@" >> "${argsLog}"
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "describe" ]; then
  cat <<'JSON'
{"type":"object","properties":{"host":{"type":"string"},"port":{"type":"integer"},"mode":{"type":"string","enum":["full","incremental"]}},"required":["host","port"]}
JSON
  exit 0
fi
if [ "$1" = "discover" ]; then
  echo '{"streams":[]}'
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const response = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        autoInstall: true,
        driver: "source-demo",
        params: {
          mode: "invalid",
        },
      }),
    });
    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), {
      code: "VALIDATION",
      message: "host is required",
      error: "host is required",
      details: [
        { field: "host", message: "host is required" },
        { field: "port", message: "port is required" },
        { field: "mode", message: "mode must be one of full, incremental" },
      ],
      field: "host",
    });
    const args = fs.readFileSync(argsLog, "utf8");
    assert.match(args, /describe source-demo --json-schema/);
    assert.doesNotMatch(args, /discover source-demo --refresh/);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(argsLog, { force: true });
  }
});

test("connector actions require explicit autoInstall before first use", async () => {
  const bin = path.join(
    os.tmpdir(),
    `celeritas-needs-install-${process.pid}.sh`,
  );
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const response = await server.fetch("/api/connectors/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ driver: "source-postgres-ipc" }),
    });
    assert.equal(response.status, 409);
    assert.deepEqual(await response.json(), {
      code: "NEEDS_INSTALL",
      message:
        "connector source-postgres-ipc is not installed; rerun with { autoInstall: true } or install it first",
      error:
        "connector source-postgres-ipc is not installed; rerun with { autoInstall: true } or install it first",
    });
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
  }
});

test("dry-run validates and discovers without invoking load, and inspect returns bounded preview", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-dryrun-${process.pid}.sh`);
  const runMarker = path.join(
    os.tmpdir(),
    `celeritas-dryrun-run-${process.pid}.txt`,
  );
  fs.rmSync(runMarker, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "validate" ]; then
  echo "$PWD/celeritas.yml is valid."
  exit 0
fi
if [ "$1" = "discover" ]; then
  cat <<'JSON'
{"streams":[{"stream":{"name":"users","schema":{"properties":{"id":{"type":"integer"},"name":{"type":"string"}}},"preview":[{"id":1,"name":"Ada"},{"id":2,"name":"Grace"},{"id":3,"name":"Linus"}]}}]}
JSON
  exit 0
fi
if [ "$1" = "run" ]; then
  echo invoked > "${runMarker}"
  exit 1
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const dryRun = await server.fetch("/api/jobs/dry-run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        source: "source-demo",
        steps: [],
        autoInstall: true,
      }),
    });
    assert.equal(dryRun.status, 200);
    const dryRunBody = await dryRun.json();
    assert.equal(dryRunBody.run.status, "complete");
    assert.equal(dryRunBody.steps[0].rowCount, 2);
    assert.equal(dryRunBody.run.metadata.diagnostics.validation.ok, true);
    assert.equal(dryRunBody.run.metadata.diagnostics.schema.preview.length, 3);
    assert.equal(fs.existsSync(runMarker), false);

    const inspect = await server.fetch("/api/connectors/inspect", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        driver: "source-demo",
        autoInstall: true,
      }),
    });
    assert.equal(inspect.status, 200);
    assert.deepEqual(await inspect.json(), {
      columns: [
        { name: "id", dataType: "integer" },
        { name: "name", dataType: "string" },
      ],
      preview: [
        { id: 1, name: "Ada" },
        { id: 2, name: "Grace" },
        { id: 3, name: "Linus" },
      ],
    });
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(runMarker, { force: true });
  }
});

test("schedules validate cron, honor active environment, and reconcile engine list", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-schedule-${process.pid}.sh`);
  const argsLog = path.join(
    os.tmpdir(),
    `celeritas-schedule-args-${process.pid}.log`,
  );
  const schedulesFile = path.join(
    os.tmpdir(),
    `celeritas-schedule-file-${process.pid}.txt`,
  );
  fs.rmSync(argsLog, { force: true });
  fs.rmSync(schedulesFile, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
echo "$@" >> "${argsLog}"
if [ "$1" = "--environment" ]; then
  shift 2
fi
if [ "$1" = "environment" ] && [ "$2" = "list" ]; then
  echo "dev (default)"
  echo "prod"
  exit 0
fi
if [ "$1" = "schedule" ] && [ "$2" = "list" ]; then
  if [ -f "${schedulesFile}" ]; then cat "${schedulesFile}"; else echo "No schedules defined."; fi
  exit 0
fi
if [ "$1" = "schedule" ] && [ "$2" = "add" ]; then
  echo "$3 $5 $7 enabled -" > "${schedulesFile}"
  exit 0
fi
if [ "$1" = "schedule" ] && [ "$2" = "remove" ]; then
  rm -f "${schedulesFile}"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "nightly",
          definition: {
            source: "source-demo",
            steps: [{ template: "sink-demo", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const invalid = await server.fetch(`/api/jobs/${job.id}/schedule`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cron: "bad cron" }),
    });
    assert.equal(invalid.status, 400);

    const environments = await server.fetch("/api/environments");
    assert.equal(environments.status, 200);
    const envBody = await environments.json();
    assert.equal(envBody.environments.length, 2);

    const setActive = await server.fetch("/api/environments/active", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "prod" }),
    });
    assert.equal(setActive.status, 200);

    const added = await server.fetch(`/api/jobs/${job.id}/schedule`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cron: "0 2 * * *" }),
    });
    assert.equal(added.status, 200);

    const schedules = await server.fetch("/api/schedules");
    assert.equal(schedules.status, 200);
    const schedulesBody = await schedules.json();
    assert.equal(schedulesBody[0].jobId, job.id);
    assert.equal(schedulesBody[0].drift, null);

    const logs = fs.readFileSync(argsLog, "utf8");
    assert.match(logs, /--environment prod schedule add/);
    assert.match(logs, /--environment prod schedule list/);

    const removed = await server.fetch(`/api/jobs/${job.id}/schedule`, {
      method: "DELETE",
    });
    assert.equal(removed.status, 200);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(argsLog, { force: true });
    fs.rmSync(schedulesFile, { force: true });
  }
});

test("runs merge engine job history and state bookmarks, and full-refresh is passed through", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-history-${process.pid}.sh`);
  const argsLog = path.join(
    os.tmpdir(),
    `celeritas-history-args-${process.pid}.log`,
  );
  fs.rmSync(argsLog, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
echo "$@" >> "${argsLog}"
if [ "$1" = "--environment" ]; then
  shift 2
fi
if [ "$1" = "job" ] && [ "$2" = "list" ]; then
  echo "success 3s prod:source-demo-to-sink-demo  (prod:source-demo-to-sink-demo-1710000000)"
  exit 0
fi
if [ "$1" = "state" ] && [ "$2" = "list" ]; then
  echo "prod:source-demo-to-sink-demo"
  exit 0
fi
if [ "$1" = "state" ] && [ "$2" = "get" ]; then
  echo '{"bookmarks":{"users":{"updated_at":"2024-01-01T00:00:00Z"}}}'
  exit 0
fi
if [ "$1" = "environment" ] && [ "$2" = "list" ]; then
  echo "prod"
  exit 0
fi
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "run" ]; then
  echo "Records:"
  echo "  users: 1"
  echo "Total records: 1"
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    await server.fetch("/api/environments/active", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: "prod" }),
    });
    const source = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "source-demo",
          name: "source",
          params: {},
        }),
      })
      .then((res) => res.json());
    const target = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "sink-demo",
          name: "target",
          params: {},
        }),
      })
      .then((res) => res.json());
    const job = await server
      .fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: "full-refresh-job",
          definition: {
            sourceConnectorId: source.id,
            targetConnectorId: target.id,
            steps: [{ template: "sink-demo", args: {} }],
          },
        }),
      })
      .then((res) => res.json());

    const run = await server.fetch(`/api/jobs/${job.id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ autoInstall: true, fullRefresh: true }),
    });
    assert.equal(run.status, 200);

    const runs = await server.fetch("/api/runs");
    assert.equal(runs.status, 200);
    const runsBody = await runs.json();
    const engineRun = runsBody.runs.find(
      (entry) => entry.id === "prod:source-demo-to-sink-demo-1710000000",
    );
    assert.equal(
      engineRun.metadata.diagnostics.engineStateId,
      "prod:source-demo-to-sink-demo",
    );
    assert.deepEqual(engineRun.metadata.diagnostics.state, {
      bookmarks: { users: { updated_at: "2024-01-01T00:00:00Z" } },
    });

    const logs = fs.readFileSync(argsLog, "utf8");
    assert.match(
      logs,
      /--environment prod run source-demo sink-demo --full-refresh/,
    );
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(argsLog, { force: true });
  }
});

test("source instance selection rules persist and constrain discover/run behavior", async () => {
  const bin = path.join(os.tmpdir(), `celeritas-select-${process.pid}.sh`);
  const rulesFile = path.join(
    os.tmpdir(),
    `celeritas-select-rules-${process.pid}.txt`,
  );
  const argsLog = path.join(
    os.tmpdir(),
    `celeritas-select-args-${process.pid}.log`,
  );
  fs.rmSync(rulesFile, { force: true });
  fs.rmSync(argsLog, { force: true });
  writeStubCeleritas(
    bin,
    `#!/bin/sh
echo "$@" >> "${argsLog}"
if [ "$1" = "init" ]; then
  mkdir -p "$PWD/project"
  printf 'name: project\\n' > "$PWD/project/celeritas.yml"
  exit 0
fi
if [ "$1" = "add" ]; then
  printf 'name: %s\\n' "$2" >> "$PWD/celeritas.yml"
  exit 0
fi
if [ "$1" = "config" ]; then
  exit 0
fi
if [ "$1" = "select" ] && [ "$3" = "--remove" ]; then
  rm -f "${rulesFile}"
  exit 0
fi
if [ "$1" = "select" ] && [ "$3" != "" ]; then
  rule="$3"
  if [ "$4" != "" ] && [ "$4" != "--exclude" ]; then
    rule="$3.$4"
  else
    rule="$3.*"
  fi
  if [ "$4" = "--exclude" ] || [ "$5" = "--exclude" ]; then
    rule="!$rule"
  fi
  echo "$rule" > "${rulesFile}"
  exit 0
fi
if [ "$1" = "select" ]; then
  if [ -f "${rulesFile}" ]; then
    echo "Selection rules for \`$2\`:"
    while IFS= read -r line; do
      echo "  - $line"
    done < "${rulesFile}"
  else
    echo "No selection rules for \`$2\`."
  fi
  exit 0
fi
if [ "$1" = "validate" ]; then
  echo ok
  exit 0
fi
if [ "$1" = "discover" ]; then
  if [ -f "${rulesFile}" ] && grep -q "users\\.\\*" "${rulesFile}"; then
    cat <<'JSON'
{"streams":[{"stream":{"name":"users","schema":{"properties":{"id":{"type":"integer"}}}}}]}
JSON
  else
    cat <<'JSON'
{"streams":[{"stream":{"name":"users","schema":{"properties":{"id":{"type":"integer"}}}}},{"stream":{"name":"orders","schema":{"properties":{"order_id":{"type":"integer"}}}}}]}
JSON
  fi
  exit 0
fi
exit 0
`,
  );
  const server = await startServer({ CELERITAS_BIN: bin });
  try {
    const source = await server
      .fetch("/api/connector-instances", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          driver: "source-demo",
          name: "source",
          params: {},
          selection: ["users.*"],
        }),
      })
      .then((res) => res.json());
    const dryRun = await server.fetch("/api/jobs/dry-run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        job: {
          id: "job-select",
          name: "dry-run-select",
          definition: {
            sourceConnectorId: source.id,
            steps: [],
          },
        },
        autoInstall: true,
      }),
    });
    assert.equal(dryRun.status, 200);
    const body = await dryRun.json();
    assert.deepEqual(body.run.metadata.diagnostics.schema.columns, [
      { name: "id", dataType: "integer" },
    ]);
    const args = fs.readFileSync(argsLog, "utf8");
    assert.match(args, /select source-demo users/);
  } finally {
    await server.stop();
    fs.rmSync(bin, { force: true });
    fs.rmSync(rulesFile, { force: true });
    fs.rmSync(argsLog, { force: true });
  }
});
