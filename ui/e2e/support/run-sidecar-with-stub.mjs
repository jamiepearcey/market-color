import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { once } from "node:events";
import { spawn } from "node:child_process";

const uiDir = path.resolve(new URL("../..", import.meta.url).pathname);
const serverEntry = path.join(uiDir, "server", "celeritas-server.mjs");
const apiPort = Number(process.argv[process.argv.indexOf("--port") + 1] || 8787);
const manifestPort = apiPort + 1;
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "celeritas-e2e-"));
const stubBin = path.join(tmpDir, "celeritas-stub.sh");
const homeDir = path.join(tmpDir, "home");
const manifestUrl = `http://127.0.0.1:${manifestPort}/registry.manifest.json`;

fs.mkdirSync(homeDir, { recursive: true });
fs.writeFileSync(
  stubBin,
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
  { mode: 0o755 },
);

const manifestServer = http.createServer((_req, res) => {
  res.statusCode = 200;
  res.setHeader("content-type", "application/json");
  res.end(
    JSON.stringify({
      name: "Celeritas E2E Registry",
      homepage: "https://registry.example.test",
      packages: [
        {
          name: "smoke-source-package",
          title: "Smoke Source Package",
          summary: "Smoke-test source package",
          description: "A deterministic package exposed only for the Playwright smoke test.",
          type: "source",
          version: "1.0.0",
          author: { name: "Celeritas QA", email: "qa@example.test" },
          repository: "https://example.test/smoke-source-package",
          license: "MIT",
          artifact: {
            publisher: "qa",
            package: "smoke-source-package",
            install_command: "celeritas hub install qa/smoke-source-package",
          },
          manifest: {
            capabilities: ["discover"],
            settings: [],
          },
          tags: ["smoke", "test"],
          category: "Files",
          packaged_by: { name: "Celeritas QA", email: "qa@example.test" },
          packaged_at: "2026-01-01T00:00:00.000Z",
          install_command: "celeritas hub install qa/smoke-source-package",
        },
      ],
    }),
  );
});
manifestServer.listen(manifestPort, "127.0.0.1");
await once(manifestServer, "listening");

const sidecar = spawn(process.execPath, [serverEntry], {
  cwd: uiDir,
  env: {
    ...process.env,
    HOME: homeDir,
    HOST: "127.0.0.1",
    PORT: String(apiPort),
    LOG_LEVEL: "warn",
    CELERITAS_BIN: stubBin,
    CELERITAS_REGISTRY_URL: manifestUrl,
  },
  stdio: "inherit",
});

async function shutdown(signal) {
  if (sidecar.exitCode === null) {
    sidecar.kill(signal);
    await once(sidecar, "exit").catch(() => {});
  }
  await new Promise((resolve) => manifestServer.close(resolve));
  fs.rmSync(tmpDir, { force: true, recursive: true });
  process.exit(0);
}

process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("SIGTERM", () => void shutdown("SIGTERM"));

sidecar.on("exit", async (code) => {
  await new Promise((resolve) => manifestServer.close(resolve));
  fs.rmSync(tmpDir, { force: true, recursive: true });
  process.exit(code ?? 0);
});
