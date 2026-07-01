# Celeritas UI ⇄ Engine sidecar server

A local Node sidecar HTTP server that wires the Celeritas control UI to the
**real** Celeritas engine by shelling out to the prebuilt `celeritas` binary.

It is the "server mode" referenced in `src/lib/api.ts` (Tauri → **server** →
preview). See the contract in [`../docs/WIRING.md`](../docs/WIRING.md) and
[`ADR-0015`](../../docs/decisions/ADR-0015-ui-engine-wiring.md).

Node ESM, **built-ins only** (`node:http`, `node:child_process`, `node:fs`,
`node:path`, `node:os`, `node:url`) — no npm dependencies.

## Start

```bash
npm run server                  # if the package.json `server` script is wired
# or directly:
node server/celeritas-server.mjs
```

Then point the UI at it (default `http://127.0.0.1:8787`) — it serves under
`/api/*`.

The server also exposes `GET /health` as a probe-friendly alias of
`GET /api/health`.

Copy [`server/.env.example`](./.env.example) when you want an explicit local
config file; it enumerates every env var the sidecar currently reads.

## Container

Build from the `ui/` directory so the Docker context includes `server/`:

```bash
docker build -f server/Dockerfile -t celeritas-sidecar .
```

Run it with a writable project volume and, optionally, a mounted Celeritas
runtime root that contains `bin/celeritas`:

```bash
docker run --rm -p 8787:8787 \
  -e CELERITAS_API_TOKEN=replace-me \
  -v celeritas-sidecar-data:/home/node/.celeritas-ui \
  -v /absolute/path/to/celeritas-root:/opt/celeritas:ro \
  celeritas-sidecar
```

Notes:

- The container defaults to `HOST=0.0.0.0`, so bearer auth is enabled by
  default through `CELERITAS_API_TOKEN=change-me`. Override that value in real
  deployments.
- If `/opt/celeritas/bin/celeritas` is not mounted, `GET /health` still
  responds with `engine:false` / `ENGINE_MISSING`, which is enough for a basic
  container liveness check.
- `PORT` and `CELERITAS_BIN` are also wired as Docker build args for operators
  who want a non-default internal port or binary path.

## Environment variables

| Var              | Default                                   | Meaning                                              |
| ---------------- | ----------------------------------------- | ---------------------------------------------------- |
| `PORT`           | `8787`                                     | Listen port.                                         |
| `HOST`           | `127.0.0.1`                                | Listen host. Non-loopback hosts require token auth.  |
| `CELERITAS_ROOT` | two levels up from `ui/` (the repo root)  | Celeritas repo root (binary + catalog live under it).|
| `CELERITAS_BIN`  | `<CELERITAS_ROOT>/bin/celeritas`          | Path to the engine binary.                           |
| `CELERITAS_CORES`| _(unset)_                                  | If set, pins `/api/engine` `cores` (envOverride).    |
| `CELERITAS_API_TOKEN` | _(unset)_                             | Required bearer token when set; mandatory for non-loopback binds. |
| `CORS_ORIGINS`   | loopback UI defaults                       | Comma-separated origin allowlist override.            |
| `LOG_LEVEL`      | `info`                                     | Sidecar log level: `debug`, `info`, `warn`, `error`. |
| `MAX_BODY_BYTES` | `1048576`                                  | Max JSON request body size before the server returns `413`. |
| `REGISTRY_FETCH_TIMEOUT_MS` | `10000`                         | Per-attempt registry fetch timeout before retry/fail. |
| `REGISTRY_CACHE_TTL_MS` | `60000`                              | Fresh-registry cache TTL before refetch.              |
| `CELERITAS_BUILD_VERSION` | `ui/package.json` version        | Build version surfaced in startup logs and `/health`. |
| `CELERITAS_BUILD_SHA` | `dev`                                   | Build git SHA surfaced in startup logs and `/health`. |
| `LOG_FORMAT`      | `pretty`                                   | `pretty` for local dev, `json` for machine-parsed logs. |
| `RUN_HISTORY_LIMIT` | `200`                                   | Maximum retained run-history entries. |
| `RATE_LIMIT_WINDOW_MS` | `60000`                              | Sliding window for install/run rate limits. |
| `INSTALL_RATE_LIMIT_COUNT` | `3`                             | Max install attempts per window. |
| `RUN_RATE_LIMIT_COUNT` | `10`                                 | Max run/dry-run attempts per window. |
| `INSTALL_MAX_INFLIGHT` | `1`                                  | Concurrent install cap. |
| `RUN_MAX_INFLIGHT` | `2`                                      | Concurrent run/dry-run cap. |
| `FS_ALLOWED_ROOTS` | `$HOME,<project-parent>`                 | Comma-separated allowlist for file browsing and file-like params. |
| `CELERITAS_REGISTRY_PINS` | _(unset)_                         | JSON map of trusted registry URLs to `sha256:` manifest digests. |
| `CELERITAS_ERROR_REPORT_WEBHOOK` | _(unset)_                  | Optional webhook for unhandled server error reports. |

Default allowed browser origins are `http://127.0.0.1:4500`,
`http://127.0.0.1:5173`, and `http://localhost:5173`. Set
`CORS_ORIGINS=origin1,origin2` to override that allowlist.

Every response includes an `x-request-id` header. At `LOG_LEVEL=info` the
server emits one structured JSON line per request with the request id, method,
path, status, duration, and any `celeritas` command(s) run for that request.
At `LOG_LEVEL=warn`, those request-complete lines are suppressed.

Malformed JSON request bodies return HTTP `400 { "error": ... }`; oversized
request bodies return HTTP `413 { "error": ... }`. Non-2xx responses use the
shape `{ code, message, error, details?, field? }`.

`SIGINT` and `SIGTERM` trigger a graceful shutdown: the server stops accepting
new work, lets in-flight requests finish for up to 10 seconds, then exits
cleanly. Starting a second server on the same port prints a friendly
port-in-use message and exits with code 1.

When `CELERITAS_API_TOKEN` is set, every `/api/*` route except `/api/health`
requires `Authorization: Bearer <token>`. The server refuses to bind to any
non-loopback host unless that token is configured.

Engine-mutating operations are serialized per project directory. The sidecar
queues concurrent run/install-style requests rather than returning `409`, so
engine invocations against `~/.celeritas-ui/project` do not interleave.
Long-running runs and registry installs can also stream incremental
stdout/stderr over Server-Sent Events (SSE) when the caller supplies a stable
`runId` or `installId`.
First use no longer auto-installs connectors implicitly: run/test/action
requests must opt in with `autoInstall: true`, otherwise the sidecar returns
`409 NEEDS_INSTALL`.

Connector and package identifiers are validated against
`^[a-z0-9][a-z0-9._/-]*$` before they are used in spawned `celeritas`
commands. Malicious names are rejected with `400 { code:"VALIDATION", error, field }`.

Secret handling:
- Inline secret values are never persisted directly in `instances.json`.
- When a connector instance is saved with an inline secret, the sidecar stores a
  session-scoped `secret-session://...` reference on disk and keeps the actual
  secret only in the sidecar's volatile in-memory map.
- When connector config is materialized into `celeritas.yml`, secret settings are
  also written as `secret-session://...` refs in the manifest config block
  rather than plaintext values.
- Runs/tests in the same sidecar session resolve those refs back to the real
  value before invoking `celeritas`, injecting them through the engine's normal
  env-precedence path (`process env` / `.env` over manifest config).
- After a sidecar restart, unresolved `secret-session://...` refs must be
  re-entered. Decision: the sidecar does not depend on an OS keychain yet.
- The desktop Tauri shell is the durable secret-custody path: it stores
  connector secrets and secrets-backend tokens in the OS keychain and persists
  only references on disk.

Registry install trust model:
- Installs are allowed only from trusted registry URLs: `CELERITAS_REGISTRY_URL`,
  the saved `registry-urls` pref list, or the built-in local/prod defaults.
- Each trusted registry must also have a manifest integrity pin
  (`CELERITAS_REGISTRY_PINS` or the saved `registry-pins` pref) matching the
  fetched `sha256:` digest.
- Publishers are allowlisted to `cargo`, `pip`, `npm`, and `local`.
- The sidecar reconstructs a safe argv install plan from structured manifest
  fields instead of executing the raw manifest `install_command` through a
  shell.

Registry fetch behavior:
- Registry reads use a bounded timeout with one retry and a small backoff.
- Manifest payloads are schema-checked (`schema_version`, `registry`, `packages[]`)
  before they are trusted.
- If a refetch fails after a previously cached success, the sidecar serves the
  stale cached manifest instead of failing closed for read-only browsing.

## Project directory

The sidecar owns a working Celeritas project at **`~/.celeritas-ui/project`**
(`celeritas init project` is run once, idempotently, on first need). All CLI
calls run with `cwd` = that project dir. Connector instances/jobs are
materialized into it (`celeritas add` + `config set`) before `run`/`schedule`.

Persisted JSON stores in the project dir:

| File              | Contents                                  |
| ----------------- | ----------------------------------------- |
| `instances.json`  | `ConnectorInstanceRecord[]`               |
| `targets.json`    | user `Target[]` (the `embedded` target is synthetic) |
| `jobs.json`       | `Job[]`                                    |
| `runs.json`       | `JobRun[]` (newest first)                  |
| `run-steps.json`  | `{ [runId]: JobRunStep[] }`               |
| `prefs.json`      | UI prefs key→value                        |

Each managed store file is written atomically as a versioned envelope
`{ "version": 1, "data": ... }` using temp-file + `fsync` + rename. If a store
file is corrupt on read, the sidecar quarantines it to `*.bak`, recreates a
fresh version-1 store with the fallback value, and keeps serving requests.

## Catalog source

`GET /api/connectors` is built from the pre-generated authoritative catalog at
`<CELERITAS_ROOT>/site/src/data/connectors.json` (settings + `scope` + category
+ secret/enum metadata). Setting `scope: connection` → `params`, `scope: run` →
`jobParams`. If that file is missing, the server falls back to live
`celeritas hub list` + each connector's `plugin.celeritas.yml`. Connector
**actions** (test/inspect/run) always hit the real binary.

## Endpoints (base `/api`, JSON in/out)

### Health / engine
- `GET  /health` → `{ ok, engine:"celeritas", version, buildVersion, buildSha, compatible, supportedRange, locks, project }`
- `GET  /api/health` → same payload
  (responds even before init; `compatible:false` warns when the binary falls
  outside the sidecar's supported CLI range; `locks` summarizes `celeritas lock --check`
  drift as `{ healthy, stale[], missing[] }`)
- `GET  /engine` → `EmbeddedConfig` `{ cores, maxCores, running:1, envOverride }`

### Connectors (catalog + actions)
- `GET  /connectors` → `ConnectorSpec[]` (sources + sinks, with `ETag` + `Cache-Control`)
- `POST /connectors/test` `{ spec, driver, kind, params, jobParams, selection? }` → `ConnectorTestResult`
  (source → `discover`, sink → `test`; honest pass/fail from exit code; set
  `autoInstall:true` to allow first-use install/build; connector config is
  preflight-validated against `describe --json-schema` when available)
- `POST /connectors/inspect` `{ spec, driver, params, jobParams, selection? }` → `SourceSchema`
  (from the `discover` catalog; set `autoInstall:true` to allow first-use install/build)
- `POST /connectors/action` `{ spec, driver, kind, params, jobParams, action, selection? }` → `ConnectorActionResult`

### Connector instances (persisted)
- `GET    /connector-instances` → `ConnectorInstanceRecord[]`
- `POST   /connector-instances` (body `ConnectorInstanceRecord`) → saved record
  (`selection?: string[]` persists `celeritas select` include/exclude rules for sources)
  and each record carries `lock` status from `celeritas lock --check`
- `DELETE /connector-instances/:id` → `{ ok:true }`

### Targets (persisted; always includes the `embedded` target)
- `GET    /targets` → `Target[]`
- `POST   /targets` (body `Target`) → saved
  Decision: server mode is explicitly gated to the embedded local engine for
  now. Attempts to save a `remote` target return `501 CONFIG_INVALID` with a
  clear "coming soon" message instead of shipping a half-working remote path.
- `DELETE /targets/:id` → `{ ok:true }`

### Jobs (persisted)
- `GET    /jobs` → `Job[]`
- `POST   /jobs` (body `Job`) → saved
- `DELETE /jobs/:id` → `{ ok:true }`

### Job execution
- `POST /jobs/:id/run` → `{ run: JobRun, steps: JobRunStep[] }`
  (materialize source+sink, `celeritas run <source> <sink>`, parse `Records:` /
  `Total records:` into one step per stream; set `autoInstall:true` to allow
  first-use connector install/build; saved source-instance `selection` rules are
  synchronized into the project before discover/run)
- `POST /jobs/:id/dry-run` → same shape (discover/validate only, no load)
- `POST /jobs/dry-run` `{ source, steps, autoInstall?, fullRefresh? }` → same
  shape for an unsaved draft job (validate + discover only, no load)
- `POST /jobs/describe-steps` (body `{ job }`) → `JobRunStep[]` (status `pending`)
- `GET /runs/:id/stream` → `text/event-stream` of `command`, `stdout`, `stderr`,
  and terminal `status` events for a run started with body `{ runId: "<id>" }`

### Runs (persisted)
- `GET /runs` → `JobRun[]` (newest first; merges sidecar runs with engine job/state history)
- `GET /runs/:id/steps` → `JobRunStep[]`

### Schedules
- `GET    /schedules` → reconciled engine/UI schedule view with drift markers
- `POST   /jobs/:id/schedule` `{ cron }` → validated `celeritas schedule add` (+ persist on the job)
- `DELETE /jobs/:id/schedule` → `celeritas schedule remove`

### Environments
- `GET /environments` → `{ active, environments:[{ name, default, active }] }`
- `PUT /environments/active` `{ name|null }` → persist the selected active environment overlay

### Prefs / secrets / files
- `GET /prefs/:key` → `{ value }`; `PUT /prefs/:key` `{ value }` → `{ ok:true }`
- `POST /secrets-keeper/test` `{ url, token? }` → `{ status, message }` (HTTP probe)
- `GET /fs/list?path=` → `{ entries:[{name,path,dir}], base }` (in-app file browser)

### Cache semantics
- `GET /registry` also returns `ETag` + `Cache-Control`.
- Repeating either `GET /connectors` or `GET /registry` with a matching
  `If-None-Match` returns `304 Not Modified`.
- `GET /registry/install/:id/stream` returns the same SSE event types for
  `POST /registry/install` requests started with body `{ installId: "<id>" }`.

## Error codes

The sidecar emits stable machine-readable `code` values on non-2xx responses:

- `AUTH_REQUIRED`: missing or wrong bearer token.
- `AUTH_FAILED`: connector/engine authentication failed.
- `CONFIG_INVALID`: connector or engine configuration failed validation.
- `CONNECTIVITY`: the engine reached a network/connectivity failure.
- `CONNECTOR_NOT_FOUND`: referenced connector/plugin name is unknown.
- `ENGINE_FAILED`: the engine process failed or a CLI-backed operation could not complete.
- `ENGINE_MISSING`: `CELERITAS_BIN` does not exist or cannot be spawned.
- `NEEDS_INSTALL`: the connector exists but must be installed before use.
- `NOT_FOUND`: unknown route or missing persisted resource like a job.
- `ORIGIN_FORBIDDEN`: mutating request from a disallowed browser origin.
- `REGISTRY_FETCH_FAILED`: registry manifest fetch/network failure.
- `REGISTRY_PACKAGE_NOT_FOUND`: requested registry package name is absent.
- `REGISTRY_SCHEMA`: fetched registry manifest failed schema checks.
- `REGISTRY_UNTRUSTED`: install refused because the registry URL or publisher is not trusted.
- `SHUTTING_DOWN`: the sidecar is draining in-flight requests for shutdown.
- `TIMEOUT`: the engine or a registry-related subprocess exceeded its timeout.
- `VALIDATION`: malformed JSON, invalid request body, or bad identifiers/fields.

## Honesty / error handling

- Every handler is try/caught; the process never crashes. Non-2xx responses are
  `{ code, message, error, details?, field? }`.
- CLI failures are surfaced honestly: a source that can't connect returns
  `{ status:"fail", message:<stderr tail> }`; a run that exits non-zero is
  `status:"failed"` with the stderr tail as the step/run log. Nothing is faked.
- `/api/health` reports `engine:false` plus `code:"ENGINE_MISSING"` when the
  binary is absent, and other CLI-backed routes return typed `ENGINE_MISSING`
  or `TIMEOUT` errors instead of raw spawn traces.
- Dry-run now performs real engine validation plus discovery without loading,
  inspect returns merged discover columns plus bounded preview rows when the
  catalog includes them, and `GET /runs` folds in `celeritas job list` /
  `celeritas state` history alongside sidecar-recorded runs.
- First-time `run`/`test` of a native connector may be slow (the engine builds it
  from source via `cargo install --path`), so `run`/`el`/`test`/`discover` use a
  600s timeout; other CLI calls default to 60s.
