# Celeritas UI ⇄ Engine wiring contract

Authoritative contract for the sidecar server and the `api.ts` client. See
ADR-0015. Engine binary: `ETL/celeritas/bin/celeritas` (relative to the celeritas
repo root, which the server resolves at startup; override with `CELERITAS_BIN`).

## Modes (in `src/lib/api.ts`)

Per call: **Tauri** (`isDesktop()`) → **server** (if `serverBase()` health-OK) →
**preview mock** (existing behavior). Add `src/lib/server.ts`:
- `SERVER_BASE = import.meta.env.VITE_CELERITAS_API ?? "http://127.0.0.1:8787"`
- `serverReady()` — cached probe of `GET {SERVER_BASE}/api/health` (ok within ~800ms)
- `srv<T>(path, init?)` — fetch helper returning JSON, throws on non-2xx
- export `engineMode(): "desktop" | "server" | "preview"` for a status indicator

Only the ETL functions below switch to server mode. Functions for deleted donor
surfaces (packs, notebooks, clusters, bindings, staged, templates, quant
functions, aiChat) keep their current preview/no-op behavior.

## Types

All response bodies match the existing TS interfaces in `src/lib/types.ts`:
`ConnectorSpec`, `ConnectorParam`, `ConnectorInstanceRecord`, `ConnectorTestResult`,
`ConnectorActionResult`, `SourceSchema`, `Target`, `Job`, `JobRun`, `JobRunStep`,
`EmbeddedConfig`. Do not invent new shapes — map engine output INTO these.

## Endpoints (base `/api`, JSON in/out, permissive CORS for the vite origin)

### Health / engine
- `GET /health` → `{ ok:true, engine:"celeritas", version:string, project:string }`
- `GET /engine` → `EmbeddedConfig` `{ cores, maxCores, running, envOverride }` (local host info; `running:1`)

### Connectors (catalog) — `listConnectors`
- `GET /connectors` → `ConnectorSpec[]` (sources AND sinks).
  Build from `celeritas hub list` (name, role: extractor→`source`, loader→`target`,
  and the `local:<path>` to the extension). For each, read the extension's
  `plugin.celeritas.yml` (path from hub list) to get settings; split by `scope`:
  `connection` → `params`, `run` → `jobParams`. Map setting `kind`
  (string|integer|number|boolean|secret|enum) → `ConnectorParam.kind`
  (string|number|number|bool|secret|enum); `secret`→`sensitive:true`; `enum`→`options`.
  Set `kind` (source/target), `driver`=name, `category` from path segment
  (databases|files|object-storage|apis|finance|cache), `available:true`, `actions:[]`
  (sources may include a `{name:"preview",label:"Preview",flag:"--discover"}`).
  (You MAY instead read `ETL/celeritas/site/src/data/connectors.json` which already
  has settings+scope+category — either source of truth is acceptable; prefer whichever
  is most robust, but the catalog must list all first-party connectors.)

### Connector test / discover / inspect
- `POST /connectors/test` body `{ spec, driver, kind, params, jobParams }`
  → `ConnectorTestResult { status:"pass"|"fail", message, elapsedMs }`.
  Materialize the connector into the project (`celeritas add` if needed +
  `config set` the params), then: source → `celeritas discover <name>` (success ⇒
  connects); sink → `celeritas test <name>`. Capture stderr into `message` on fail.
- `POST /connectors/inspect` body `{ spec, driver, params, jobParams }`
  → `SourceSchema { columns:[{name,dataType}], preview:[...] }` from
  `celeritas discover <name>` catalog (+ a small sample if available).
- `POST /connectors/action` body `{ spec, driver, kind, params, jobParams, action }`
  → `ConnectorActionResult { status:"complete"|"failed", message, rows, elapsedMs }`.

### Connector instances (persisted in project store) — `saveConnectorInstanceRecord`, `deleteConnectorInstanceRecord`, instances list
- `GET /connector-instances` → `ConnectorInstanceRecord[]`
- `POST /connector-instances` body `ConnectorInstanceRecord` → saved record
- `DELETE /connector-instances/:id` → `{ ok:true }`

### Targets (engine endpoints) — `listTargets`, `saveTarget`, `removeTarget`
- `GET /targets` → `Target[]` (always include an `embedded` local target
  `{ id:"embedded", name:"Local engine", kind:"embedded", url:"" }`)
- `POST /targets` body `Target` → saved; `DELETE /targets/:id` → `{ ok:true }`

### Jobs — `listJobs`, `saveJob`, `deleteJobRemote`
- `GET /jobs` → `Job[]` (persisted in `jobs.json` in the project)
- `POST /jobs` body `Job` → saved `Job`. A Job binds a **source** + a **sink** (+
  optional `definition.schedule` cron). Map: `definition.source` = source spec/instance
  name; reuse `definition.steps[0].template` = sink spec/instance name (the UI's
  existing job model). Persist faithfully.
- `DELETE /jobs/:id` → `{ ok:true }`

### Job execution — `runJob`, `dryRunJob`, `describeSteps`
- `POST /jobs/:id/run` → `{ run: JobRun, steps: JobRunStep[] }`. Ensure source+sink
  are added+configured in the project, then `celeritas run <source> <sink>` (or `el`).
  Parse the per-stream "Records:" summary + total → `JobRun { id, jobId, status:
  "complete"|"failed", trigger:"manual", startedAt, finishedAt }` and one
  `JobRunStep` per stream `{ runId, stepIdx, template, args, status, rowCount, log }`.
  Persist the run. Use a generous timeout (native connectors may build on first run).
- `POST /jobs/:id/dry-run` → same shape; use discovery/validation only (no load).
- `POST /jobs/describe-steps` body `{ job }` → `JobRunStep[]` (planned steps, status "pending").

### Runs — `listRuns`, `listRunSteps`
- `GET /runs` → `JobRun[]` (persisted runs, newest first; may enrich with `celeritas job list`)
- `GET /runs/:id/steps` → `JobRunStep[]`

### Schedules — schedule add/list (Scheduled Jobs)
- `GET /schedules` → array from `celeritas schedule list`
- `POST /jobs/:id/schedule` body `{ cron }` → `celeritas schedule add` for the job; returns `{ ok:true }`
- `DELETE /jobs/:id/schedule` → `celeritas schedule remove`; `{ ok:true }`

### Prefs / secrets / dialogs (local, simple)
- `GET /prefs/:key` → `{ value }`; `PUT /prefs/:key` body `{ value }` → `{ ok:true }` (persist `prefs.json`)
- `POST /secrets-keeper/test` `{ url, token? }` → `{ status, message }` (probe URL; honest result)
- File pickers (`pickDataFile`/`pickDirectory`/`listDataTree`): server cannot open a
  native dialog. `GET /fs/list?path=` → `{ entries:[{name,path,dir}] }` for a simple
  in-app browser; `pickDataFile`/`pickDirectory` in server mode resolve to a returned
  path string or fall back to the existing preview behavior. Low priority.

## Server implementation notes
- Node ESM, Node built-in `http` + `child_process` (no new npm deps). Port 8787
  (env `PORT`). `npm run server` script (add to package.json `scripts` — Foundation
  owns package.json; the client agent may add the script).
- Resolve celeritas repo root = two levels up from `ui/` (`../../`), binary at
  `<root>/bin/celeritas` unless `CELERITAS_BIN` set. Project dir `~/.celeritas-ui/project`.
- Run every CLI call with `cwd` = project dir; capture stdout+stderr+exit code;
  map non-zero to a real failure with the stderr tail as `message`.
- `GET /health` must respond even before the project is initialized.
- Self-test with curl before reporting: `/health`, `/connectors` (lists connectors),
  and one `/connectors/inspect` or `/connectors/test` against `source-csv-arrow` or
  `source-filesystem-csv-ipc`.

## Acceptance
- `node server/celeritas-server.mjs` serves `/api/health` ok and `/api/connectors`
  returns the real first-party connectors (sources + sinks).
- With the server running, the web build (vite) shows engine mode = "server", the
  Sources/Targets catalog is the REAL connectors, and a job run hits the engine.
- `npm run build` (tsc + vite) stays green.

---

## Registry / Store (ADR-0016) — additional endpoints

Lets the UI look up a registry manifest from online, search additional sources &
targets, view full package details, and install. The registry manifest shape is
the ADR-0013 publication format (`registry.manifest.json`): per package
`name, title, summary, description, type(source|target), version,
author{name,email,url}, repository, license, artifact{publisher,package,install_command},
manifest{capabilities,settings}, tags, category` (+ optional `publishedAt`).

Default registry URL resolution (server): `CELERITAS_REGISTRY_URL` env → the
`registry-urls` pref (array) → fallback to the local website
`http://127.0.0.1:4321/registry.manifest.json` then prod
`https://celeritas.dev/registry.manifest.json`. The manifest is fetched over HTTP
and cached (~60s); the server cross-references `celeritas hub list` to mark which
packages are already installed locally.

### Endpoints (base `/api`)
- `GET /registry?url=<optional>` → `{ registry:{name,homepage,...}, packages: RegistryPackage[] }`
  where `RegistryPackage` = the normalized publication record PLUS:
  `installed:boolean` (in local `hub list`), `installCommand:string`
  (`artifact.install_command` or `celeritas add <type> <name>`),
  `packagedBy:{name,email,url}` (=author), `packagedAt:string` (publishedAt or
  registry date). Sources AND targets; supports `?type=source|target`, `?q=<text>`,
  `?category=<cat>` filters (filtering may also be client-side).
- `GET /registry/package?name=<name>&url=<optional>` → a single `RegistryPackage`
  with full detail (all fields above + settings schema + capabilities + tags).
- `POST /registry/install` body `{ name, url? }` → `{ status:"installed"|"failed",
  message, installCommand, log }`. Runs the package's `install_command` (cargo/pip/
  npm/local) within the project, or for `publisher:"pip"` does `celeritas hub add`
  (+ make it resolvable). Honest status; capture stdout/stderr tail into `log`.
  Generous timeout (cargo `--path` builds from source). After success the connector
  must appear in `GET /connectors`.
- `GET /registry/urls` → `string[]` (configured registry URLs);
  `PUT /registry/urls` body `{ urls:string[] }` → `{ ok:true }` (persist `prefs.json`).

### Client (`api.ts` + `server.ts`) — server-mode functions
- `listRegistry(url?)`, `getRegistryPackage(name,url?)`, `installRegistryPackage(name,url?)`,
  `getRegistryUrls()`, `setRegistryUrls(urls)` — all server-mode (no Tauri/preview
  variants required; in preview, return the bundled `registry.manifest.json`-style
  seed so the Store still renders offline).
