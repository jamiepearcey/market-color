# market-color API server

The single backend the market-color UI talks to: one standalone Node server
(`market-color-server.mjs`) that owns Postgres (config + state), run
orchestration, report generation, the grounded chat, and the read-only graph /
briefs surfaces. CORS-enabled so the static UI (Vite dev, static host, or the
Tauri desktop app) can call it from another origin.

Node ESM. The HTTP layer and the graph/briefs modules use **Node built-ins +
`fetch` only**; Postgres-backed features use `pg`, and chat uses the Anthropic
+ MCP SDKs (all already in `ui/package.json`).

## Start

```bash
npm run api          # from ui/ — node server/market-color-server.mjs
```

Default bind `0.0.0.0:8787`. If `dist/` exists (after `npm run build`) it is
also served statically, so a single process can host the whole app.

## Layout

| File | Role |
| --- | --- |
| `market-color-server.mjs` | HTTP server: `/api/*` (JSON), `/chat` (NDJSON stream), `/chat/health`, `/health`, static `dist/`. |
| `api.mjs` | Pure-ish router: `handleApi(method, pathnameWithQuery, body) -> {status, json}`. |
| `db.mjs` | Postgres pool + migrations (desks, sources, documents, runs, reports). |
| `orchestrator.mjs` | Daily-run pipeline: scrape → decompose facts → index to Qdrant. |
| `reports.mjs` | Grounded report generation (Claude + the facts MCP tools). |
| `chat-core.mjs` | Chat session core: Claude with the Qdrant-facts MCP server (graph-aware tools, grounding scaffold). |
| `graph.mjs` | Transmission map: scrolls `market_facts` from Qdrant, builds the entity co-occurrence + causal graph (60s in-process cache). |
| `briefs.mjs` | Desk briefs: lists/reads `data/briefs/dt=YYYY-MM-DD/<desk>.md` (+ sibling `.json`), path-traversal safe. |
| `celeritas-server.mjs` (+ `core/parser/registry/engine-errors/errors/store-files.mjs`) | Legacy Celeritas ETL sidecar this UI skeleton was ported from; kept only because its test suite still guards the shared helpers. Not part of the market-color app. |

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8787` | Listen port. |
| `HOST` | `0.0.0.0` | Listen host. |
| `MARKET_CORS_ORIGIN` | `*` | `Access-Control-Allow-Origin` value. |
| `DATABASE_URL` | `postgres://localhost:5432/market_color` | Postgres for desks/sources/runs/reports. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint (facts index + graph). |
| `MARKET_FACTS_COLLECTION` | `market_facts` | Qdrant collection of atomic facts. |
| `MARKET_BRIEFS_DIR` | `<repo>/data/briefs` | Root of the rendered desk briefs. |
| `ANTHROPIC_API_KEY` | _(unset)_ | Enables `/chat` and report generation. |
| `MARKET_MCP_COMMAND` / `MARKET_MCP_ARGS` | `uv` + repo defaults | Override how the Qdrant-facts MCP server is spawned. |
| `MARKET_UV` | `uv` | `uv` binary used to spawn pipeline scripts. |
| `MARKET_SINCE_HOURS` | `48` | Scrape window for daily runs. |

The server boots (and answers `/api/health` honestly) even when Postgres,
Qdrant, or the Anthropic key are missing — each feature degrades independently.

## Endpoints

### Meta
- `GET /health` → `{ ok:true }` (liveness)
- `GET /chat/health` → `{ available }` (is chat configured?)
- `GET /api/health` → `{ db, qdrant, anthropic }` (dependency probes)

### Chat
- `POST /chat` `{ prompt, threadId?, model? }` → NDJSON stream of
  `{ type:"event", event }` lines, ending with `{ type:"result", result }`.

### Desks / sources / runs / reports (Postgres)
- `GET/POST /api/desks`, `DELETE /api/desks/:id`
- `GET/POST /api/sources`, `PATCH|DELETE /api/sources/:id`,
  `POST /api/sources/upload` (ad-hoc document → decompose + index run)
- `GET /api/runs`, `GET /api/runs/:id`, `POST /api/runs` `{ date? }` (trigger daily run)
- `GET /api/reports`, `GET /api/reports/:id`, `POST /api/reports` `{ prompt, title?, run_date? }`
  (async generation; client polls)

### Transmission map (Qdrant, no DB needed)
- `GET /api/graph?desk=&since=YYYY-MM-DD&until=YYYY-MM-DD&min_weight=1&max_nodes=150`
  → `{ nodes:[{ id, count }], edges:[{ source, target, weight, causal }] }`
  Built by scrolling the facts collection with a desk / `published_ordinal`
  day-range filter. Nodes are fact entities; undirected co-occurrence edges are
  weighted by shared-fact count; facts carrying `cause_entities`
  (feature-detected) also emit directed `causal:true` edges cause → subject
  entity. Trimmed to the `max_nodes` highest-count entities; co-occurrence
  edges below `min_weight` are dropped. Responses are cached in-process per
  query for 60s.
- `GET /api/graph/entity?name=<entity>&desk=` → `{ entity, facts:[<payload>] }`
  — up to 50 fact payloads mentioning the entity, newest first.

### Desk briefs (filesystem)
- `GET /api/briefs` → `[{ date, desk }]` from `MARKET_BRIEFS_DIR/dt=<date>/<desk>.md`
- `GET /api/briefs/:date/:desk` → `{ date, desk, markdown, data }` where `data`
  is the parsed sibling `<desk>.json` (or `null`). Date/desk are strictly
  validated and the resolved path is confined to the briefs dir — no traversal.

`data/briefs/dt=2026-07-01/energy.{md,json}` in the repo is a hand-written dev
fixture (`"fixture": true` in the json); the brief-generator workstream
replaces it with real output using the same layout.

Errors: non-2xx responses are `{ "error": <message> }`; unknown routes return
`404 { "error": "no route for ..." }`.

## Tests

```bash
node --test 'server/__tests__/*.test.mjs'   # all server tests
node --test server/__tests__/graph.test.mjs server/__tests__/briefs.test.mjs  # new surfaces only
```

`graph.test.mjs` exercises the graph builder against mocked Qdrant scroll
pages (pagination, filters, caching, causal edges); `briefs.test.mjs` covers
listing/reading against a temp `MARKET_BRIEFS_DIR` including traversal
attempts. The remaining suites (`core`, `parser`, `registry`, `engine-errors`,
`store-files`, `server.integration`) belong to the legacy Celeritas sidecar
kept alongside.
