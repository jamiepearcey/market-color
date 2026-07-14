# market-color UI

The client for **market-color**: a chat-first market-intelligence app that surfaces the
*current news drivers* behind desk moves, grounded in real article text indexed as atomic
facts in Qdrant. See the repo root [`README.md`](../README.md) for the pipeline and
[`docs/decisions.md`](../docs/decisions.md) (D15) for how the facts + graph are wired in.

React 18 + Vite 6 + TypeScript + Tailwind v3 single-page app. It is a pure static client;
everything it shows is served by the standalone API server in [`server/`](./server/README.md).

## Views (left icon rail)

- **Chat** — grounded LLM chat (`/chat`, NDJSON stream). Claude answers using the
  Qdrant-facts MCP tools and cites its claims; retrieved facts render as cards linking to
  the source articles.
- **Reports** — generate + read grounded reports written from the indexed facts
  (automatically after each daily run, or ad-hoc from a prompt).
- **Desk Briefs** — read the per-desk daily briefs rendered to
  `data/briefs/dt=YYYY-MM-DD/<desk>.md` (+ optional `<desk>.json`): the markdown note,
  a red/green movers table, and driver sections with citations.
- **Transmission Map** — force-directed graph of fact entities (`/api/graph`): node size ∝
  fact count, edge width ∝ shared-fact co-occurrence, amber directed arrows are causal
  links (emitted once facts carry `cause_entities`). Click a node to read that entity's
  facts with source links.
- **Daily Runs** — trigger/inspect the scrape → decompose → index pipeline runs.
- **Sources** — manage RSS sources and upload ad-hoc documents for ingestion.
- **Desks** — manage the desk taxonomy used to classify and steer retrieval.
- **Settings** — API base URL for static/Tauri deploys pointing at a remote API
  (`src/lib/api-base.ts`).

## Develop

```bash
npm install
npm run api    # standalone API server on :8787
npm run dev    # Vite dev server on :1420, proxies /api + /chat to :8787
```

Backing services by feature (everything else still renders, with honest offline states):

- **Transmission map**: Qdrant only — `QDRANT_URL` (default `http://localhost:6333`),
  collection `MARKET_FACTS_COLLECTION` (default `market_facts`).
- **Desk briefs**: files under `MARKET_BRIEFS_DIR` (default `<repo>/data/briefs`).
- **Desks / sources / runs / reports**: Postgres — `DATABASE_URL`
  (default `postgres://localhost:5432/market_color`).
- **Chat**: `ANTHROPIC_API_KEY` (plus Qdrant for retrieval).

## Build, test, lint

```bash
npm run build                  # tsc --noEmit && vite build → dist/
npm run test                   # vitest: src/**/*.test.{ts,tsx} (jsdom)
node --test 'server/__tests__/*.test.mjs'  # server tests (node:test)
npm run lint                   # eslint
npm run typecheck              # tsc --noEmit only
```

Note: `e2e/` (Playwright) still contains the legacy Celeritas smoke/a11y specs from the UI
skeleton this app was ported from — they target views that no longer exist and need a
rewrite before they are useful here.

## Deploy shapes

- **Single origin** — the API server also serves `dist/`, so one process does everything.
- **Static + remote API** — host `dist/` anywhere; point it at the API in Settings or via
  the `VITE_API_BASE` build env.
- **Desktop (Tauri)** — `src-tauri/` wraps the web build as a "Market Color" desktop app;
  bundling needs the Rust toolchain + Tauri CLI (the web build does not). Signing notes
  live in [`src-tauri/SIGNING.md`](./src-tauri/SIGNING.md).
