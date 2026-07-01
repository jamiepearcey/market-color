# ADR 0001 — Market Color application layer (API + static client, Postgres, fact RAG)

Status: accepted · 2026-07-01

## Context

The corpus layer (see the root `README.md`) crawls financial news into a dated
parquet corpus. We want a product on top of it: chat with the corpus, run a
daily pipeline, manage desks/sources, upload ad-hoc reports, and generate
grounded reports — usable from a desktop app and a browser against one shared
backend.

## Decision

**One API server, many static clients.**

- A single **Node API server** (`ui/server/market-color-server.mjs`) owns all
  state and compute: **Postgres** (config + state), the run **orchestrator**,
  **report** generation, and the **chat**/retrieval loop. It is CORS-enabled and
  can run on a separate host. Endpoints: `/api/*` (JSON), `/chat` (NDJSON stream),
  `/chat/health`, `/health`, and optional static hosting of the UI build.
- The **UI is a pure static build** (Vite → `dist/`) with a **configurable API
  base URL** (`src/lib/api-base.ts`). It is bundled into a thin **Tauri** shell
  for desktop and can equally run in a browser; both point at the same API.

**Retrieval is fact-level RAG over Qdrant, via MCP.**

- Articles are **decomposed into atomic, source-attributed facts**
  (`decompose_facts.py`, Claude structured output) and indexed into the
  `market_facts` Qdrant collection (`index_facts.py`), reusing the corpus
  indexer's embedding + TurboQuant config so vectors are interchangeable.
- The LLM (Claude `claude-opus-4-8`, adaptive thinking) reaches those facts
  through an **MCP** tool `search_market_facts` served by
  `mcp/qdrant_facts_server.py`. The API is the MCP client. Chat and reports share
  the same agentic loop (`chat-core.mjs`: `runChat` / `runOnce`).

**Postgres is the single source of truth for configuration.**

- Desks, sources (RSS + ad-hoc uploads), runs, reports, and settings live in
  Postgres (`db.mjs`, self-migrating + self-seeding from `config/feeds.json` and
  the built-in desk taxonomy). The Python pipeline never touches Postgres — the
  orchestrator **materializes** the enabled sources + desk taxonomy to JSON
  (`config/*.generated.json`) and passes them to the pipeline (`--feeds`,
  `--desks-file`), keeping the pipeline a pure compute step.

**Daily runs are dated and orchestrated.**

- `triggerDailyRun(date)` records a `runs` row and streams a status the UI polls:
  scrape → decompose → index → report, each scoped to the `dt=<date>` partition.
  Ad-hoc uploads run the same decompose→index path via `--input-json`.

## Consequences

- Multiple clients (desktop, browser) share one backend and one dataset.
- The pipeline stays decoupled from the app DB and remains runnable standalone.
- Requires a running Postgres + Qdrant + `ANTHROPIC_API_KEY` for the full stack;
  the API degrades gracefully (health reports which dependencies are down).
- The Celeritas ETL template's unused views + Tauri Rust backend are retained on
  disk but unwired/minimized; they can be pruned later.

## Alternatives considered

- **Vite-dev-bridge as backend** (initial sketch): rejected — couples the backend
  to the dev server and can't be shared by multiple/remote clients.
- **Whole-article RAG**: kept available (`index_corpus.py`) but the product uses
  fact-level retrieval for tighter, more citable grounding.
- **Config in JSON files**: rejected in favor of Postgres so desks/sources are
  editable from the UI and shared across clients.
