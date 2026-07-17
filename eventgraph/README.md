# eventgraph

A financial **event / sensitivity knowledge graph** extracted from news feeds,
designed so the narrative layer *joins* to formal calendars, realised price/factor
data, and prediction markets. Isolated sub-project; integration considered later.

## The three planes

| plane | what | source | join key |
|---|---|---|---|
| **narrative** | what an article asserts — entities, causal edges, sensitivities, sentiment, forward views | LLM extraction (one pass) | — |
| **formal** | scheduled events, factor defs, prediction-market contracts | external feeds | `series_id`, `contract_ref` |
| **realised** | actual returns, realised betas, event surprises, implied probs | price/derivative data | `entity.identifier` |

The extractor builds the **narrative plane + the join keys**; the formal and
realised planes are populated by downstream (non-LLM) jobs.

## Storage split — Postgres vs DuckLake

Rule: **Postgres holds only what mutates / needs uniqueness / is curated; DuckLake
holds the append-only OLAP facts.** (One Postgres instance can also *be* the
DuckLake catalog — see `lake/0001_attach.sql`.)

- **`schema/`** — Postgres tier (thin): `entity` + `entity_alias` (canonical
  resolution, upserted during ingest), `factor`, `event_series`, `proposition`
  (lifecycle: open→resolved), `extraction_run`.
- **`lake/`** — DuckLake tier (DuckDB dialect): `document`, `chunk`, `doc_figure`,
  `event`, `causal_event_edge`, `sensitivity_edge`, `sentiment_annotation`,
  `probability_annotation` (facts) + `realised_*` / `calendar_event` /
  `market_implied_prob` (derived) + analytics views (hub index, prob divergence).

We never UPDATE a fact row: realised metrics are separate append-only tables
joined by natural key.

## Pipeline (Rust, `src/`)

```
feed  →  extract           →  normalize                    →  { Postgres, DuckLake }
raw      DocExtraction        resolve entities -> ids,          upsert dims/lifecycle
docs     (narrative JSON)     chunk, assign keys, GraphBatch    append facts
```

- `model.rs` — the type system: `DocExtraction` (LLM contract) + `GraphBatch`
  (normalized rows mirroring the two schemas).
- `feed.rs` — `Feed` trait + raw-doc JSONL reader (Bloomberg-parquet feed later).
- `extract.rs` — `Extractor` trait + Groq client; the rich schema prompt.
- `normalize.rs` — entity resolution + `GraphBatch` assembly + chunking.
- `pg.rs` — emits Postgres upsert SQL (dims + lifecycle).
- `lake.rs` — writes per-table JSONL + a DuckLake load script (applied via the
  `duckdb` CLI); no native libduckdb dependency.
- `main.rs` — CLI: `ingest`, `scrape`.
- `scrape/` — self-contained feed-scraping job (Rust port of `../crawl_corpus.py`):
  scrapes every feed in `../config/feeds*.json` and writes the same
  date-partitioned Parquet corpus (`dt=YYYY-MM-DD/part-000.parquet`, 22 cols,
  append-merge dedup by content_hash + canonical_url). Structured as idempotent,
  retryable **activities** (`fetch_feed`, `resolve_url`, `fetch_and_extract`,
  `write_partitions`) orchestrated by a standalone rayon **workflow**
  (blocking ureq + rayon, no async runtime). `ParquetFeed` (in `feed.rs`) reads
  the output back so `eventgraph ingest --feed <corpus>` consumes it directly.

  `eventgraph scrape [--feeds a.json,b.json] [--out DIR] [--since-hours 48]`
  `[--max-per-feed 200] [--concurrency 12] [--limit N]`

  **Temporal (honest note):** `scrape/temporal.rs` documents how each activity
  maps onto a Temporal activity and the workflow onto a Temporal workflow, plus a
  feature-gated (`--features temporal`) worker *skeleton*. There is **no** hard
  dependency on a Temporal SDK and **no** running server is required — the rayon
  runner is the shipped orchestrator; Temporal is the documented durable drop-in.

  **Google-News decode (full coverage, pure Rust):** ~40% of feeds are
  `google_news_rss` (EM central banks, IMF, World Bank, VoxEU, US Treasury, PBoC,
  most EM regional press). `resolve_url` decodes both forms: the older base64
  `CBMi…` embedded-URL fast-path, and the opaque `AU_yqL…` batchexecute form via a
  pure-Rust port of the maintained `googlenewsdecoder` flow — GET the article page,
  scrape the `data-n-a-sg`/`data-n-a-ts` signature+timestamp from
  `c-wiz > div[jscontroller]`, POST them to `/_/DotsSplashUi/data/batchexecute`,
  and parse the publisher URL out of the response. Concurrency to news.google.com
  is capped (semaphore) with a small per-request delay. No Python dependency.

  Approximations vs the Python: extraction is `scraper` (`<article>/<p>` text with
  RSS-summary fallback), not trafilatura-grade; sitemap discovery is not ported
  (feeds + Google-News decode only).

## Status

Schema (both tiers) + type system done. Rust pipeline modules are the next build
step. Extraction reuses the proven Groq `gpt-oss-120b` path from the parent
`receptors/` work; the rich taxonomy (events/sensitivities/sentiment/propositions)
is validated before any full re-extract.
