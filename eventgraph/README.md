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

## Formal plane: free authoritative calendars (`scripts/formal_calendar.py`)

Populates `eg.calendar_event` + `event_series` from free, authoritative,
timestamp-precise sources (deliberately not scraped consensus calendars):

- `alfred` — US macro first prints via FRED/ALFRED `output_type=4`
  (actual = as-first-published, `realtime_start` = publication date; no revision
  lookahead). 10 tracked series (CPI/PPI/NFP/GDP/…); needs a free FRED key.
- `ics` — generic agency release-schedule ICS ingester; schedule-only rows,
  `--map 'regex=SERIES_ID'` to align series ids. TZID local times (incl. BLS's
  `US-Eastern`) are converted to UTC via zoneinfo (DST-correct). Live-verified:
  `https://www.bls.gov/schedule/news_release/bls.ics` (needs a real User-Agent)
  and `https://www.bea.gov/news/schedule/ics`.
- `edgar` — corporate 8-K/10-Q/10-K acceptance timestamps per ticker from the
  SEC submissions API (`{TICKER}-8-K` series) — the exact moment the market
  could know.

Same convention as the realised scripts: append-only JSONL into
`<graph-dir>/lake/`, joined on `series_id`/`event_time`.

Lineage (`lake/0007_lineage.sql`): every formal-plane row carries `source`
(feed kind), `source_ref` (exact origin: FRED series id, ICS URL, EDGAR
accession number, HF revision sha, chain-file glob) and `ingested_at` (fetch
time, UTC) — the formal-plane analogue of the narrative plane's `run_id`.
Schedules mutate upstream, so "as currently known" = latest `ingested_at` per
(series_id, event_time).

`lake/0008_provenance.sql` unifies the two vocabularies WITHOUT a re-run
(facts already carry the columns): `eg.v_provenance` = one row per fact across
BOTH planes with (plane, fact_table, fact_key, doc_id, chunk_id, source,
source_ref, known_at) — narrative resolves fact→doc→run→`eg.extraction_run`
(a tiny lake mirror of the PG dim; backfill one-off via postgres ATTACH,
documented in the file), formal passes through 0007's lineage columns.

`lake/0009_surprise.sql` — the surprise join promised in 0002:
`eg.v_event_surprise` matches each narrative `event` to the nearest formal
`calendar_event` (same `series_id`, ±36h) → `surprise` (actual − expected/prior),
`surprise_pct`, and `narrative_err` (what the article CLAIMED the print was
minus truth — mis-reporting detector). OLD-SHAPE SAFE: narrative side needs
only 0002 base columns, so pre-migration graphs (incl. in-flight extraction
runs) load and join unchanged; pre-0007 calendar rows (NULL `ingested_at`)
rank as the oldest fetch and are superseded by any stamped re-fetch. Caveat:
joins hit only where the extractor's `series_id` slug resolves to a canonical
calendar id — closed by `lake/0010_series_alias.sql` +
`scripts/resolve_series.py`: an `eg.series_alias` mapping table (same doctrine
as `entity_alias`; facts never rewritten, so it works on in-flight graphs)
populated by word-boundary keyword rules (macro → US-CPI/US-NFP/…/US-FOMC)
plus an earnings pass (`event_type` ∈ earnings/guidance/… + resolved US symbol
→ `{TICKER}-8-K`). `v_event_surprise` resolves through it (latest mapping
wins; unmapped slugs fall through unchanged). `--dry`/`--unmapped` to preview
and dump the residual tail for manual mapping.

**Implied-probability feed** (`scripts/implied_prob.py` + `lake/0011_market_quote.sql`)
— closes the `v_prob_divergence` loop (narrative `probability_annotation` vs
market-implied prob per proposition):

- `kalshi` — Kalshi public market data (no auth); YES prob from the `*_dollars`
  fields (already 0..1), `--series KXFED,…` to target event series.
- `polymarket` — Gamma API (no auth); YES prob from `outcomePrices`,
  `--query` word-boundary keyword filter.
- Both append `eg.market_quote` (raw, keyed by the market's natural key, full
  lineage). `eg.proposition_contract` maps `proposition_id ↔ contract_ref`
  (backfill from PG `proposition.contract_ref` or a text-match pass);
  `eg.v_market_implied` resolves quotes into the `market_implied_prob` shape →
  feeds `v_prob_divergence`. Fed-funds-futures / CDS implied probs are the same
  table with a different `instrument` — not yet wired (no clean free futures API).

**Loading the formal plane** (`scripts/load_formal.py`): the Rust
`eventgraph ingest` path loads the narrative tables; this loads the
Python-written formal outputs with the same convention (`.read` the idempotent
DDL 0002–0011, then `INSERT ... BY NAME` for JSONL / `read_parquet` for the
option snapshots). `calendar_event` / `market_quote` / `series_alias` /
`proposition_contract` (json), `option_iv` / `option_surface` (parquet);
`event_series` is intentionally skipped (it's the Postgres dim). `--db` for a
research DuckDB file, `--catalog`/`--data-path` for a DuckLake attach,
`--recreate` for a clean deterministic reload. `src/lake.rs` applies the same
0002–0011 DDL list, so the Rust ingest path also creates the formal tables.

**Options layer** (`scripts/options_surface.py` + `lake/0006_options.sql`):

- `hf-iv` — [gauss314/options-IV-SP500](https://huggingface.co/datasets/gauss314/options-IV-SP500)
  (Apache-2.0): NAME-LEVEL daily IV — moneyness-ladder IVs + HV terms + VIX,
  3,893 US symbols, 2019-10..2023-07 (3.16M rows) → `eg.option_iv` parquet.
  The free single-name implied layer (vol risk premium, wing skew at query time).
- `chains` — full EOD chain files (Kaggle "SPY Options EOD 2010-2023",
  optionsDX; manual download) → one `eg.option_surface` row per (underlier,
  date): front/next ATM IV, put skew, term slope, ATM-straddle implied move.
  Vendor column names vary → `--col ours=theirs`.
- `eg.v_event_implied` joins the surface onto `calendar_event`: the implied
  move / crash-skew standing just before each macro event.

## Status

Schema (both tiers) + type system done. Rust pipeline modules are the next build
step. Extraction reuses the proven Groq `gpt-oss-120b` path from the parent
`receptors/` work; the rich taxonomy (events/sensitivities/sentiment/propositions)
is validated before any full re-extract.
