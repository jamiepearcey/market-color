# market-color

A per-desk **market-color** system: given the assets a trading desk cares about, surface the *current news drivers* behind their moves — **grounded in real article text, point-in-time correct, and honest enough to say "no clear driver" when the evidence is thin.**

Clean-sheet successor to `research/news-narrative-explainer` (v1–v3). Split out 2026-06-30.

---

## Why this exists

The old `news-narrative-explainer` (v3, Rust/DuckDB/GDELT) produced confident macro narratives that, when validated against the underlying documents, turned out to be the **taxonomy's prior dressed up with a high confidence score** — not the actual driver. Root cause: **GDELT GKG has no article body text**, so retrieval generalised to keyword/theme priors. Its own LLM-judge layer caught this (marked the narratives `unsupported`/`refined`).

The fix is not more taxonomy — it's **real full article text to ground on**. That is what `market-color` is built around.

### North star (the two failure modes we design against)
- **No local-search generalisation** — never reach for a plausible macro prior when the evidence isn't there. Abstain instead.
- **No temporal inconsistency** — everything point-in-time correct: only what was knowable as-of time *T*, no stale/look-ahead leakage, reproducible.

---

## Architecture / pipeline

```
 SOURCES (config/feeds.json: 63 curated feeds, per-desk + global)
    │
    ▼
 [1] CRAWL  crawl_corpus.py
    │   sitemap + RSS discovery (uncapped) → trafilatura full-body extraction
    │   → dedup → date-partitioned parquet.  Robots-respecting, resumable, incremental.
    ▼
 data/news_corpus/dt=YYYY-MM-DD/part-000.parquet   (the corpus: full article bodies)
    │
    ▼
 [2] INDEX  index_corpus.py
    │   quality gate (drop non-articles) → per-article desks (embedding zero-shot)
    │   → embed title+body → Qdrant (TurboQuant), point-in-time + desk payload
    ▼
 Qdrant collection "market_color"   (doc-level retrieval, desk-scoped, as-of, abstaining)
    │
    ▼
 [3] FACTS + GRAPH  (GraphRAG experiment)
    │   codex exec → atomic facts per doc (claim/subject/predicate/object/
    │   direction/magnitude/time/cause/entities/confidence, with provenance)
    ▼
 graph_experiment.py
    │   merge facts → facts.parquet → embed CLAIMs → Qdrant "market_color_facts"
    │   (entities indexed = graph adjacency)
    │   search = SEED (vector) → EXPAND (1-hop shared entities) → CAUSAL CHAINS
    ▼
 grounded, transmission-aware market color   (per desk, point-in-time, cited)
```

---

## Components

| File | What it does | Run |
|---|---|---|
| `config/feeds.json` | 63 curated feeds; per-feed `scope`/`tier`/`desks`/`access`(open\|headline\|paywall\|licensed)/`method`(rss\|google_news_rss). | — |
| `crawl_corpus.py` | **Corpus builder.** 2-stage sitemap+RSS discovery → `trafilatura` full-body extraction → dedup → parquet. Hardened (bounded extraction pool + per-call timeout + `os._exit` so a hung `trafilatura` can't deadlock), **incremental flush** (`--flush-every`), **`--resume`** (skips already-collected URLs), point-in-time window filter, Google-News link decoding. | `uv run --with httpx --with feedparser --with trafilatura --with pyarrow --with python-dateutil python crawl_corpus.py --since-hours 72` |
| `scrape_news_feeds.py` | Older RSS-headline-only scraper (superseded by crawler for bodies; kept for feed validation). | `... python scrape_news_feeds.py --validate` |
| `index_corpus.py` | **Retrieval index.** Ports the "fast Qdrant" recipe (TurboQuant bits2 + oversampling/rescore). Embeds real `title+body`. Adds: (1) **quality gate** `is_real_article()` (drops section/listing/report/nav/stub pages, ~15%); (2) **per-article desks** via embedding zero-shot vs `DESK_ANCHORS` (fixes source-level mis-tagging); (3) **score-floor abstention** (`--min-score` → `no_strong_match`). Point-in-time (`published_ordinal` day-range + `published_epoch` as-of). | `uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed --with python-dateutil python index_corpus.py index --recreate` then `... search --desk energy --min-score 0.45 --query "..."` |
| `graph_experiment.py` | **GraphRAG experiment.** Merges Codex fact batches → `facts.parquet`; light entity canonicalization; embeds each `claim` into Qdrant `market_color_facts` with entities as a KEYWORD payload (the graph adjacency). `search` = seed (vector) → expand (1-hop via shared entities) → causal chains. | `... python graph_experiment.py build --recreate` then `... search --query "..."` |
| `facts_work/` | Codex fact-extraction workspace: batch inputs, prompt template, parallel driver `run_codex_batches.sh`, per-batch `energy_facts_*.jsonl`, merged `facts.parquet`. | `MAXP=3 bash facts_work/run_codex_batches.sh` |

See `docs/decisions.md` for *why* each choice was made, and `docs/sourcing-research.md` for the premium-wire / vendor research.

---

## Current state (as of 2026-07-01)

**Corpus** (3-day pull, 2026-06-27 → 07-01):
- **1,712 full-body articles**, ~**1.68M tokens**, 4.4 MB parquet. 922 stale-dated pages dropped by the point-in-time filter.
- Extraction by access tier: **open 84%** (1,323/1,568), headline 60% (378/628, mostly paywalled bodies), paywall ~30% (11/35).
- Per-desk full-body: macro 1,025 · equities 606 · fx 319 · geopolitics 283 · **energy 198** · asia 196 · metals 145 · crypto 145 · rates 102. (macro/equities inflated by multi-tagging; focused desks are the clean slices.)

**Index**: 1,712 → **1,446 indexed** after the quality gate. A 20-query eval judged **~70% of queries return genuinely useful top results** at zero tuning; the three fixes (quality gate, per-article desks, abstention) closed the main failures (listing-page contamination, rates desk mis-tagging).

**Facts + graph experiment** (energy desk): **86 gated articles → 450 atomic facts** via parallel Codex, embedded + graphed. `seed → expand → causal chains` produces a readable, grounded transmission map (e.g. *Hormuz shut → LNG supply down → Pakistan demand up*; *US-Iran MoU → banks cut oil forecasts*; *oil-price spike → US production up*). See `docs/decisions.md` §GraphRAG.

**The premium-wire reality** (from research, full detail in `docs/sourcing-research.md`): open-web full text is ours to crawl (this repo). **Bloomberg full text is only available via a ~$100k+/yr enterprise Data License / B-PIPE** — no aggregator carries it, scraping is prohibited. Reuters full text via LSEG MRN; WSJ/Reuters/FT bodies via Dow Jones Factiva DNA. GDELT is metadata-only (why v3 failed).

---

## Known limitations

- **Coverage is DM / Western / English / markets-desk-centric.** Weak for EM sovereign risk, commodity local-language, political-event risk (no ACLED, no EM-regional/local outlets, no shipping/ag feeds). See `docs/decisions.md` §Use-case coverage.
- **Only the news layer exists.** Cross-asset/impact-mapping use cases also need a structured-data layer (CDS, spreads, FX reserves, World Bank/IMF/BIS) that is not built.
- **Entity canonicalization is coarse** (near-dup nodes) and `cause` is free-text (not resolved to entity nodes → causal chains are readable but not yet fully entity→entity traversable).
- Premium paywalled sources (Bloomberg/WSJ/FT) stay headline-only without a licensed feed.

---

## Roadmap

1. **Entity/cause canonicalization** → true entity→entity causal graph (multi-hop transmission).
2. **Subgraph → grounded desk brief renderer** (the last mile: retrieval → the actual market-color note, with citations).
3. **Scale facts to all desks** (same pipeline, more Codex batches).
4. **Structured-data layer** (CDS/spreads/reserves/World Bank/IMF/BIS) for impact-mapping use cases.
5. Broaden sources for EM/commodity/political coverage.

---

## Runtime / environment notes

- **Python via `uv run --with ...`** (no project venv). Python 3.14 present; scripts pin deps per-invocation.
- **Qdrant**: local binary at `../news-narrative-explainer/v3/tmp/qdrant-bin/qdrant` (v1.18.2, arm64), storage in `qdrant_storage/`. Start with `QDRANT__STORAGE__STORAGE_PATH=... <binary>`. (Docker `qdrant/qdrant` also works.)
- **Embeddings**: fastembed `all-MiniLM-L6-v2` (384-d, local, no server). Ollama also present (`all-minilm`, `embeddinggemma`, `Qwen3-4B-Instruct`).
- **LLM for fact extraction**: **no Groq/Anthropic/OpenAI API keys in env.** Fact extraction runs on **Codex CLI** (`codex exec`, ChatGPT auth, free, non-interactive). NB: the "never spawn codex from Claude (hangs)" caution is the *interactive* TUI only — `codex exec -C <dir> -s workspace-write --skip-git-repo-check - < prompt` run in the background is fine.
- Local Qwen3-4B (Ollama) is the fallback fact extractor if Codex is unavailable.
