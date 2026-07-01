# Decision log

ADR-lite record of the choices behind `market-color` and their rationale. Newest context at the relevant section. See `sourcing-research.md` for the vendor/premium-wire evidence and `../README.md` for the pipeline.

---

## D1 — Clean-sheet rebuild around full article text
**Context:** `news-narrative-explainer` v1–v3 produced confident macro narratives that its own LLM-judge marked `unsupported`/`refined`. The narratives were the deterministic factor-taxonomy's *prior* (`provenance=taxonomy-fallback`), not the day's actual driver, with an uncalibrated `fit_score` and `fit_confidence=1.0`.
**Root cause:** GDELT GKG exposes **no article body text** — retrieval generalised to keyword/theme priors.
**Decision:** Start `market-color` as a full-text corpus project. Grounding requires real article bodies. **Status: done.**

## D2 — North star & the two failure modes
**Decision:** Deliver **per-desk market color** that (a) does **not** fall to *local-search generalisation* (abstain rather than reach for a macro prior when evidence is thin) and (b) is **point-in-time correct** (no stale/look-ahead leakage, reproducible). Every downstream choice is judged against these two.

## D3 — Sourcing: crawl open web, include marquee names as headline-tier
**Decision:** Curate 63 feeds spanning global wires + per-desk verticals. Include Bloomberg/WSJ/FT/Economist even though paywalled (**headline** access tier — capture headline+link+snippet). Route no-RSS wires (Reuters/AP) through **Google News RSS** for discovery.
**Rationale:** Don't drop the best sources on RSS-availability grounds; tag *how* we get each (`access`/`method`) and solve access per-tier. **Status: done (`config/feeds.json`).**

## D4 — Google News is discovery, not a content source
**Context:** Google News RSS returns headline stubs (no body, ~100-item cap, obfuscated `CBMi…`/`AU_yqL…` links).
**Decision:** Use it only to *discover* URLs for no-RSS wires; decode links (`googlenewsdecoder`) and fetch the publisher. Body availability then depends on the publisher (AP/Kitco/Metal.com = yes; Reuters/WSJ = 401/paywall). **Status: done — recovered AP/Kitco/Metal.com/Fastmarkets bodies.**

## D5 — Crawler: sitemap+RSS discovery → trafilatura; hardened & resumable
**Decisions:**
- Discovery via **news sitemaps** (uncapped) + RSS; extraction via **trafilatura** (benchmark-leading OSS extractor).
- **Hardening:** trafilatura can hang on pathological pages and exhaust the thread pool → deadlock. Fixed with a bounded extraction pool + per-call `asyncio.wait_for` timeout + HTML size cap + `os._exit` at shutdown.
- **Incremental flush** (`--flush-every`) so progress is durable mid-run; **`--resume`** reads existing corpus `doc_id`s and skips them.
- **Point-in-time hygiene:** drop articles whose real publish time is outside the window (sitemaps surface stale URLs). 922 dropped on the 3-day pull.
**Status: done.** Result: 1,712 full-body articles.

## D6 — Premium wires require a licensed feed (don't scrape)
**Decision:** Bloomberg/Reuters/WSJ/FT full text is **not** obtainable free or by scraping. Bloomberg only via its own ~$100k+/yr Data License/B-PIPE; WSJ/Reuters/FT via Factiva DNA; Reuters via LSEG MRN. Treat these as a procurement decision, parked unless a desk mandates it. Full evidence in `sourcing-research.md`. **Status: documented, not wired.**

## D7 — GDELT: dropped
**Context:** GDELT was v3's substrate and is metadata-only. Considered re-using it as a Layer-2 *anomaly/z-score* signal, but the user's prior experiments with that "don't look good."
**Decision:** **Drop GDELT.** It's the wrong tool for grounded narrative; the value is in full-text + structured market data, not GDELT tone/volume. **Status: dropped.**

## D8 — Graph database: embedded, not a server; and build the graph from facts
**Context:** Asked whether to use a graph DB.
**Decision:**
- The **news corpus** is not a graph problem → keep parquet + DuckDB + Qdrant.
- The **transmission/impact layer** *is* graph-shaped, but at this scale (hundreds of entities, low-thousands of edges) use **DuckDB adjacency tables or embedded Kùzu — NOT Neo4j/a server.**
- Originally "defer the graph — no edge data." **This was superseded** when the user proposed extracting atomic facts: **the facts ARE the edges.** Building the graph from corpus-extracted facts is legitimate now. **Status: graph built from facts (see D11).**

## D9 — Atomic facts + graph over summarization (GraphRAG)
**Decision:** Reduce each doc to **atomic facts** (checkable claims w/ provenance) rather than summaries. Facts give: grounding-by-construction, corroboration = in-degree, multi-hop transmission, clean point-in-time, cheaper query-time. **Summarization is not discarded** — it becomes the *final render over a retrieved fact subgraph* (grounded generation). This is GraphRAG. **Status: experiment done (D11).**

## D10 — Cost: the corpus is tiny; process-once, cheap models for bulk
**Context:** Whole 3-day corpus ≈ **2M input tokens**; a desk-day ≈ ~135K tokens (~100–150 docs).
**Decisions:**
- Cost is **cents**: whole-corpus classify+summarize is ~$2.50 (Anthropic Haiku) / ~$0.14–1.58 (Groq 8B→70B). Not the deciding factor — choose for *grounding quality*, not price.
- **Process each doc once at ingest**, not per query; assemble briefs from stored structured outputs.
- Let **embeddings do the grouping (free)**; use the LLM only to summarize/judge representatives — cheaper than mapping every doc and equally grounded.

## D11 — Fact extraction runs on Codex CLI; parallel batches
**Context:** No Groq/Anthropic/OpenAI API keys in env. Ollama has Qwen3-4B (local fallback).
**Decision:** Use **`codex exec`** (ChatGPT auth, free, non-interactive) as the fact-extraction engine. The recorded "never spawn codex from Claude (hangs)" caution is the *interactive* TUI only; `codex exec -C <dir> -s workspace-write --skip-git-repo-check - < prompt` run in the **background** is fine. Parallelise via `facts_work/run_codex_batches.sh` (batches ×12 docs, `MAXP=3` concurrent, resume-safe).
**Result:** 86 gated energy articles → **450 atomic facts**. Job posting → 0 facts (negative test passed). **Status: done.**

## D12 — Index quality fixes (from a 20-query eval)
**Context:** A 20-query eval of `market_color` scored ~70% good at zero tuning; failures were listing-page contamination and rates-desk mis-tagging.
**Decisions (all in `index_corpus.py`, corpus untouched, re-indexable):**
- **Quality gate** `is_real_article()` — drops section/listing/report/nav/stub pages by title/prose heuristics + source-specific URL rules (EIA `/todayinenergy/`, OilPrice `.html`). ~15% dropped (1,712 → 1,446).
- **Per-article desks** — embedding zero-shot vs `DESK_ANCHORS` replaces coarse source-level tags (fixed the rates desk: Fed/Treasury FXStreet articles now surface). `source_desks` kept for reference.
- **Score-floor abstention** — `--min-score` drops sub-threshold hits → `no_strong_match` (the anti-generalisation guard).
**Status: done, verified.**

## D13 — GraphRAG experiment: seed → expand → causal chains
**Design (`graph_experiment.py`):** merge facts → `facts.parquet`; light entity canonicalization (`ALIASES` + normalize, drop clause-entities); embed each `claim` into Qdrant `market_color_facts` with **entities indexed as KEYWORD = graph adjacency**. `search` = **seed** (vector over facts) → **expand** 1-hop via shared-entity Qdrant filter → **causal chains** (`cause → subject`).
**Result:** produces readable, grounded transmission maps (e.g. *Hormuz shut → LNG supply down → Pakistan demand up*; *US-Iran MoU → banks cut oil forecasts*; *oil-price spike → US production up*) — the structured, traversable output summarization can't.
**Rough edges:** entity canonicalization still coarse (near-dup nodes); `cause` is free-text, not resolved to entity nodes (chains readable but not fully entity→entity traversable); MiniLM scores modest on thin sub-topics.
**Status: experiment validated; next = canonicalization + subgraph→brief renderer.**

## D14 — Use-case coverage audit (what we can/can't serve)
**Context:** Evaluated 5 target cases (EM sovereign risk, FX/rates macro, commodity macro, political-event risk, wider economic-impact mapping).
**Findings:** Coverage is **commodity 🟢 / FX-rates DM 🟡 / EM-sovereign, political, impact-mapping 🔴**. Two structural gaps: (1) corpus is DM/Western/English/markets-centric — missing EM-regional, shipping, agriculture, non-G3 central banks, ACLED; (2) **only the news layer exists** — these cases also need a structured-data layer (CDS, spreads, FX reserves, World Bank/IMF/BIS) and event DBs (ACLED). **Status: documented as roadmap; not built.**

## D15 — Integrate our facts + graph into the parallel UI/MCP system
**Context:** A separate parallel agent built a market-color chat/reports UI (`ui/`) whose backend (`ui/server/chat-core.mjs`) spawns `mcp/qdrant_facts_server.py` (tool `search_market_facts`, collection `market_facts`) and reuses our `index_corpus.embed`. But its fact pipeline was scaffolded-not-run (empty `market_facts`), its fact schema had entities but no causal graph, and it was separate from our facts (`market_color_facts`).
**Decision (user: "inc the graph"):** Bridge our facts *and* the graph into the UI, additively:
1. `bridge_facts_to_market_facts.py` (our file) — loads `facts_work/facts.parquet` (450 energy facts) into `market_facts` with a **superset payload**: the UI's fields + our graph fields (`subject`/`predicate`/`object`/`cause`/`confidence`), same MiniLM + TurboQuant config.
2. `mcp/qdrant_facts_server.py` (their file, **additive, backward-compatible** edit): `search_market_facts` now returns each fact's `cause`/`subject`/`predicate` and, with `expand=True`, appends a **graph expansion via personalized PageRank** (spreading activation from the seed entities over the co-occurrence graph — multi-hop-aware, cross-desk; ranks non-seed facts by graph proximity, with `shared_entities` count). Pure-Python sparse PPR, graph built+cached from `market_facts` once per process. The UI becomes GraphRAG-aware through the tool it already calls — **no `chat-core.mjs` change.**
   - "Change 1" (2026-07-01): swapped the initial 1-hop `entities MatchAny` expansion for PPR — verified it surfaces the transmission chain (Hormuz reopening + US-Iran MoU → oil-price falls → Brent forecast cuts) instead of near-duplicate LNG forecasts.
   - "Now" recommendations (2026-07-01): added **agentic traversal primitives** to the MCP — `facts_for_entity`, `neighbors`, `trace_causes` (drivers/effects), `entity_path` (BFS ≤3 hops, visited set, frontier cap 30) — plus the **reasoning scaffold + grounding self-check** in `ui/server/chat-core.mjs` CHAT_SYSTEM (search → follow graph → assemble driver→asset→effect chain → corroborate → assert only fact-backed edges, else abstain). chat-core lists MCP tools dynamically, so Claude (Opus 4.8, 6 tool rounds) picks them up with no other wiring. All 4 tools verified.
   - Deferred ("Later"): community detection + per-community summaries (GraphRAG "global" mode) for broad "what's the theme" questions — worth it once facts scale beyond energy/450. And harden entity/cause canonicalization (near-dup nodes; resolve `cause` to entity nodes) after the E2E flow is nailed.
**Verified:** tool returns seed + graph-neighbor facts with causal drivers (Hormuz/LNG seeds expand to EU jet-fuel/polyolefins). **Caveats:** (a) `market_facts` is **energy-only** (450 facts) — scale by extracting facts for all desks and re-bridging; (b) if the parallel agent runs its own `decompose_facts.py`/`index_facts.py --recreate`, it will overwrite `market_facts` and (unless they add the graph fields to their payload) drop `cause`/`subject`/`predicate` — reconcile then. **Status: done, energy scope.**
