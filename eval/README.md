# eval — retrieval eval / backtest harness

Frozen regression cases for the `market_color` Qdrant index, replayed with the
**exact production search semantics** (`run_eval.py` imports `index_corpus.py`
and calls its `embed` / `_client` / `_filter` directly — same model, same desk
and as-of filters, same TurboQuant rescore params, same score-floor abstention).
No shelling out, so a full run takes seconds after the embedding model loads.

Every retrieval expectation was grounded against a real corpus row
(DuckDB over `data/news_corpus/dt=*/part-*.parquet`, window 2026-06-27 → 07-01)
before being frozen.

## Run

```bash
# from the repo root (needs Qdrant at :6333 with the market_color collection)
uv run --with 'qdrant-client>=1.15' --with fastembed --with 'duckdb>=1.0' \
  python eval/run_eval.py
```

Options:

| flag | meaning |
|---|---|
| `--only <id>` | run a single case (repeatable) |
| `--k <n>` | retrieval depth (default 8; never below a case's `expect.in_top`) |
| `--qdrant-url URL` | Qdrant endpoint (default `http://localhost:6333`) |
| `--cases PATH` | alternate cases file |
| `--verbose` | print top hits for every case, not just failures |

Output: one `PASS`/`FAIL` line per case, then a JSON summary
`{total, passed, failed, by_kind, failures:[ids]}`. Exit code **1** if any
case fails — safe to wire into CI or run after any re-index / `index_corpus.py`
change.

## Case kinds — what each protects

- **`retrieval`** — *grounding*. A desk-scoped query must surface a known,
  verified-in-corpus document in the top `expect.in_top` hits at/above
  `min_score`. Catches regressions in embedding text construction, the quality
  gate, per-article desk assignment (a doc silently dropping off its desk), and
  index corruption.
- **`abstain`** — *abstention calibration* (the anti-generalisation north
  star). Queries about topics **verified absent** from the corpus window must
  return `no_strong_match` at `min_score` 0.45. Catches the v3 failure mode:
  confidently returning a plausible-but-wrong doc instead of saying "no
  evidence".
- **`asof`** — *point-in-time correctness*. The same query run with an `as_of`
  **before** the target doc's publish time must exclude it
  (`expect.present: false` — no look-ahead leakage), and with a **later**
  `as_of` must surface it (`expect.present: true`). Cases come in
  before/after pairs. Note: corpus publish timestamps are day-granularity
  (midnight UTC), so cutoffs sit on day boundaries.

## Case format (`cases.jsonl`, one JSON object per line)

```json
{"id": "r-energy-hormuz-flows", "kind": "retrieval", "desk": "energy",
 "query": "Strait of Hormuz reopening crude oil flows", "min_score": 0.45,
 "expect": {"any_url_contains": ["hormuz"], "in_top": 5}}
```

Fields:

- `id` — unique; prefix by kind (`r-`, `ab-`, `asof-`).
- `kind` — `retrieval` | `abstain` | `asof`.
- `desk` — optional; string or list, applied as the same `desks` MatchAny
  filter as `index_corpus.py search --desk`.
- `query`, `min_score` (default 0.45 if omitted).
- `as_of` — `asof` cases only; ISO timestamp, same semantics as `--as-of`
  (`published_epoch <= as_of`).
- `expect` (not for `abstain`):
  - matchers, **OR-ed together** — a hit counts if *any* matcher matches:
    - `any_url_contains`: case-insensitive substrings, any of the list;
    - `any_source`: exact `source_name` values;
    - `any_title_regex`: Python regex, `re.search` on the title.
  - `in_top` — rank window the match must appear in (default 5);
  - `present` (`asof` only) — whether the target doc should appear.

## Adding cases

1. **Ground it first.** Find the real row(s) with DuckDB, e.g.:
   ```bash
   uv run --with 'duckdb>=1.0' python -c "
   import duckdb
   print(duckdb.sql(\"\"\"SELECT source_name, published_utc, title, url
     FROM read_parquet('data/news_corpus/dt=*/part-*.parquet')
     WHERE extraction_ok AND title ILIKE '%hormuz%'\"\"\"))"
   ```
   For `abstain` cases, verify the topic is genuinely absent (keyword-check the
   bodies, then confirm the search actually abstains). For `asof` pairs, pick a
   doc whose publish date lets a day-boundary cutoff exclude it, and check no
   *earlier* doc also matches your URL/title patterns.
2. Make the matcher **specific to the story, not the phrasing** — a distinctive
   URL slug beats a title regex; several alternative slugs are fine (coverage of
   the same story by multiple outlets).
3. Append the JSON line to `cases.jsonl`, then `run_eval.py --only <id> --verbose`.
4. **Never weaken a frozen case to make it pass.** If retrieval genuinely
   misses (e.g. the doc exists but scores under the floor, or an off-topic doc
   beats the abstention floor), leave the case failing — that is the harness
   doing its job. Only fix cases whose *expectation* was wrong (bad slug, topic
   not actually absent, wrong desk for the story).

## Known-failing cases (genuine, kept deliberately)

- `ab-cocoa-ghana` — "cocoa futures surge after Ghana and Ivory Coast harvest
  failure" (cocoa verified absent from the window) retrieves an unrelated
  *Guinea gold-refining hub* story at 0.488, above the 0.45 floor. This is a
  real abstention-calibration gap: West-Africa+commodity surface overlap beats
  the floor. Fix belongs in retrieval (better floor calibration or reranking),
  not in the case.
