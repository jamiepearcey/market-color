# Findings — event-driven & prediction-market edge hunt (2026-07-25)

Testing whether event / prediction-market structure yields a **tradeable** edge.

## Bottom line

**No durable, tradeable edge survived proper out-of-sample testing.** What we found
is real *contemporaneous / venue-specific structure* and reusable *instrumentation* —
not signal. This is consistent with market efficiency in liquid, well-covered corners;
the only candidate edges live in thin/retail venues with minuscule capacity, and even
those did not clear a genuine cross-regime bar.

The decisive discipline throughout was the sequence **placebo-null → real control →
cross-regime out-of-sample**. Out-of-sample stability was the gate that killed every
apparent positive.

---

## 1. Event-conditional correlation (FOMC de-diversification)

Rate-sensitive clusters (banks + homebuilders + rate-proxy REITs/utilities)
re-correlate on FOMC day *beyond* the market and Treasury-curve factors:

- residual correlation jumps **~0.22 → 0.43** on the event day, then decays over ~3 sessions
- cluster-robust; survives a 300-draw **placebo null (p = 0.003)**; holds a within-window split
- mechanism-specific: a *defensives* cluster stays null (Δ −0.04); a broad universe is null

**But** it fails a genuine across-regime out-of-sample test, and is essentially a known
sector/rate factor effect. **Real, not tradeable.**
`scripts/event_regime_correlation.py`

## 2. Embedding-defined risk baskets

News-quote embeddings (all-MiniLM) build genuinely **cross-sector** "risk" baskets that
beat a *real* GICS same-sector control on FOMC (**+0.091 vs +0.003** on sector-neutralised
residuals). So the embedding is a good basket **constructor** / risk-mapping tool.

**But** no cut survives out-of-sample. **Method works; edge is null.**
`scripts/event_embedding_risk.py`, `scripts/gics.py`

## 3. Kalshi macro YES-overpricing (the one promising lead)

Macro/economic contracts (CPI/Fed/jobs/GDP) with **YES priced 50–90c** are systematically
overpriced; **buying NO netted +0.16/contract, cluster-robust t ≈ 3**, net of pessimistic
taker costs, under a pre-registered temporal train/test split.

**Caveat:** single ~72-day regime. Kalshi's public API only retains **~2.5 months** of
settled history, so cross-regime testing is impossible there — we could not separate a
durable bias from that window's macro outcomes happening to skew NO.
`scripts/pm_calibration.py`, `scripts/pm_calibration_oos.py`, `scripts/kalshi_macro_backfill.py`

## 4. Polymarket multi-regime test (the decider)

Pulled **1,369 genuine on-chain-settled** macro contracts spanning **2023–2026** (Polymarket
macro didn't meaningfully exist pre-2023) and applied the *same* pre-registered buy-NO rule.

| Year | buy-NO net / contract | sign |
|------|-----------------------|------|
| 2023 | −0.082 | − |
| 2024 | +0.006 | + |
| 2025 | +0.038 | + |
| 2026 | +0.134 | + |
| **Overall** | **+0.047 (t = 1.24)** | not significant |

Favorite-side edge −0.045 (YES mildly overpriced). Sub-categories mostly positive
(cpi +0.08, jobs +0.065, gdp +0.03) except fed −0.057.

**Result = WEAK / PARTIAL, not clean either way.** YES-overpricing is *directionally*
present (positive 3 of 4 years, growing over time) — **but** overall net is **not
statistically significant** (t = 1.24), 2023 is negative, and the strongest year (2026)
**overlaps the Kalshi window**, so it is not an independent regime (outcome-skew confound
not excluded).

**Verdict:** the Kalshi 20-point magnitude was almost certainly **inflated by
single-window / venue effects**. A faint, direction-consistent YES-overpricing *tendency*
may exist in retail macro prediction markets, but it is unstable across years,
insignificant net of costs, and tiny-capacity — **not a demonstrated tradeable edge.**
Viable at most as a small **personal-scale hobby** if it survives forward paper-trading;
nothing scalable.
`scripts/polymarket_macro_pull.py`, `scripts/pm_calibration_polymarket.py`

---

## Where the honest value is (infrastructure, not signal)

- **Daily options-chain snapshotter** accruing an un-backfillable implied-vol series
  (Yahoo, via a `curl_cffi` crumb workaround) — `scripts/options_snapshot.py`.
- **Cron jobs** banking prediction-market settlements and options snapshots daily.
- Reusable **calibration / event-regime / embedding-basket / pre-registered-OOS** harnesses.

## Still untested (needs data to accrue)

- **Options implied-vs-realised event vol** — the variance risk premium, conditioned on
  narrative attention (news-graph density). This is the one branch with a genuinely
  different prior. It needs the options history the snapshotter is now accruing; revisit
  in a few weeks.

## Method note

The sequence placebo-null → real-control → cross-regime OOS caught two false positives
along the way — a spurious GICS-tag artifact, and an energy result accidentally run
against the wrong event — and cut the Kalshi single-window edge down to a weak, unproven
tendency. Out-of-sample stability was the decisive gate every time.

---

## 5. Classification layer + news → volume (the original question, concluded)

The long-standing "classify news into standardised sectors, then predict volume" thread.
First: a standardised classification layer (`scripts/taxonomy.py` contract +
`scripts/classify_layer.py`) — canonical event types/mechanisms, GICS sector (+MACRO,
+explicit UNK), suggested-vs-verified identifiers — emitted as **additive side-cars**
(`<graph>/classification/*.jsonl`), never mutating the extraction lake.

Coverage: Bloomberg 42% non-UNK sector / 74% events / 78% edges mapped, **1,134 resolved
securities**; India 50% / 47% / 72% but **0 verified tickers** (its `nifty50_resolved.json`
is hallucinated — Jindal Steel→JSWSTEEL — so India is sector-level only until a real
NSE/BSE resolution pass).

**Then the payoff test** (`scripts/news_volume.py`), on 679 tickers / 491,790 ticker-days,
of which only **0.9% carry news**:

| model (news-day conditional, temporal OOS) | OOS R² |
|---|---|
| news only (no volume lags) | **0.036** — news IS informative |
| volume-AR incl. `vol_z(T)` | **0.347** |
| volume-AR + news | **0.345** — incremental **−0.002 (null)** |

Event-type indicators beyond raw news arrival: **−0.002**. Unscheduled subset (n=1,128):
**−0.013**. (A full-panel version showed ≈0 but was *diluted by construction* — 99.1% of
rows have all-zero news features — and should not be cited as the finding.)

**Verdict: news is INFORMATIVE BUT REDUNDANT.** Same-day volume `vol_z(T)` is a sufficient
statistic for news arrival; the text adds nothing over the tape — the volume analogue of
every return result above.

**Timing reframe:** on news days `vol_z(T)` = **+0.421** vs `vol_z(T+1)` = **+0.176** — the
spike lands ON the news day and is >half decayed by T+1, so next-day was the wrong horizon.
The sharp question (does news *lead* volume **intraday**) is **unanswerable with this
corpus**: `published_at` is date-resolution, so news and volume fall in the same bucket.
That is a data-resolution ceiling, not a modelling failure — it needs intraday news
timestamps (the live scrape has them) *and* intraday volume (we do not have it).

**Genuinely useful standalone output** — expected next-day abnormal volume by event type,
usable for execution/liquidity planning with no predictive edge required:
earnings **+0.60** (n=701) · IPO +0.45 · guidance **+0.43** (n=267) · rating_action +0.26 ·
legal_regulatory +0.12 · m_and_a +0.07.
