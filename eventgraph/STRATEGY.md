# eventgraph — strategy & measurement doctrine

## The thesis (and the mistake it corrects)

News is a **cross-sectional, idiosyncratic** force: a specific event drives a specific
name *relative to* the market. Our first news-conditioned risk model aggregated news
into 9 macro factors and read the net — but the market portfolio is a *diversified
portfolio*, engineered to cancel idiosyncratic drivers. So "aggregate news nets to ≈0
and barely moves the tail" was almost a **tautology**. We measured news against the one
target it is structurally guaranteed to be weak against.

**Re-aim the unit of analysis from `factor` to `(event → name)`, keep the conditioning,
and news explains a great deal.**

## Evidence (measured, not asserted)

`cross_sectional_ic.py` on the 2010–2013 corpus (1,246 aligned name-day observations,
413 names; abnormal return = residual on the engine's 9 factors):

| Measure | Result | Read |
|---|---|---|
| **Contemporaneous name-level IC** | **+0.128 (t +4.5)** | News aligns with names' *idiosyncratic* moves — strongly. |
| Lag +1 name-level IC | −0.00 (t −0.1) | News **explains** but does not **predict** next-day → efficient pricing. |
| Attribution precision, big movers (\|z\|>2, same-day directional edge) | **66%** (68/103, vs 50% chance) | When we have a same-day directional edge on a big mover, we call the sign right 2/3 of the time. |
| Descriptive event-study (prior) | **t = 3.6** | Independent confirmation the contemporaneous signal is real. |

The aggregate model on the SAME graph produced a ~−0.0004 tail shift (null-looking).
Same data, same graph — the level of analysis was the whole difference.

### Coverage is a DATA-DENSITY limit, not a method limit
Full-panel coverage (`novelty_coverage.py`): of 16,905 big idiosyncratic movers across 413
names, only **1.1%** have a graph edge within 2 days — because the 6k-doc corpus is just
**2.3 docs/day across the whole market**. Where we DO have a same-day directional edge, the
sign is right **66%**. So coverage is bounded by how much news we ingested, not by the method.
**The 446k-doc Bloomberg corpus (already in hand) is the coverage unlock (~75× density) — a
data-scale move, not a new model.** This is the single highest-leverage next step.

### Concentrated coverage — recent DENSE corpus as a cheap test set
Rather than ingest the 446k backlog, `concentrated_coverage.py` uses the live corpus
(**150 docs/day over 20 days**, June–July 2026) and measures coverage only inside that
window (abnormal return vs liquid Yahoo factor proxies, since the engine's 9-factor panel
ends 2026-06-23). Result — coverage is a **density dial**, and signal quality is an
**instrument-mix** question:

| kind | coverage | same-day precision | contemp IC |
|---|---|---|---|
| **single-name security** | 58/253 (23%) | 54% | **+0.165 (t 2.2)** |
| macro/country ETF proxy | 36/74 (49%) | 46% | +0.075 (t 0.9) — null |
| all | 94/327 (29%) | 50% | +0.095 (t 1.7) |

Two lessons: (1) density lifts coverage 1.1% → 29% as predicted; (2) the directional signal
lives in **single-name securities** (IC +0.165, t 2.2 — matches the historical +0.128), and
is **null on macro ETF proxies** (their residual-vs-market is noise, macro news direction is
ambiguous). The 2026 corpus looked null in aggregate only because the 8b model emits no
tickers, so macro/geopolitics entities route to country ETFs and dilute the securities signal.
**Ticker/company-resolution quality is therefore a first-order lever, not just corpus scale.**

### How much value is in the signal? (measured, `signal_value.py`)
On the rigorous 9-factor historical set, split by instrument:
- **Attribution/nowcast value (robust):** when the graph names a driver for a security, the
  stock shows an expected **+0.46% same-day abnormal move in the claimed direction (t 3.2)**;
  big-mover directional **precision 62%**; and flagged events are **large (~7% avg abnormal)**.
- **Cross-sectional R² ≈ 1%** — news explains ~1% of idiosyncratic-move variance (small per-name,
  scales with coverage).
- **Tradable forward alpha: NOT established** — next-day follow-the-signal edge is +0.17%/event
  but t=1.3; the implied +44%/yr / Sharpe 0.82 is optimistic, un-costed, statistically fragile.
  **The value is nowcast/attribution, not alpha** — do not build a trading product on it yet.

### Resolution doctrine — US-listed clean bars only (`resolve_yahoo.py`)
Improving ticker resolution (Yahoo search over unresolved company/bank names, verified by
quoteType==EQUITY + price existence; no LLM) grew the securities bucket, but a US-vs-foreign
split proved a sharp rule:

| kind | contemp IC | precision |
|---|---|---|
| **security_us** (US-listed / ADR) | **+0.16–0.18 (t 2.3–2.6)** | 60% |
| security_fx (foreign local .KS/.NS/.T) | ~0 / negative | noise |
| etf_proxy (macro/country) | +0.07 (null) | 46% |

Foreign LOCAL listings misalign with US-session news timing + the US-proxy factor model → pure
dilution. **Resolve to US listings / ADRs; never the local line.** With US-preference, the
aggregate recent-corpus IC became significant (t 2.5) where it had been diluted to null.

**Ticker-emitting pass (`resolve_llm.py`)** — the 8b emits names not tickers, so a capable model
(gpt-oss-120b) proposes a ticker + issuer from headline context, and an anti-hallucination gate
keeps it ONLY if it verifies against Yahoo (real price history AND the ticker's actual issuer name
matches the model's claim). This got the **brand/subsidiary→listed-parent** mappings nothing else
can — waymo→GOOGL, whatsapp→META, taco bell→YUM, peacock/nbc→CMCSA, zoox/AWS→AMZN, intel foundry→
INTC — and the gate rejected the hallucinations/private names (kalshi→a bogus KAL, cxmt, jera→wrong
line) plus returned listed=false for ~158/190 (private/gov/index/crypto). Net across all passes:

| resolution stage | company/bank edges → a security |
|---|---|
| SEC + OpenFIGI | 348/675 (52%) |
| + Yahoo search | 435/675 (64%) |
| + LLM ticker-emit (verified) | **459/675 (68%)** |

Caveat measured honestly: the LLM-added names are low-vol large-caps, so on a *calm 20-day* window
they add few |z|>1.5 movers and the IC barely moves (+0.158→+0.163); their value is **attribution
correctness** (Waymo news → Alphabet, not dropped), which compounds over longer windows / single-
name-heavy corpora. The remaining unresolved ~32% is largely genuinely non-tradeable.

### Novelty shape (route to any predictive signal)
Bucketing signals by prior-30-day mentions of the name: brand-**new** news has a strong
contemporaneous IC (+0.13, t 3.9) but a **null lag+1** (priced same-day); **emerging**
(1–2 prior) stories carry the only positive forward hint (lag+1 IC +0.088, t 1.4 — suggestive,
not significant); **established** (3+) news mean-reverts (lag+1 −0.075). If a tradable edge
exists, it is in *developing* stories, and it is small — consistent with efficient markets.

Per-mechanism **name-level** reliability vs the **pooled** scalar we had been using —
the pooled average destroyed conditional signal:

| mechanism | name-level mean(dir·z) | t | pooled S_dir |
|---|---|---|---|
| monetary_policy | +0.43 | 4.1 | +0.15 |
| competition | +1.13 | 4.0 | ~0 |
| rating_action | +1.02 | 3.4 | +0.21 |
| demand_change | +0.30 | 3.1 | +0.17 |
| regulation | +0.17 | 2.6 | +0.08 |
| geopolitics | +0.26 | 1.6 | +0.03 |

Geopolitics' pooled 0.03 was hiding a conditional ~0.26 — it averages bullish
(defense/oil) against bearish (risk-off) to nothing. The signal is **conditional**, not
marginal.

## Robust points — build the product on these
- **Contemporaneous attribution** (IC +0.128, t 4.5; precision 61%). The graph reliably
  says *why a name moved*.
- **Conditional, per-(mechanism × name) reliability** — large where the pooled scalar was zero.
- **Measured per-name sensitivities** (engine betas) — economically sane.
- **Provenance-gated extraction** — the drivers we name are grounded, not hallucinated.

## Weak points — stop doing these
1. **Aggregation into market factors** — cancels idiosyncratic drivers by construction. *The core error.*
2. **Pooled scalar S_dir** — marginalizes over the context that carries the information.
3. **Forecast framing on daily bars** — daily t+1 is null (efficient pricing); most news
   is priced in minutes, so a daily crystal ball is the wrong promise.
4. **No novelty/saliency filter** — first-mention/consensus-deviating news drives;
   restatement and color don't. Pooling them dilutes every driver.

## Objectives (what to measure = what has meaning)

| Objective | Metric | Status |
|---|---|---|
| Cross-sectional explanatory power | name-level IC + hit rate, contemp & lagged | ✅ built (`cross_sectional_ic.py`) |
| Attribution coverage & precision | of \|z\|>2 movers: % explained, % sign-correct | ◑ precision done; full coverage needs the whole price panel |
| Conditional reliability | mean(dir·z) per (mechanism × entity-type × sign) | ✅ mechanism cut done; entity-type/sign next |
| Novelty lift | S_dir(first-mention) − S_dir(restated) | ☐ needs a novelty score on edges |
| Network propagation | hub-stress → neighbours' abnormal-return elasticity | ☐ |
| Market-neutral driver IR | long +driver / short −driver names, beta-hedged | ☐ the real "is it tradable" test |

## Attribution surface — BUILT (`attribution.py`)
The shippable product on the validated US single-name signal, all views provenance-backed
(every driver carries its verbatim quote + source doc), event-study window [d−1, d]:
- **`movers`** — biggest abnormal US-security movers in the window, each attributed to its top
  news driver, with the headline coverage stat ("we attribute X% of movers"). On eg_live: 17%
  (corpus-density-bound), ✓/✗ flags show when the edge direction agrees with the realised move.
- **`explain TICKER`** — a name's abnormal days, each with ranked drivers (mechanism · reliability
  tag · cause → direction · confidence · verbatim quote). E.g. GS +7.96% ✓ bank-strength/AI-boom.
- **`drives TICKER`** — the name's driver profile: mechanisms & cause-entities that recur,
  annotated with measured name-level reliability (e.g. NFLX ← Carl Rinsch fraud, Reed Hastings).
Honestly surfaces 8b extraction quality (a $53B PayPal bid mis-tagged "down" shows as ✗opposes),
so the reliability tags + agreement flags let a user discount weak/noisy drivers.

## Products to present to users (not "the tail shifted −0.0004")
1. **Explain this move** — a name moved; rank its news drivers with confidence + provenance quote. *(Robust today.)*
2. **What drives this name** — per-instrument driver profile: which mechanisms/entities/hubs it's historically sensitive to, with measured reliabilities. *(Measurable now.)*
3. **What's moving today and why** — today's biggest abnormal movers, each attributed, plus the headline coverage stat "we explain X% of today's movers."
4. **Novel-driver alert** — surface *new* drivers for a name (first mention / consensus break) — where any forward information lives.
5. **Contagion map** — a hub is stressed → which downstream names are at risk (the graph's structural edge over a flat model).

## What to measure next (priority order)
1. ✅ **DONE — ticker/company resolution** (`resolve_yahoo.py`): +75 US-preferred securities;
   proved the US-listed-only doctrine and lifted the aggregate IC to significant (t 2.5). Remaining:
   ~half the unresolved tail is genuinely non-tradeable (private/political/crypto); the tradeable
   remainder needs a ticker-emitting extraction pass (or a US-large-cap-focused corpus) to grow further.
2. **Scale the corpus — run the full 446k BBG docs** (already in hand). Coverage is a density dial
   (1.1% sparse → 29% dense); scale lifts it further and thickens the cross-section for a real
   daily IC / IR. A data-scale move, not a new model.
2. **Ship the attribution product** ("explain this move", "what drives this name") on the validated
   contemporaneous signal — it does not need the forecast to work.
3. **Conditional S_dir table** on entity-type × sign, then **push onto edges** so drivers rank by
   *conditional* reliability, not the pooled scalar (monetary_policy/competition/rating_action are
   far stronger at name level than pooled).
4. **Emerging-story forward signal**: the only predictive hint (lag+1 +0.088) is in developing
   stories — re-test at BBG scale with a proper novelty/attention score; if it survives, that is the
   tradable sliver.
5. **Intraday** returns to test whether the daily forecast null is efficiency or mismeasurement.
6. **Market-neutral driver book** as the decisive tradability test (aggregation can't cancel a spread).
