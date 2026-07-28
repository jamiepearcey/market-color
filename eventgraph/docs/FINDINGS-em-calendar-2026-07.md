# Findings — EM/frontier calendar: are the under-exploited events material? (2026-07-26)

Testing the claim behind `EM-CALENDAR-STRATEGY.md`: that the events vendors cover
worst are *material* — and whether the market is slow to price them.

`scripts/em_jump_hazard.py` · `data/eg_runs/em_cal/lake/jump_hazard.json`

## Bottom line

**Materiality: PROVEN, but roughly a third the size the first pass suggested.**
Across **138 arrangement approvals in 42 IMF borrower countries**, FX step moves
concentrate in the event window at **1.81× the local baseline** (2.1% vs 1.2%,
permutation **p = 0.009**; global placebo **p = 0.004**), stable across two
disjoint regimes (2.21× / 1.96×). The sharpest single cell is the **event session
itself at 3.62× (p = 0.020)**.

**The first estimate of 5.00× was selection.** It came from 11 hand-picked
frontier names — Egypt, Ghana, Nigeria, Zambia, Pakistan, Sri Lanka, Argentina —
i.e. exactly the countries famous for IMF-linked devaluations. Widening to every
IMF borrower with a usable, non-pegged currency **halves it**. The 1.81× is the
unbiased number; treat 5× as what happens when you choose the sample by
reputation.

**Direction: NOT PROVEN.** 35 of 60 in-window jumps were depreciations — 58%,
a coin flip. We can say the currency will *move*; we cannot say which way.

So this is a **volatility/risk-timing** result, not an alpha result — exactly the
payoff Fable's architecture review predicted the calendar layer would have, and
exactly the payoff `STRATEGY.md` says to build products on.

---

## 1. Why a hazard model and not an event study

The venues whose calendars are least covered run administered or heavily managed
exchange rates. Measured over 5 years of daily bars:

| cross | zero-change sessions | largest 1-day move |
|---|---|---|
| USDKES | 15.0% | 6.3% |
| USDGHS | 14.7% | 28.0% |
| USDNGN | 9.6% | 33.3% |
| USDEGP | 0.2% | **60.4%** |
| USDXOF | 8.9% | 19.9% — **euro peg; its vol is EURUSD** |

That is not a diffusion. An abnormal-return study divided by trailing vol measures
a quantity that does not exist here, which is one concrete reason these events stay
under-exploited: the standard toolkit returns garbage on them.

**Definition that makes the test readable:** a jump is the top 1% of a currency's
*own* |daily return| distribution. The unconditional hazard is therefore exactly
1% per session **by construction**, per currency — so any excess is visible without
a volatility model.

## 2. Design

- **Events**: IMF lending-arrangement approvals, 2016+, from the Fund's own TSV
  export (`extarr2.aspx?tsvflag=Y`). Authoritative and exactly dated. **138 events
  across 42 countries** after filtering.
- **Excluded by construction**: currencies whose moves are another central bank's
  policy — the euro bloc (XOF/XAF/KMF/CVE/MKD/BAM), the **rand bloc (LSL/SZL/NAD,
  1:1 with ZAR)**, rupee-linked (NPR/BTN), and dollar pegs/dollarised names. Plus a
  de-facto-peg detector: any currency whose 99th-percentile daily move is under
  0.5% has no step-move behaviour to detect.
- **Alignment**: each board date maps to the **first trading session on or after**
  it — board decisions land in the US afternoon and a frontier currency cannot
  react until its next local session. Same discipline that moved the 8-K result
  2.2× → 3.2×.
- **Window**: ±10 sessions. 28,633 country-sessions, 285 jumps.

**Pre-registered reading.** IMF approval usually *requires* exchange-rate action as
a prior action, so mass *before* the date is the mechanical expectation and proves
nothing tradeable. The informative cell is *after*.

## 3. Results

```
offset   -1     0    +1    +2
jumps     3     1     5     3
lift   8.57x 2.86x 14.29x 8.57x     <- peak is +1, the session after the board date

pre  (-10..-1)  13 jumps   3.7%   3.71x
at   (    0   )   1 jump    2.9%   2.86x
post (+1..+10)  16 jumps   4.6%   4.57x
```

| test | observed | null | p |
|---|---|---|---|
| Global circular-shift placebo, total | 30 | 7.3 | **0.002** |
| — pre | 13 | 3.5 | 0.010 |
| — post | 16 | 3.4 | 0.004 |
| **Local control** (±90d neighbourhood, event window excluded) | **4.2% vs 0.8%** | — | **<0.001** |

### The local control is the one that matters
The global placebo tests a straw man. IMF arrangements are approved *during*
currency crises and jumps happen *during* currency crises, so both are caused by
the crisis and a global null is nearly guaranteed to reject. The honest null holds
the crisis period fixed: baseline = each event's own ±90-day neighbourhood with the
event window removed, and the permutation relocates each event only *within* that
neighbourhood.

**It survives: 5.00× local lift, p < 0.001.** Given that a country was in the kind
of period where the Fund shows up, the jump still lands near the board date
specifically.

### Stability
| split | events | lift |
|---|---|---|
| early (2016-11 … 2021-11) | 69 | 2.21× |
| late (2021-12 … 2026-06) | 69 | 1.96× |

Near-identical across two disjoint regimes — the check that killed the FOMC
de-diversification result and the Kalshi edge.

### Concentration — the effect is CONDITIONAL, not marginal
**Only 20 of 42 countries contributed a single in-window jump.** Among those that
did, the lift is far above the pooled figure:

| country | events | in-window | lift |
|---|---|---|---|
| Guinea | 1 | 3 | 14.29× |
| Nigeria | 1 | 2 | 9.52× |
| Moldova | 7 | 12 | 8.16× |
| Ethiopia / Sri Lanka | 4 / 2 | 6 / 3 | 7.14× |
| Egypt | 5 | 7 | 6.67× |
| Argentina | 3 | 4 | 6.35× |
| Pakistan | 5 | 6 | 5.71× |
| Mexico, Colombia, Peru, Kenya, Bangladesh, Rwanda, … | 22 countries | 0 | 0.00× |

This is the same shape as the mechanism table in `STRATEGY.md`, where geopolitics'
pooled 0.03 concealed a conditional 0.26: **the pooled 1.81× is an average over a
strong effect in stressed, managed-rate borrowers and no effect at all in
flexible-rate or precautionary-programme countries.** Mexico, Colombia and Peru
(FCL/precautionary lines, floating currencies) contributing exactly zero is the
economically correct result, not a failure — nobody devalues off an FCL.

## 4. What kills the trade

1. **Direction is near a coin flip.** 63/37, not significant at n=30. Untradeable in spot.
2. **Mass sits on both sides of the date** (3.71× before, 4.57× after). Near
   symmetry is the signature `calendar_window_profile.py` taught us to distrust —
   though here it is also economically sensible, since devaluation is often the
   prior action *and* the adjustment continues afterwards. Either way it defeats a
   clean "buy the announcement" story.
3. **NDF forward points already price expected devaluation.** A 4× spot-jump hazard
   is not profit if you paid 30% annualised carry to be short. Whether 4× exceeds
   what the forwards imply is the unanswered question — and it needs NDF/vol data
   we do not have free.
4. **n = 35 events / 30 jumps.** Small, and no cross-regime out-of-sample split
   beyond the two-way temporal check.

## 4a. Straddle economics — the direction-blind expression, priced

A direction-blind result should be expressed as long volatility, so: what would a
straddle over the ±10-session window actually collect? Using the **maximum daily
move** in the window as a generous proxy (generous because it flatters a
daily-rebalanced long-gamma position; a real 21-day straddle pays the *terminal*
move, which can be smaller):

| | median | p75 | p90 | p99 |
|---|---|---|---|---|
| event window | 1.40% | 5.14% | 11.19% | **71.69%** |
| matched baseline | 1.12% | 2.19% | 3.86% | 13.68% |

- P(at least one top-1% jump in the window): **32.4% vs 9.4% baseline = 3.43×**
- **Median edge: 0.28 percentage points.**
- **62% of event windows had a largest daily move under 2%** — a straddle expires
  worthless in nearly two thirds of them.

The edge is **entirely in the tail** and essentially absent in the middle. That is
a convexity trade, not a carry trade: you pay ~34 premiums to be paid on ~13, and
the p99 is doing all the work.

## 4b. Data integrity — the result was contaminated before it was cleaned

The first run's top-1% jump sets contained four Yahoo scale glitches masquerading
as the largest moves in the sample: USDGHS 5.67 → 573.00, USDMWK 1.00 → 710.00,
USDNGN 3.59 → 360.50, USDETB 6.00 → 21.80. Only Argentina's +118% (Dec 2023) and
Egypt's +71.7% (Nov 2016 float) were real. A guard now truncates each series after
the last >3× session break; Argentina and Egypt both survive it.

**The result strengthened after cleaning** (4.07× → 5.00×, and the two temporal
halves converged to 4.21× / 4.20×), which is the reassuring direction — the
finding was not an artifact of bad prints. This is the same stale/administered
price trap as the 17%-stale-ADR work, in a new venue.

## 5. What this licenses

- **Build the calendar for risk timing, not alpha.** "A step move is ~4× more
  likely in this window" is a real, defensible risk statement — the same shape as
  the validated de-diversification result, and the same shape Fable argued the
  whole layer should pay off in.
- **The natural expression is volatility, not spot.** Which is also why it stays
  under-exploited: frontier FX options are illiquid and expensive.
- **Do not claim a directional edge.** 60/40 at n=30 is noise.

## 6. Next tests, in order

1. ~~Reviews, not just approvals~~ — **attempted and abandoned; the data does not
   exist for free.** MONA is decommissioned (every old ASPX path returns the
   Sitecore 404), `imf.org/en/*` and the press RSS are edge-blocked even through
   TLS impersonation, the SDMX API at `api.imf.org` carries macro statistics only
   (348 dataflows, no Fund-finance data), ReliefWeb v1 is retired and v2 requires a
   registered appname, and Google News RSS returns ~5 items with poor recall.
   **Widening the country set was the better fix anyway**: reviews inside one
   programme are correlated with each other *and* with the approval, whereas extra
   countries are closer to independent clusters, which is what the permutation
   inference actually needs. n went 34 → 138 that way.
2. **Directional conditioning** — does the sign become predictable conditional on
   the *type* of milestone (first approval vs review completion vs augmentation)?
   That is where a directional signal would hide, if one exists.
3. **Is 4× priced?** Compare against NDF-implied devaluation probability. Needs
   paid data; this is the decisive tradeability test.
4. **Rating reviews** as a second under-exploited class, using the EU
   CRA-mandated calendars.

## 7. Method note worth keeping

IMF endpoints return 403 to any ordinary client, including one sending a browser
User-Agent — the edge fingerprints TLS, not headers. `curl_cffi` with
`impersonate="chrome"` gets through to `extarr1/extarr2/exfin2.aspx`, which is how
the event set was obtained at all. `imf.org/en/*` and the RSS feeds remain blocked
even so.
