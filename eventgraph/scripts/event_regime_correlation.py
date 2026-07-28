# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
EVENT-REGIME CONDITIONAL CORRELATION — the "filtered betas" spike.

Premise (the lesson from the news work): the causal graph never beat raw price
betas; it only made the betas STICK to a named event. So drop the claim that
news adds information and keep the one claim that is defensible and look-ahead
free — that the SECOND-MOMENT structure (betas / correlations) is STATE
DEPENDENT, and a calendar lets you select the states EX ANTE. This is not a new
factor; it is ordinary correlation, conditioned on calendar-defined regimes.

Two guardrails, both learned the hard way, decide whether this is signal or
eventgraph-redux:

  1. NEUTRALISE FIRST. On a day the whole market moves, every name co-moves
     through its market beta — raw event-day correlation is high automatically
     and means nothing. So the unit is the RESIDUAL (abnormal) return after a
     proxy-factor regression (ACWI/TLT/UUP/GLD/EEM, same neutraliser as
     persistence_signal.py). If RESIDUAL correlation ALSO rises in the event
     regime, that is genuine event-specific de-diversification — the thing a
     factor hedge misses.
  2. POOL + TEST OUT OF SAMPLE. Event days are rare, so conditional moments are
     noisy. We pool every instance of one event type, bootstrap over EVENTS
     (not days — the event is the unit), and split-half validate: pairs whose
     residual correlation is elevated on first-half events must stay elevated on
     held-out second-half events, else it is overfit.

Regimes are defined off the calendar in TRADING-DAY space around each event
anchor: pre [-P,-1], impact [0,+I], post [+I+1,+I+POST], baseline = everything
else. The headline is the DELTA vs baseline (corr_impact - corr_baseline) and
the pre->impact->post decay curve — never the level.

What it is NOT: a directional forecast. It predicts a correlation/hedge-failure
regime around a KNOWN date, not which way the surprise breaks.

Calendar source, in priority order:
  * <graph-dir>/lake/calendar_event.jsonl  (the formal plane; series_id joined
    to event_type via event_series.jsonl — populate with formal_calendar.py)
  * --event-dates 2024-01-31,2024-03-20,...   (ad-hoc, one type)
  * --demo-fomc                                (built-in REAL FOMC decision days
                                                2023-2025, so this runs today)

Usage:
  uv run scripts/event_regime_correlation.py --demo-fomc
  uv run scripts/event_regime_correlation.py --graph-dir data/eg_live --event-type fomc
  uv run scripts/event_regime_correlation.py --event-dates 2025-01-29,2025-03-19,2025-05-07 --label cpi
"""
import argparse, json, sys, collections, datetime as dt
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, SKIP  # shared price loader

PROXIES = ["ACWI", "TLT", "UUP", "GLD", "EEM"]  # live factor neutraliser (persistence_signal)

# Built-in demo calendar: REAL FOMC decision (announcement) days. Public,
# scheduled a year ahead => look-ahead free. Replace with the formal plane for
# production. (Verify against the Fed's published schedule before trusting.)
DEMO_FOMC = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

# A liquid, sector-spread universe so a regime shift is visible (rate-sensitives,
# growth, defensives, energy). Used when no graph entity_symbol.jsonl is present.
DEMO_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "CRM",
    "JPM", "BAC", "GS", "WFC", "C", "XOM", "CVX", "COP", "JNJ", "PFE",
    "UNH", "KO", "PG", "WMT", "HD", "CAT", "BA", "DIS", "T", "VZ",
]


def load_universe(gd):
    """Graph securities if present (entity_symbol.jsonl), else the demo set."""
    for p in (gd / "entity_symbol.jsonl", Path("out/entity_symbol.jsonl")):
        if p.exists():
            names = set()
            for l in open(p):
                j = json.loads(l)
                s = j.get("symbol", "")
                if j.get("kind") == "security" and "." not in s and not s.startswith("^") and s not in SKIP:
                    names.add(s)
            if names:
                return sorted(names)
    return list(DEMO_UNIVERSE)


def load_calendar(gd, event_type):
    """Event days for `event_type` from the formal plane (series_id->type)."""
    lake = gd / "lake"
    cal, series = lake / "calendar_event.jsonl", lake / "event_series.jsonl"
    if not cal.exists():
        return []
    etype = {}
    if series.exists():
        for l in open(series):
            j = json.loads(l); etype[j["series_id"]] = j.get("event_type")
    days = []
    for l in open(cal):
        j = json.loads(l)
        if event_type in (None, etype.get(j.get("series_id"))):
            t = (j.get("event_time") or "")[:10]
            if t:
                days.append(t)
    return sorted(set(days))


def fetch_returns(names, cache, p1, p2):
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in names}
    return {s: r for s, r in px.items() if len(r) > 60}


def neutralize(px, prox):
    """Residual daily returns after regression on the proxy factors."""
    pdays = sorted(set.intersection(*[set(prox[p]) for p in prox]))
    pset = set(pdays)
    out = {}
    for s, r in px.items():
        days = [d for d in sorted(r) if d in pset]
        if len(days) < 60:
            continue
        y = np.array([r[d] for d in days])
        X = np.column_stack([np.ones(len(days))] + [[prox[p][d] for d in days] for p in prox])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[s] = dict(zip(days, y - X @ beta))
    return out


def pair_corr(a1, a2, dayset, min_common=6):
    common = [d for d in dayset if d in a1 and d in a2]
    if len(common) < min_common:
        return None
    x = np.array([a1[d] for d in common]); y = np.array([a2[d] for d in common])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_offdiag(ar, names, dayset):
    """Mean pairwise residual correlation over a day-set (the regime statistic)."""
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = pair_corr(ar[names[i]], ar[names[j]], dayset)
            if c is not None:
                vals.append(c)
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def regime_daysets(trading_days, anchors, pre, impact, post):
    """Map event anchor dates to trading-day windows. Returns per-offset day
    lists (offset -> [days]) and the merged pre/impact/post/baseline sets."""
    idx = {d: i for i, d in enumerate(trading_days)}
    n = len(trading_days)
    # snap each anchor to the first trading day on/after it
    anch_idx = []
    for a in anchors:
        k = np.searchsorted(trading_days, a)
        if k < n:
            anch_idx.append(int(k))
    by_off = collections.defaultdict(list)
    for k in anch_idx:
        for o in range(-pre, impact + post + 1):
            j = k + o
            if 0 <= j < n:
                by_off[o].append(trading_days[j])
    pre_s = sorted({d for o in range(-pre, 0) for d in by_off[o]})
    imp_s = sorted({d for o in range(0, impact + 1) for d in by_off[o]})
    post_s = sorted({d for o in range(impact + 1, impact + post + 1) for d in by_off[o]})
    window = set(pre_s) | set(imp_s) | set(post_s)
    base_s = [d for d in trading_days if d not in window]
    return by_off, pre_s, imp_s, post_s, base_s, len(anch_idx)


def bootstrap_impact(ar, names, trading_days, anchors, impact, seed, B=200):
    """Event-level bootstrap of the impact-regime mean correlation (the event is
    the unit of resampling, since #events is the true sample size)."""
    rng = np.random.default_rng(seed)
    n = len(trading_days)
    keys = []
    for a in anchors:
        k = int(np.searchsorted(trading_days, a))
        if k < n:
            keys.append(k)
    if len(keys) < 3:
        return None
    stats = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ds = sorted({trading_days[keys[p] + o] for p in pick for o in range(0, impact + 1) if 0 <= keys[p] + o < n})
        m, _ = mean_offdiag(ar, names, ds)
        if np.isfinite(m):
            stats.append(m)
    return np.array(stats) if stats else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", default="data/eg_live")
    ap.add_argument("--tickers", default=None, help="comma list overriding the universe (e.g. a rate-sensitive cluster)")
    ap.add_argument("--event-type", default=None, help="filter formal calendar to this event_type")
    ap.add_argument("--event-dates", default=None, help="comma list YYYY-MM-DD (ad-hoc calendar)")
    ap.add_argument("--demo-fomc", action="store_true", help="use built-in real FOMC decision days")
    ap.add_argument("--placebo", action="store_true",
                    help="falsification control: replace the calendar with the same number of RANDOM "
                         "dates — a real event effect must vanish here")
    ap.add_argument("--label", default=None, help="name for the ad-hoc event set")
    ap.add_argument("--pre", type=int, default=3)
    ap.add_argument("--impact", type=int, default=1, help="impact window = [0, IMPACT] trading days")
    ap.add_argument("--post", type=int, default=4)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--rates-factor", action="store_true",
                    help="enrich the neutraliser with SHY+IEF so TLT+IEF+SHY span the curve "
                         "(level/slope/curvature) — tests whether the event Δ is idiosyncratic "
                         "de-diversification or just a shared rates-factor loading")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--placebo-dist", type=int, default=0, metavar="B",
                    help="run B random date-sets to build the null distribution of the impact Δ, "
                         "and report where the REAL event Δ falls (the decisive falsification: the "
                         "impact statistic is upward-biased on scattered day-sets, so the event Δ "
                         "must beat the placebo null, not just beat baseline)")
    ap.add_argument("--seed", type=int, default=20260724)
    a = ap.parse_args()
    gd = Path(a.graph_dir)

    # ---- calendar ----
    if a.event_dates:
        anchors = sorted(a.event_dates.split(",")); label = a.label or "custom"
    elif a.demo_fomc:
        anchors = sorted(DEMO_FOMC); label = "fomc (demo)"
    else:
        anchors = load_calendar(gd, a.event_type); label = a.event_type or "all"
        if not anchors:
            sys.exit(f"no calendar at {gd/'lake'/'calendar_event.jsonl'} — populate it "
                     f"(formal_calendar.py) or pass --demo-fomc / --event-dates")
    anchors = [d for d in anchors if d >= a.start]
    if a.placebo:  # falsification: same count of random dates instead of real events
        rng = np.random.default_rng(a.seed + 99)
        lo = dt.date.fromisoformat(a.start).toordinal()
        hi = dt.date.today().toordinal() - 10
        k = len(anchors) or 24
        anchors = sorted({dt.date.fromordinal(int(x)).isoformat() for x in rng.integers(lo, hi, k * 3)})[:k]
        label = f"PLACEBO({k}) vs {label}"

    # ---- prices + neutralise ----
    cache = gd / "prices"; cache.mkdir(parents=True, exist_ok=True)
    p1 = int(dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime.now(dt.UTC).timestamp())
    names = [t.strip().upper() for t in a.tickers.split(",")] if a.tickers else load_universe(gd)
    proxies = PROXIES + ["SHY", "IEF"] if a.rates_factor else PROXIES
    print(f"event set '{label}': {len(anchors)} anchors from {anchors[0]}..{anchors[-1]}")
    print(f"fetching {len(names)} names + {len(proxies)} proxies{' (rates-factor)' if a.rates_factor else ''} ...")
    prox = fetch_returns(proxies, cache, p1, p2)
    if len(prox) < len(proxies):
        sys.exit(f"only {len(prox)}/{len(proxies)} proxies fetched (network?) — cannot neutralise")
    ar = neutralize(fetch_returns(names, cache, p1, p2), prox)
    names = sorted(ar)
    if len(names) < 6:
        sys.exit(f"only {len(names)} names with residual series — need a live network for prices")
    trading_days = sorted(set().union(*[set(v) for v in ar.values()]))
    print(f"{len(names)} names neutralised over {len(trading_days)} trading days "
          f"({trading_days[0]}..{trading_days[-1]})")

    # ---- regimes ----
    by_off, pre_s, imp_s, post_s, base_s, n_ev = regime_daysets(trading_days, anchors, a.pre, a.impact, a.post)
    base_c, base_n = mean_offdiag(ar, names, base_s)
    pre_c, _ = mean_offdiag(ar, names, pre_s)
    imp_c, imp_n = mean_offdiag(ar, names, imp_s)
    post_c, _ = mean_offdiag(ar, names, post_s)

    print(f"\n=== EVENT-REGIME RESIDUAL CORRELATION  [{label}, {n_ev} events]  "
          f"windows pre[-{a.pre},-1] impact[0,{a.impact}] post[{a.impact+1},{a.impact+a.post}] ===")
    print(f"  {'regime':10} {'mean_resid_corr':>16} {'Δ vs baseline':>15}")
    for nm, c in (("baseline", base_c), ("pre", pre_c), ("impact", imp_c), ("post", post_c)):
        d = "" if nm == "baseline" else f"{c-base_c:+.3f}"
        print(f"  {nm:10} {c:16.3f} {d:>15}")

    # ---- decay curve across offsets ----
    print(f"\n  offset decay (mean resid corr by trading-day offset from event):")
    line = "   "
    for o in range(-a.pre, a.impact + a.post + 1):
        c, _ = mean_offdiag(ar, names, sorted(set(by_off[o])))
        tag = "E" if o == 0 else f"{o:+d}"
        line += f"{tag}:{c:+.2f}  "
    print(line)

    # ---- event-level bootstrap of the impact elevation ----
    boot = bootstrap_impact(ar, names, trading_days, anchors, a.impact, a.seed)
    if boot is not None:
        lo, hi = np.percentile(boot, [5, 95])
        frac = float(np.mean(boot > base_c))
        print(f"\n  impact bootstrap (event-resampled, N={n_ev}): mean {boot.mean():.3f} "
              f"[90% CI {lo:.3f},{hi:.3f}], P(>baseline)={frac:.0%}")

    # ---- placebo null distribution of the impact Δ (the decisive control) ----
    placebo = None
    if a.placebo_dist and not a.placebo:
        rng = np.random.default_rng(a.seed + 7)
        lo = dt.date.fromisoformat(a.start).toordinal(); hi = dt.date.today().toordinal() - 10
        null = []
        for _ in range(a.placebo_dist):
            ds = sorted({dt.date.fromordinal(int(x)).isoformat() for x in rng.integers(lo, hi, n_ev * 3)})[:n_ev]
            _, _, imp2, _, _, _ = regime_daysets(trading_days, ds, a.pre, a.impact, a.post)
            c2, _ = mean_offdiag(ar, names, imp2)
            if np.isfinite(c2):
                null.append(c2 - base_c)
        null = np.array(null)
        real_d = imp_c - base_c
        pctile = float(np.mean(null < real_d))
        placebo = {"b": len(null), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                   "real_delta": real_d, "percentile": pctile, "p_value": float(np.mean(null >= real_d))}
        print(f"\n  PLACEBO NULL (B={len(null)} random date-sets): impact Δ null mean {null.mean():+.3f}, "
              f"95th pct {np.percentile(null,95):+.3f}")
        print(f"    real event Δ {real_d:+.3f}  ->  {pctile:.0%} percentile of null, p={placebo['p_value']:.3f}")
        print("    (real Δ must sit in the RIGHT TAIL, p<0.05, to beat the scattered-day-set bias)")

    # ---- top calendar-conditioned de-diversifying pairs ----
    dedive = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cb = pair_corr(ar[names[i]], ar[names[j]], base_s)
            ci = pair_corr(ar[names[i]], ar[names[j]], imp_s)
            if cb is not None and ci is not None:
                dedive.append((names[i], names[j], cb, ci, ci - cb))
    dedive.sort(key=lambda x: -x[4])
    print(f"\n  top {a.top} pairs that RE-CORRELATE under the event (base -> impact):")
    for s1, s2, cb, ci, dl in dedive[:a.top]:
        print(f"    {s1:5}~{s2:5}  {cb:+.2f} -> {ci:+.2f}   Δ{dl:+.2f}")

    # ---- split-half OOS persistence ----
    mid = anchors[len(anchors) // 2]
    a1 = [d for d in anchors if d < mid]; a2 = [d for d in anchors if d >= mid]
    persist = None
    if len(a1) >= 3 and len(a2) >= 3:
        _, _, i1, _, _, _ = regime_daysets(trading_days, a1, a.pre, a.impact, a.post)
        _, _, i2, _, _, _ = regime_daysets(trading_days, a2, a.pre, a.impact, a.post)
        # pairs selected as elevated on the FIRST half, tested on the SECOND
        sel = []
        for i in range(len(names)):
            for j in range(len(names)):
                if j <= i:
                    continue
                cb = pair_corr(ar[names[i]], ar[names[j]], base_s)
                c1 = pair_corr(ar[names[i]], ar[names[j]], i1)
                if cb is not None and c1 is not None and (c1 - cb) > 0.15:
                    sel.append((names[i], names[j], cb))
        held = []
        for s1, s2, cb in sel:
            c2 = pair_corr(ar[s1], ar[s2], i2)
            if c2 is not None:
                held.append(c2 - cb)
        if held:
            persist = (len(sel), len(held), float(np.mean(np.array(held) > 0)), float(np.mean(held)))
            print(f"\n  OOS split-half (link<{mid}, test>= {mid}): {persist[0]} pairs elevated in H1; "
                  f"{persist[2]:.0%} STAY elevated in H2, mean H2 Δ {persist[3]:+.3f}")
            print("  (>50% and positive mean = the calendar-conditioned de-diversification persists)")

    out = {
        "label": label, "n_events": n_ev, "universe": len(names),
        "windows": {"pre": a.pre, "impact": a.impact, "post": a.post},
        "regime_corr": {"baseline": base_c, "pre": pre_c, "impact": imp_c, "post": post_c},
        "delta_impact": imp_c - base_c,
        "decay": {str(o): mean_offdiag(ar, names, sorted(set(by_off[o])))[0] for o in range(-a.pre, a.impact + a.post + 1)},
        "top_dedive": [{"pair": [s1, s2], "base": cb, "impact": ci, "delta": dl} for s1, s2, cb, ci, dl in dedive[:a.top]],
        "oos_persistence": None if persist is None else {"n_sel": persist[0], "n_held": persist[1], "frac_persist": persist[2], "mean_h2_delta": persist[3]},
        "placebo_null": placebo,
    }
    dest = gd / "event_regime_correlation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")
    print("  HONEST READ: a positive impact Δ that survives the bootstrap AND the OOS split is a "
          "real, calendar-timed hedge-failure signal — NOT a directional forecast. A null Δ means "
          "the event regime is just the baseline (news redux); learn it cheaply and move on.")


if __name__ == "__main__":
    main()
