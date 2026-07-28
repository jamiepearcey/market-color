# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
WHICH NEWS TOPICS DURABLY CHANGE A COMPANY'S RISK PROFILE?

THE DISTINCTION THIS TESTS. Everything measured so far is a SPIKE: abnormal
volume/vol jumps on the news day and decays (vol_z 0.42 on the day -> 0.18 by
T+1). A spike is a transient. A different and more useful question is whether an
event class shifts the company's risk LEVEL durably — a regime change rather
than a blip. Plausibly some do (a credit event or major litigation may
permanently re-rate a name's volatility) while others are pure transients
(earnings: big spike, back to normal within days).

    spike        vol on the event day vs normal
    regime shift vol over [T+1, T+21] vs [T-21, T-1]   <- this script

This is a DESCRIPTIVE / contemporaneous question, which is the register where
news has repeatedly been shown to work here — not a claim of incremental
predictability over price, which has been tested and is null.

MEASURE. Per event, the log ratio of post- to pre-event idiosyncratic vol
(market-model residuals, so this is company-specific risk, not market moves):

    shift_i = log( std(eps over [T+1, T+21]) / std(eps over [T-21, T-1]) )

Positive = risk durably higher after the event. Aggregated per event class.

CONTROL (the discipline this project arrived at the hard way). The same
statistic on RANDOM non-event days for the same tickers gives the empirical
noise floor: any class must clear that floor to count. Vol is mean-reverting and
autocorrelated, so a naive "post > pre" reading would otherwise pick up ordinary
vol dynamics rather than an event effect. Also reported: a bootstrap CI per
class, and the persistence share (fraction of events with a positive shift).

Usage:
    uv run scripts/news_risk_regime_change.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
PRE, POST = 21, 21
BETA_WIN = 250
RNG = np.random.default_rng(20260726)
MIN_EVENTS = 40


def returns_series(sym: str, cache: Path) -> dict[str, float]:
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or ind["quote"][0]["close"]
    except Exception:
        return {}
    dd, px = [], []
    for t, c in zip(ts, cl):
        if c and c > 0:
            dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
    return {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}


def main() -> None:
    # ---- news: ticker -> {day: set(std_event_type)} ----
    ent = {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        if j.get("resolution_status") == "resolved_security" and j.get("resolved_ticker"):
            ent[j["entity_id"]] = j["resolved_ticker"].upper()
    docday = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])
    events = collections.defaultdict(set)          # (ticker, day) -> types
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if e in ent and d:
            events[(ent[e], d)] |= (doctypes.get(doc) or {"other"})

    # ---- prices -> idiosyncratic residuals ----
    cache = G / "prices"
    tickers = {tk for tk, _ in events}
    rets = {}
    for tk in tickers:
        r = returns_series(tk.replace("/", "-"), cache)
        if len(r) > 600:
            rets[tk] = r
    daycount = collections.Counter()
    for r in rets.values():
        daycount.update(r.keys())
    thresh = max(30, int(0.30 * len(rets)))
    alldays = sorted([d for d, c in daycount.items() if c >= thresh])
    dayidx = {d: i for i, d in enumerate(alldays)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])

    resid = {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < 600:
            continue
        e = np.full(len(alldays), np.nan)
        for i in range(BETA_WIN, len(alldays)):
            sl = slice(i - BETA_WIN, i); m = ok[sl]
            if m.sum() < 150:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean(); den = float(fc @ fc)
            if den > 0 and ok[i]:
                e[i] = y[i] - float((yy - yy.mean()) @ fc / den) * F[i]
        resid[tk] = e
    print(f"tickers with idiosyncratic residual series: {len(resid)}")

    def shift_at(tk, i):
        """log(post-vol / pre-vol) of idiosyncratic returns around index i."""
        e = resid[tk]
        pre = e[i - PRE:i]; post = e[i + 1:i + 1 + POST]
        pre = pre[np.isfinite(pre)]; post = post[np.isfinite(post)]
        if len(pre) < PRE - 3 or len(post) < POST - 3:
            return None
        a, b = pre.std(), post.std()
        if not (a > 0 and b > 0):
            return None
        return math.log(b / a)

    # ---- observed shifts per event class ----
    by_type = collections.defaultdict(list)
    used = 0
    for (tk, d), types in events.items():
        if tk not in resid:
            continue
        i = dayidx.get(d)
        if i is None or i < BETA_WIN + PRE + 5 or i + POST + 2 >= len(alldays):
            continue
        s = shift_at(tk, i)
        if s is None:
            continue
        used += 1
        for t in types:
            by_type[t].append(s)
    print(f"events with a measurable pre/post window: {used}")

    # ---- control: random NON-event days, same tickers ----
    evdays = {(tk, d) for (tk, d) in events}
    ctrl = []
    tks = [t for t in resid]
    for _ in range(6000):
        tk = tks[RNG.integers(len(tks))]
        i = int(RNG.integers(BETA_WIN + PRE + 5, len(alldays) - POST - 2))
        if (tk, alldays[i]) in evdays:
            continue
        s = shift_at(tk, i)
        if s is not None:
            ctrl.append(s)
    ctrl = np.array(ctrl)
    lo, hi = np.percentile(ctrl, [2.5, 97.5])
    print(f"control (random non-event days): n={len(ctrl)} mean={ctrl.mean():+.4f} "
          f"95% band [{lo:+.4f}, {hi:+.4f}]  <- the empirical noise floor")

    def boot_ci(x, b=3000):
        x = np.array(x)
        m = np.array([x[RNG.integers(0, len(x), len(x))].mean() for _ in range(b)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print(f"\n=== DURABLE risk-profile change by event class "
          f"(log post/pre idiosyncratic vol, ±{PRE}d) ===")
    print(f"  {'event class':18} {'n':>6} {'mean shift':>11} {'95% CI':>20} "
          f"{'% up':>6}  vs control")
    rows = [(t, v) for t, v in by_type.items() if len(v) >= MIN_EVENTS]
    for t, v in sorted(rows, key=lambda kv: -np.mean(kv[1])):
        a = np.array(v)
        clo, chi = boot_ci(a)
        beats = "YES" if clo > ctrl.mean() else ("—" if chi > ctrl.mean() else "below")
        print(f"  {t:18} {len(a):6d} {a.mean():+11.4f} [{clo:+.4f}, {chi:+.4f}] "
              f"{100 * np.mean(a > 0):5.0f}%  {beats}")

    print("\n  'vs control' = YES when the class's bootstrap CI sits entirely ABOVE the\n"
          "  random-day mean, i.e. the durable risk increase is not just ordinary vol\n"
          "  dynamics. A spike (measured elsewhere: earnings +1.12 vol-z on the day)\n"
          "  is NOT the same as a durable shift — a class can spike hard and revert.")


if __name__ == "__main__":
    main()
