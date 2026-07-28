# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
IMMEDIATE (same-day) RISK BY NEWS TOPIC — the contemporaneous register where
news demonstrably works.

WHY T+0 AND NOT T+1. Everything predictive in this project is null: news adds
nothing over price at T+1 or beyond. But the timing evidence says why — the move
lands ON the news day (vol_z +0.42 on the day vs +0.18 by T+1). So the question
with a real answer is not "what will happen tomorrow" but "given this class of
news just landed, how much risk is being realised RIGHT NOW". That is a
measurement, not a forecast, and it is directly usable for execution sizing,
liquidity planning and intraday risk limits.

THREE same-day measures per event class, all standardised by the name's own
trailing 60-day idiosyncratic vol so they are comparable across tickers:

    |abn move|   |idiosyncratic return| / trailing sigma      (risk magnitude)
    tail rate    P(|idio return| > 2 sigma)                   (jump risk)
    vol-z        abnormal log-volume z                        (attention/flow)

Idiosyncratic = market-model residual (rolling 250d beta on an equal-weighted
market factor), so this is company-specific risk, not the market moving.

CONTROL. Random non-event days for the same tickers give the empirical noise
floor, and every figure is reported as a RATIO to that control. Without it a
reader cannot tell whether "1.3 sigma" is large — the control says what a
nothing-day looks like.

Usage:
    uv run scripts/news_immediate_risk.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
from pathlib import Path

import sys
from pathlib import Path

import numpy as np
from liquidity import tradeable, usable_window  # noqa: E402,F401
import panel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
BETA_WIN = 250
TRAIL = 60
RNG = np.random.default_rng(20260726)
MIN_EVENTS = 40


def returns_and_volume(sym: str, cache: Path):
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}, {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; ind = res["indicators"]; q = ind["quote"][0]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or q["close"]
        vol = q.get("volume") or []
    except Exception:
        return {}, {}
    dd, px, vv = [], [], []
    for t, c, v in zip(ts, cl, vol):
        if c and c > 0:
            dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c)); vv.append(float(v) if v else np.nan)
    rets = {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}
    lv = np.log(np.array([v if v and v > 0 else np.nan for v in vv]))
    vz = {}
    for i in range(TRAIL, len(dd)):
        w = lv[i - TRAIL:i]; w = w[np.isfinite(w)]
        if len(w) > 30 and w.std() > 0 and np.isfinite(lv[i]):
            vz[dd[i]] = float((lv[i] - w.mean()) / w.std())
    return rets, vz


def main() -> None:
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
    events = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if e in ent and d:
            events[(ent[e], d)] |= (doctypes.get(doc) or {"other"})

    # SIGMA COMES FROM THE SHARED CACHE. This script used to re-derive residuals
    # itself, removing only the market factor — while the gate that judges these
    # same events now removes market AND a hierarchical SIC sector factor. The two
    # disagreed by ~5% on the headline class. Reading the cache makes that
    # impossible. Volume-z is still loaded here because the cache does not carry it.
    P = panel.load("us")
    events = {k: v for k, v in P.events.items()}
    alldays = P.days
    dayidx = P.day_index
    cache = G / "prices"
    vzs = {}
    for tk in P.tickers():
        _r, v = returns_and_volume(tk.replace("/", "-"), cache)
        vzs[tk] = v
    print(f"tickers: {len(P.tickers())} (shared cache; market AND sector removed)")

    def measures(tk, i):
        """(|idio move| in sigma, tail flag, abnormal volume z) on day i."""
        z = P.sigma.get(tk, {}).get(str(i))
        if z is None:
            return None
        z = float(z)
        return z, 1.0 if z > 2.0 else 0.0, vzs.get(tk, {}).get(alldays[i], np.nan)

    by_type = collections.defaultdict(list)
    for (tk, d), types in events.items():
        if tk not in P.sigma:
            continue
        i = dayidx.get(d)
        if i is None or i + 2 >= len(alldays):
            continue
        m = measures(tk, i)
        if m is None:
            continue
        for t in types:
            by_type[t].append(m)

    evdays = set(events)
    tks = P.tickers()
    ctrl = []
    for _ in range(9000):
        tk = tks[RNG.integers(len(tks))]
        i = int(RNG.integers(0, len(alldays) - 2))
        if (tk, alldays[i]) in evdays:
            continue
        m = measures(tk, i)
        if m:
            ctrl.append(m)
    C = np.array([[a, b, c] for a, b, c in ctrl])
    c_move = float(np.nanmean(C[:, 0])); c_tail = float(np.nanmean(C[:, 1]))
    c_vz = float(np.nanmean(C[:, 2]))
    print(f"control (random non-event days, n={len(C)}): "
          f"|move|={c_move:.3f}sigma  tail={c_tail:.1%}  vol-z={c_vz:+.3f}")

    def boot(x, b=2000):
        x = np.array(x)
        m = np.array([np.nanmean(x[RNG.integers(0, len(x), len(x))]) for _ in range(b)])
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print(f"\n=== IMMEDIATE (same-day) RISK BY EVENT CLASS ===")
    print(f"  {'event class':18} {'n':>6} {'|move| sig':>11} {'x ctrl':>7} "
          f"{'tail%':>7} {'x ctrl':>7} {'vol-z':>7}")
    rows = [(t, np.array([[a, b, c] for a, b, c in v]))
            for t, v in by_type.items() if len(v) >= MIN_EVENTS]
    for t, A in sorted(rows, key=lambda kv: -np.nanmean(kv[1][:, 0])):
        mv = float(np.nanmean(A[:, 0])); tl = float(np.nanmean(A[:, 1]))
        vz = float(np.nanmean(A[:, 2]))
        lo, hi = boot(A[:, 0])
        star = "*" if lo > c_move else " "
        print(f"  {t:18} {len(A):6d} {mv:11.3f}{star} {mv / c_move:7.2f} "
              f"{tl:7.1%} {tl / max(c_tail, 1e-9):7.2f} {vz:+7.3f}")

    print(f"\n  * = bootstrap 95% CI entirely above the control mean ({c_move:.3f} sigma).")
    print("  x ctrl = ratio to a random non-event day — the number that says whether a\n"
          "  figure is actually large. These are CONTEMPORANEOUS measurements (risk being\n"
          "  realised as the news lands), not forecasts; the predictive versions are null.")


if __name__ == "__main__":
    main()
