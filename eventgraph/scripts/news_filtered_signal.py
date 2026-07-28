# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
DOES FILTERING / MERGING THE NEWS RESCUE THE SIGNAL?

THE CHALLENGE THIS ANSWERS. Every previous news test in this series pooled
EVERY causal edge indiscriminately — unfiltered, unweighted, linearly counted.
That is a real methodological gap, and the same dilution error made twice
before (a 99%-news-free panel; a factor-dominated total-vol target): if only a
minority of news days are high-quality market-moving events, pooling them with
low-quality ones estimates a diluted average effect and can hide a real signal.

It matters here because the extraction ITSELF flags quality, and we ignored all
of it:
    modality        happened 98,399 | FORECAST 34,736 (26%!) | hypothetical 246 | denied 231
    direction_audit ok 41,492 | unchecked 86,104 | CONFLICT 6,770
    confidence      strong 85,445 | (missing on ~47k)
    sentiment       intensity strong 37,717 | mild 6,222

Counting a *forecast* or a *denied* event identically to a realised one is
obviously wrong on its face. And the eventgraph project's own surviving result
(correlation persistence: 63-69% for market-confirmed pairs vs 20% unlinked)
came PRECISELY from filtering — so filtering demonstrably matters in this data.

THE TEST: rebuild the news feature under progressively stricter filters and see
whether the INCREMENTAL out-of-sample R2 over a strong price-history baseline
improves. Targets are the two best-specified ones from earlier work:
    idio  = idiosyncratic vol over the next h days (market-model residual)
    vol   = abnormal log-volume z at T+h

FILTER LADDER (each strictly narrower than the last):
    all           every edge                      (what every prior test used)
    happened      modality == happened            (drop forecasts/hypothetical/denied)
    audited       + direction_audit == ok         (extractor-verified direction)
    strong        + confidence == strong          (the cleanest subset)

If the incremental rises as the filter tightens, pooling WAS diluting a real
signal and the earlier nulls are artefacts. If it stays ~0 while N falls, the
redundancy conclusion is robust to filtering — the tape already contains what
even the cleanest news subset would tell you.

Usage:
    uv run scripts/news_filtered_signal.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_volume import ols_fit, oos_r2  # noqa: E402
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
SPLIT = "2011-12-15"
HORIZONS = [5, 21]
BETA_WIN = 250

FILTERS = {
    "all":      lambda j: True,
    "happened": lambda j: j.get("modality") == "happened",
    "audited":  lambda j: j.get("modality") == "happened" and j.get("direction_audit") == "ok",
    "strong":   lambda j: (j.get("modality") == "happened"
                           and j.get("direction_audit") == "ok"
                           and j.get("confidence") == "strong"),
}


def returns_series(sym: str, cache: Path) -> dict[str, float]:
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]
        ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or ind["quote"][0]["close"]
    except Exception:
        return {}
    days, px = [], []
    for t, c in zip(ts, cl):
        if c and c > 0:
            days.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
    return {days[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}


def load_news(pred) -> dict[str, collections.Counter]:
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
    news = collections.defaultdict(collections.Counter)
    kept = 0
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        if not pred(j):
            continue
        e, d = j.get("effect_entity"), docday.get(j.get("doc_id"))
        if e in ent and d:
            news[ent[e]][d] += 1
            kept += 1
    return news, kept


def main() -> None:
    cache = G / "prices"
    base_news, _ = load_news(FILTERS["all"])
    rets = {}
    for tk in base_news:
        r = returns_series(tk.replace("/", "-"), cache)
        if len(r) > 500:
            rets[tk] = r
    print(f"tickers with news + returns: {len(rets)}")

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
            sl = slice(i - BETA_WIN, i)
            m = ok[sl]
            if m.sum() < 150:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean()
            den = float(fc @ fc)
            if den > 0 and ok[i]:
                b = float((yy - yy.mean()) @ fc / den)
                e[i] = y[i] - b * F[i]
        resid[tk] = e

    # abnormal log-volume z (for the second target)
    def volz(sym):
        p = cache / f"{sym}.json"
        if not p.exists():
            return {}
        try:
            res = json.loads(p.read_text())["chart"]["result"][0]
            ts = res["timestamp"]
            v = res["indicators"]["quote"][0].get("volume") or []
        except Exception:
            return {}
        dd, lv = [], []
        for t, x in zip(ts, v):
            if x:
                dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
                lv.append(math.log(x))
        out = {}
        a = np.array(lv)
        for i in range(60, len(dd)):
            w = a[i - 60:i]
            if w.std() > 0:
                out[dd[i]] = float((a[i] - w.mean()) / w.std())
        return out
    vz = {tk: volz(tk.replace("/", "-")) for tk in resid}

    def evaluate(news, h, target):
        Xtr, ytr, Xte, yte = [], [], [], []
        for tk, e in resid.items():
            series = vz.get(tk, {})
            for d, n in news.get(tk, {}).items():
                i = dayidx.get(d)
                if i is None or i < BETA_WIN + 120 or i + 1 + h >= len(alldays):
                    continue
                lags, ok = [], True
                for k in range(5):
                    if target == "idio":
                        w = e[i - 20 * (k + 1):i - 20 * k]
                        w = w[np.isfinite(w)]
                        if not usable_window(w, min_len=10):
                            ok = False
                            break
                        lags.append(math.log(w.std()))
                    else:
                        v = series.get(alldays[i - k])
                        if v is None or not math.isfinite(v):
                            ok = False
                            break
                        lags.append(v)
                if not ok:
                    continue
                if target == "idio":
                    fut = e[i + 1:i + 1 + h]
                    fut = fut[np.isfinite(fut)]
                    if len(fut) < h:
                        continue
                    y = math.log(max(fut.std(), 1e-8))
                else:
                    y = series.get(alldays[i + h])
                    if y is None or not math.isfinite(y):
                        continue
                base = [1.0] + lags
                nf = [float(n), math.log1p(n)]
                (Xtr if d < SPLIT else Xte).append((base, nf))
                (ytr if d < SPLIT else yte).append(y)
        if len(Xtr) < 100 or len(Xte) < 60:
            return None
        def mat(X, wn):
            return np.array([b + (nf if wn else []) for b, nf in X])
        a1, a2 = np.array(ytr), np.array(yte)
        out = []
        for wn in (False, True):
            beta = ols_fit(mat(Xtr, wn), a1)
            out.append(oos_r2(a2, mat(Xte, wn) @ beta, a1.mean()))
        return out[0], out[1], out[1] - out[0], len(Xte)

    for target, label in (("idio", "IDIOSYNCRATIC VOL"), ("vol", "ABNORMAL VOLUME z")):
        print(f"\n=== TARGET = {label} — does filtering raise the incremental? ===")
        print(f"  {'filter':10} {'edges kept':>11} {'h':>3} {'base':>8} {'+news':>8} "
              f"{'incremental':>12} {'n_test':>7}")
        for fname, pred in FILTERS.items():
            news, kept = load_news(pred)
            for h in HORIZONS:
                r = evaluate(news, h, target)
                if r:
                    print(f"  {fname:10} {kept:11d} {h:3d} {r[0]:8.4f} {r[1]:8.4f} "
                          f"{r[2]:12.4f} {r[3]:7d}")

    print("\n  Rising incremental as the filter tightens => pooling WAS diluting a real\n"
          "  signal. Flat ~0 while N falls => the redundancy result is robust to\n"
          "  filtering: the tape already holds what even the cleanest news subset says.")


if __name__ == "__main__":
    main()
