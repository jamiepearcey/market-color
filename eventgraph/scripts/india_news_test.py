# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
INDIA (emerging market) news test — the setting where news SHOULD matter most.

Every prior news study here ran on the ~679 most liquid US large caps: the most
analyst-covered, most efficiently priced securities in the world, i.e. the
hardest possible test. The recurring caveat was that news should matter more
where coverage is thin and price discovery is slower. India 2021 is that
setting, and it is now testable because `india_resolve_verified.py` produced 265
VERIFIED entity->NSE mappings (the previous `nifty50_resolved.json` was
partially hallucinated and is not used).

Same method as the US tests so the comparison is apples-to-apples:
  target    abnormal log-volume z at T+h  (and realised vol, second moment)
  baseline  volume-z (or trailing-vol) lags + weekday   <- deliberately strong
  test      INCREMENTAL out-of-sample R2 from news, on a strictly later split,
            conditional on news days

If the "news matters more in less efficient markets" hypothesis is right, the
incremental should be materially positive here even though it was ~0 in the US.
If it is ~0 here too, that is a much stronger negative than the US result alone,
because it removes the efficiency explanation.

Usage:
    uv run scripts/india_news_test.py
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

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "india2021"
SPLIT = "2021-09-01"  # temporal train/test boundary within the 2021 corpus
HORIZONS = [1, 5]


def series(sym: str) -> tuple[dict[str, float], dict[str, float]]:
    """(day -> log-volume z vs trailing 60, day -> daily log return)."""
    p = G / "prices" / f"{sym}.json"
    if not p.exists():
        return {}, {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or q["close"]
        vol = q.get("volume") or []
    except Exception:
        return {}, {}
    days, px, vv = [], [], []
    for t, c, v in zip(ts, cl, vol):
        if c and c > 0 and v:
            days.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
            vv.append(float(v))
    if len(days) < 120:
        return {}, {}
    lv = np.log(np.array(vv))
    vz = {}
    for i in range(60, len(days)):
        w = lv[i - 60:i]
        sd = w.std()
        if sd > 0:
            vz[days[i]] = float((lv[i] - w.mean()) / sd)
    rets = {days[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}
    return vz, rets


def load_news() -> dict[str, dict[str, int]]:
    """ticker -> {day: n_edges}, via the VERIFIED map only."""
    vmap = json.loads((G / "classification" / "india_ticker_map.json").read_text())
    ent2tk = {eid: v["ticker"] for eid, v in vmap.items()}
    docday = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
    news = collections.defaultdict(collections.Counter)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, d = j.get("effect_entity"), docday.get(j.get("doc_id"))
        if e in ent2tk and d:
            news[ent2tk[e]][d] += 1
    return news


def build(news, vz_by, rets_by, horizon, target):
    rows = []
    for tk, vz in vz_by.items():
        days = sorted(vz)
        pos = {d: i for i, d in enumerate(days)}
        rets = rets_by.get(tk, {})
        for d, n in news.get(tk, {}).items():
            i = pos.get(d)
            if i is None or i < 5 or i + horizon >= len(days):
                continue
            lags = [vz[days[i - k]] for k in range(5)]
            if not all(math.isfinite(v) for v in lags):
                continue
            if target == "vol":
                y = vz[days[i + horizon]]
            else:  # realised vol over the next h days (second moment)
                fut = [rets.get(days[j]) for j in range(i + 1, i + 1 + horizon)]
                fut = [f for f in fut if f is not None]
                if len(fut) < horizon:
                    continue
                y = math.log(max(np.std(fut) if horizon > 1 else abs(fut[0]), 1e-8))
            if not math.isfinite(y):
                continue
            wd = [0.0] * 4
            w = int(np.datetime64(d, "D").astype(int) + 4) % 7
            if w < 4:
                wd[w] = 1.0
            rows.append({"day": d, "lags": lags, "wd": wd,
                         "news": [float(n), math.log1p(n)], "y": y})
    return rows


def incremental(rows):
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    if len(tr) < 60 or len(te) < 40:
        return None
    def design(rs, wn):
        return np.array([[1.0] + r["lags"] + r["wd"] + (r["news"] if wn else []) for r in rs])
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    res = []
    for wn in (False, True):
        b = ols_fit(design(tr, wn), ytr)
        res.append(oos_r2(yte, design(te, wn) @ b, ytr.mean()))
    return res[0], res[1], res[1] - res[0], len(tr), len(te)


def main() -> None:
    news = load_news()
    print(f"tickers with verified mapping AND news: {len(news)}")
    tot = sum(sum(v.values()) for v in news.values())
    print(f"total news edges on those tickers: {tot}")

    vz_by, rets_by = {}, {}
    for tk in news:
        vz, rets = series(tk)
        if vz and len(vz) > 100:
            vz_by[tk] = vz
            rets_by[tk] = rets
    print(f"with usable price/volume series: {len(vz_by)}")
    nd = sum(len([d for d in news[tk] if d in vz_by[tk]]) for tk in vz_by)
    print(f"(ticker, news-day) observations: {nd}")

    for target, label in (("vol", "abnormal VOLUME z"), ("rvol", "realised VOLATILITY")):
        print(f"\n=== INDIA: {label} — incremental OOS R2 of news over the baseline ===")
        print(f"  {'h':>3} {'base R2':>9} {'+news R2':>9} {'incremental':>12} "
              f"{'n_train':>8} {'n_test':>7}")
        for h in HORIZONS:
            rows = build(news, vz_by, rets_by, h, target)
            r = incremental(rows)
            if r:
                print(f"  {h:3d} {r[0]:9.4f} {r[1]:9.4f} {r[2]:12.4f} {r[3]:8d} {r[4]:7d}")
            else:
                print(f"  {h:3d}   (insufficient observations)")

    print("\n  Compare with the US large-cap result: incremental ~-0.002 (volume, T+1).\n"
          "  A materially positive number here would support 'news matters more in less\n"
          "  efficient markets'; a ~0 here removes the efficiency explanation entirely.")


if __name__ == "__main__":
    main()
