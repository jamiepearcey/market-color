# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
NEWS -> RISK (second moments), the moment we never tested.

Every previous news study here targeted a FIRST-MOMENT-ish quantity: the LEVEL
of abnormal volume. Meanwhile the covariance work established the opposite
lesson — second moments are forecastable, first moments are not. This applies
that lesson to news: not "how much will trade" but "is this name in a
HIGHER-RISK STATE".

THREE TARGETS, all second-moment:
  vol   realised volatility over the next h days (std of daily log returns)
  tail  P(any |return| > 2 trailing-sd) over the next h days — a jump/tail flag
  absr  mean |return| over the next h days (a robust vol proxy)

Against a properly strong baseline, because volatility is highly persistent and
that is what usually eats naive "news predicts risk" claims:
  BASELINE = trailing realised vol (5 lags, log scale) + weekday
Anything news adds must be INCREMENTAL to that, measured out of sample on a
strictly later temporal split, conditional on news days.

AND the dimension-reduction question, asked properly. Previous tests used RAW
event-type indicators. Here we also build a LOW-DIMENSIONAL news state — PCA on
the standardised event-type/mechanism count vector — and ask whether a handful
of components identifies risk. That mirrors the covariance finding that low-rank
factor structure beats everything, applied to the news side.

Feature sets compared:
  base          vol lags + weekday
  +arrival      + news happened / edge count      (does ANY news raise risk?)
  +types        + per-std_event_type indicators   (does WHICH kind matter?)
  +state_k      + top-k PCs of the news count vector (the reduced state)

Usage:
    uv run scripts/news_risk.py
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_volume import ols_fit, oos_r2  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
SPLIT = "2011-12-15"
HORIZONS = [1, 5, 21]
PCA_K = 3


def returns_series(sym: str, cache: Path) -> dict[str, float]:
    """day -> daily log return, from the cached Yahoo chart JSON."""
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
    import datetime as dt
    days, px = [], []
    for t, c in zip(ts, cl):
        if c and c > 0:
            days.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
    return {days[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}


def load_news_typed() -> tuple[dict, list[str]]:
    """(ticker -> {day: Counter(std_event_type)}), sorted event-type vocabulary."""
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
    # doc_id -> std_event_types present that day (from the classification layer)
    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])
    news = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    vocab = collections.Counter()
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if e not in ent or not d:
            continue
        tk = ent[e]
        types = doctypes.get(doc) or {"other"}
        for t in types:
            news[tk][d][t] += 1
            vocab[t] += 1
    keep = [t for t, c in vocab.most_common() if c >= 200]
    return news, keep


def build(news, rets, vocab, horizon, target):
    rows = []
    for tk, r in rets.items():
        days = sorted(r)
        pos = {d: i for i, d in enumerate(days)}
        arr = np.array([r[d] for d in days])
        for d, types in news.get(tk, {}).items():
            i = pos.get(d)
            if i is None or i < 60 or i + horizon >= len(days):
                continue
            trail = arr[i - 60:i]
            sd = trail.std()
            if not (sd > 0):
                continue
            # baseline: log trailing vol at 5 lags (vol is persistent -> strong baseline)
            lags = []
            ok = True
            for k in range(5):
                w = arr[i - 20 * (k + 1):i - 20 * k] if k else arr[i - 20:i]
                s = w.std()
                if not (s > 0):
                    ok = False
                    break
                lags.append(math.log(s))
            if not ok:
                continue
            fut = arr[i + 1:i + 1 + horizon]
            if len(fut) < horizon:
                continue
            if target == "vol":
                y = math.log(max(fut.std(), 1e-8)) if horizon > 1 else math.log(max(abs(fut[0]), 1e-8))
            elif target == "absr":
                y = math.log(max(np.abs(fut).mean(), 1e-8))
            else:  # tail
                y = 1.0 if np.any(np.abs(fut) > 2 * sd) else 0.0
            wd = [0.0] * 4
            w = int(np.datetime64(d, "D").astype(int) + 4) % 7
            if w < 4:
                wd[w] = 1.0
            cnt = np.array([float(types.get(t, 0)) for t in vocab])
            rows.append({"day": d, "lags": lags, "wd": wd,
                         "n": float(sum(types.values())), "types": cnt, "y": y})
    return rows


def run(rows, vocab):
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    if len(tr) < 100 or len(te) < 60:
        return None
    # PCA on the standardised event-type count vector, fitted on TRAIN ONLY.
    Ctr = np.array([r["types"] for r in tr])
    mu, sd = Ctr.mean(0), np.maximum(Ctr.std(0), 1e-9)
    _, _, Vt = np.linalg.svd((Ctr - mu) / sd, full_matrices=False)
    W = Vt[:PCA_K].T

    def design(rs, mode):
        X = []
        for r in rs:
            row = [1.0] + r["lags"] + r["wd"]
            if mode in ("arrival", "types", "state"):
                row += [r["n"], math.log1p(r["n"])]
            if mode == "types":
                row += list(r["types"])
            if mode == "state":
                row += list(((r["types"] - mu) / sd) @ W)
            X.append(row)
        return np.array(X)

    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    out = {}
    for mode in ("base", "arrival", "types", "state"):
        b = ols_fit(design(tr, mode), ytr)
        out[mode] = oos_r2(yte, design(te, mode) @ b, ytr.mean())
    out["n_test"] = len(te)
    return out


def main() -> None:
    news, vocab = load_news_typed()
    print(f"event-type vocabulary (>=200 edges): {vocab}")
    cache = G / "prices"
    rets = {}
    for tk in news:
        r = returns_series(tk.replace("/", "-"), cache)
        if len(r) > 400:
            rets[tk] = r
    print(f"tickers with news + usable return series: {len(rets)}")

    for target in ("vol", "absr", "tail"):
        print(f"\n=== TARGET = {target}  (incremental OOS R2 over a trailing-vol baseline) ===")
        print(f"  {'h':>3} {'base':>8} {'+arrival':>9} {'+types':>8} {'+state':>8} "
              f"{'best incr':>10} {'n_test':>7}")
        for h in HORIZONS:
            rows = build(news, rets, vocab, h, target)
            res = run(rows, vocab)
            if not res:
                print(f"  {h:3d}   (insufficient observations)")
                continue
            incr = max(res[m] - res["base"] for m in ("arrival", "types", "state"))
            print(f"  {h:3d} {res['base']:8.4f} {res['arrival']:9.4f} {res['types']:8.4f} "
                  f"{res['state']:8.4f} {incr:10.4f} {res['n_test']:7d}")

    print("\n  base = trailing realised vol (5 lags) + weekday. Volatility is highly\n"
          "  persistent, so this is a deliberately strong baseline — the same discipline\n"
          "  that made the volume-AR baseline eat the earlier news signal.")


if __name__ == "__main__":
    main()
