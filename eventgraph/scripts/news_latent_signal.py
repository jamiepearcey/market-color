# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sentence-transformers", "torch"]
# ///
"""
LEARNED LATENT NEWS REPRESENTATIONS — can a supervised low-dimensional encoding
of the news text extract signal that hand-crafted features missed?

THE CHALLENGE THIS ANSWERS. Prior tests used hand-crafted news features: edge
counts, per-event-type indicators, event groups, and a PCA on a NINE-dimensional
count vector. That last one was billed as "dimension reduction" but it is a
strawman version of the real idea: it reduces an already-lossy hand-made
summary, not a learned representation of the underlying text.

The proper version: embed the verbatim quotes (all 134,366 edges carry one,
~86 chars), aggregate per (ticker, day), and then learn a LOW-DIMENSIONAL
SUPERVISED encoding — latent directions chosen because they covary with the
TARGET, i.e. "facts representative of higher predictive quality".

  PCA  finds directions of maximum VARIANCE            (unsupervised)
  PLS  finds directions of maximum COVARIANCE WITH y   (supervised)  <- the point

PLS is the right form here, not raw embeddings, because of sample size: ~4k news
days against 384 embedding dimensions would overfit catastrophically. We already
saw the shape of that failure — 9 raw event-type indicators scored -0.115 where
3 PCs scored -0.005. PLS compresses to a handful of components chosen for
predictive covariance, which is exactly the "abstract latent facts" idea made
statistically tractable.

EVERY FIT IS TRAIN-ONLY. The PLS directions, the standardisation, and the
regression are all estimated on the training split and applied unchanged to the
held-out later period. Fitting PLS on all data and then splitting would leak the
target into the representation and manufacture a signal — the single easiest way
to get a fake positive here.

BASELINE remains the strong one: price history (trailing idio-vol or volume-z
lags). The question is INCREMENTAL R2 over that, out of sample.

Backends: hashed character-ngram embeddings by default (numpy only, always
runs); --st uses all-MiniLM-L6-v2 (needs sentence-transformers+torch), the real
semantic test.

Usage:
    uv run scripts/news_latent_signal.py                 # hashed n-gram
    uv run --with sentence-transformers --with torch \\
        scripts/news_latent_signal.py --st               # semantic
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
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
BETA_WIN = 250
HASH_DIM = 256


def hashed_embed(texts: list[str], dim: int = HASH_DIM) -> np.ndarray:
    """Hashed character 4-gram + word unigram bag, L2-normalised. No deps."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        s = " " + (t or "").lower().strip() + " "
        toks = [s[j:j + 4] for j in range(max(len(s) - 3, 0))] + s.split()
        for tok in toks:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
            out[i, h % dim] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def _emb_center(_v):
    """EMB_CENTER=1 -> mean-centre the embedding space (F43/F45). Transformer
    spaces are anisotropic; uncentred cosine is dominated by proximity to the
    corpus centroid, which tracks coverage volume and therefore firm size."""
    import os, numpy as _np
    if not os.environ.get("EMB_CENTER"):
        return _v
    _v = _np.asarray(_v, dtype=float)
    _v = _v - _v.mean(0)
    _n = _np.linalg.norm(_v, axis=1, keepdims=True)
    print("[EMB_CENTER] embedding space mean-centred", flush=True)
    return _v / _np.where(_n > 0, _n, 1)


def embed_st(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_center(np.asarray(m.encode(texts, normalize_embeddings=True, show_progress_bar=False)))


def pls_fit(X: np.ndarray, y: np.ndarray, k: int):
    """Minimal NIPALS PLS1. Returns (W, mu, sd) to project new X into k comps."""
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    Xc = (X - mu) / sd
    yc = y - y.mean()
    W = []
    for _ in range(k):
        w = Xc.T @ yc
        n = np.linalg.norm(w)
        if n < 1e-12:
            break
        w /= n
        t = Xc @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        p = (Xc.T @ t) / tt
        Xc = Xc - np.outer(t, p)
        yc = yc - t * float(t @ yc) / tt
        W.append(w)
    return (np.column_stack(W) if W else np.zeros((X.shape[1], 1))), mu, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st", action="store_true", help="all-MiniLM embeddings")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--target", default="idio", choices=["idio", "vol"])
    a = ap.parse_args()

    # ---- news quotes per (ticker, day) ----
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
    quotes = collections.defaultdict(list)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, d = j.get("effect_entity"), docday.get(j.get("doc_id"))
        q = (j.get("quote") or "").strip()
        if e in ent and d and q:
            quotes[(ent[e], d)].append(q)
    print(f"(ticker, day) cells with quotes: {len(quotes)}")

    # ---- price side ----
    cache = G / "prices"
    def rets_of(sym):
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

    tickers = {tk for tk, _ in quotes}
    rets = {tk: r for tk in tickers if len(r := rets_of(tk.replace("/", "-"))) > 500}
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
    print(f"tickers with residual series: {len(resid)}")

    # ---- assemble rows ----
    h = a.horizon
    rows = []
    for (tk, d), qs in quotes.items():
        if tk not in resid:
            continue
        i = dayidx.get(d)
        if i is None or i < BETA_WIN + 120 or i + 1 + h >= len(alldays):
            continue
        e = resid[tk]
        lags, ok = [], True
        for k in range(5):
            w = e[i - 20 * (k + 1):i - 20 * k]; w = w[np.isfinite(w)]
            if not usable_window(w, min_len=10):
                ok = False; break
            lags.append(math.log(w.std()))
        if not ok:
            continue
        fut = e[i + 1:i + 1 + h]; fut = fut[np.isfinite(fut)]
        if len(fut) < h:
            continue
        rows.append({"day": d, "lags": lags, "text": " ".join(qs[:8]),
                     "n": float(len(qs)), "y": math.log(max(fut.std(), 1e-8))})
    rows.sort(key=lambda r: r["day"])
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    print(f"rows: {len(rows)}  train={len(tr)}  test={len(te)}")
    if len(tr) < 200 or len(te) < 100:
        sys.exit("insufficient observations")

    # ---- embed (fit nothing on test) ----
    enc = embed_st if a.st else hashed_embed
    print(f"embedding with {'all-MiniLM-L6-v2' if a.st else f'hashed n-gram (dim={HASH_DIM})'} ...")
    Etr = enc([r["text"] for r in tr])
    Ete = enc([r["text"] for r in te])
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    Btr = np.array([[1.0] + r["lags"] + [r["n"], math.log1p(r["n"])] for r in tr])
    Bte = np.array([[1.0] + r["lags"] + [r["n"], math.log1p(r["n"])] for r in te])

    base = oos_r2(yte, Bte @ ols_fit(Btr, ytr), ytr.mean())
    print(f"\n=== target={a.target} h={h}  baseline (price history + news count) "
          f"OOS R2 = {base:.4f} ===")
    print(f"  {'k comps':>8} {'+latent R2':>11} {'incremental':>12}")
    for k in (1, 2, 3, 5, 8, 12):
        # PLS fitted on TRAIN ONLY — residualise y on the baseline first so the
        # components chase what price history does NOT already explain.
        btr = ols_fit(Btr, ytr)
        r_tr = ytr - Btr @ btr
        W, mu, sd = pls_fit(Etr, r_tr, k)
        Ttr = ((Etr - mu) / sd) @ W
        Tte = ((Ete - mu) / sd) @ W
        Xtr = np.hstack([Btr, Ttr]); Xte = np.hstack([Bte, Tte])
        r2 = oos_r2(yte, Xte @ ols_fit(Xtr, ytr), ytr.mean())
        print(f"  {k:8d} {r2:11.4f} {r2 - base:12.4f}")

    print("\n  PLS directions, standardisation and regression are ALL fitted on train\n"
          "  only. A positive incremental that grows then falls with k is the classic\n"
          "  signal-then-overfit shape; flat ~0 everywhere means the text carries no\n"
          "  predictive content beyond what price history already holds.")


if __name__ == "__main__":
    main()
