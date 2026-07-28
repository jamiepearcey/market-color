# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "sentence-transformers", "torch"]
# ///
"""
NON-LINEAR latent news signal — the test every prior result failed to cover.

WHY THIS IS NECESSARY. Every test in this series was LINEAR: OLS for the
incremental R2, PLS for the latent representation, PCA for dimension reduction,
linear market-model factor decomposition. So the accumulated bound is properly
stated as "any LINEAR incremental contribution is below ~0.5% of variance" — it
says nothing about:

    threshold effects   only extreme / unusual news moves anything
    interactions        news matters only in a given volatility state
    manifold structure  predictive regions of embedding space that are not
                        expressible as a direction

A large non-linear signal would be invisible to everything run so far, and the
linear bound would not constrain it.

And the sample-size condition for testing this only just arrived: embedding raw
article text instead of extracted quotes took the training set from 2,171 to
21,392 rows. Gradient-boosted trees need that; at n=2,171 with 384 embedding
dimensions the exercise would have been meaningless.

THE TEST — isolate the TEXT contribution, not linear-vs-non-linear:
    GBM(baseline)                 vs   GBM(baseline + embedding)
both non-linear, so the only difference is whether the model may see the text.
Comparing a linear baseline against a non-linear text model would confound the
functional form with the information, and would manufacture an apparent win.

Also reported: OLS(baseline) for context, and permutation of the embedding block
(shuffle the text rows against the targets) as a null control — if shuffled text
"helps" as much as real text, any apparent gain is leakage or variance, not
signal.

Usage:
    uv run scripts/news_nonlinear.py            # hashed n-gram text
    uv run --with sentence-transformers --with torch \\
        scripts/news_nonlinear.py --st          # semantic
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_volume import ols_fit, oos_r2  # noqa: E402
from news_raw_latent import (  # noqa: E402
    BETA_WIN, HASH_DIM, MAX_CHARS, SPLIT, build_name_dictionary, hashed_embed,
)
from liquidity import tradeable, usable_window  # noqa: E402,F401

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"


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


def embed_st(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_center(np.asarray(m.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=256)))


def assemble(horizon: int):
    """Same raw-text pipeline as news_raw_latent.py (graph used only as a name dict)."""
    docmeta = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docmeta[j["doc_id"]] = (d, j.get("headline") or "")
    body = collections.defaultdict(list)
    for l in open(G / "lake" / "chunk.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") in docmeta and j.get("text"):
            body[j["doc_id"]].append((j.get("seq", 0), j["text"]))

    names = build_name_dictionary()
    pat = {tk: re.compile("|".join(re.escape(n) for n in sorted(v, key=len, reverse=True)[:6]))
           for tk, v in names.items()}
    cells = collections.defaultdict(list)
    for doc, parts in body.items():
        day, head = docmeta[doc]
        text = head + " — " + " ".join(t for _, t in sorted(parts))
        low = text.lower()[:4000]
        for tk, rx in pat.items():
            if rx.search(low):
                cells[(tk, day)].append(text[:MAX_CHARS])

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

    rets = {}
    for tk in sorted({t for t, _ in cells}):
        r = rets_of(tk.replace("/", "-"))
        if len(r) > 500:
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

    rows = []
    for (tk, d), texts in cells.items():
        if tk not in resid:
            continue
        i = dayidx.get(d)
        if i is None or i < BETA_WIN + 120 or i + 1 + horizon >= len(alldays):
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
        fut = e[i + 1:i + 1 + horizon]; fut = fut[np.isfinite(fut)]
        if len(fut) < horizon:
            continue
        rows.append({"day": d, "lags": lags, "n": float(len(texts)),
                     "text": " ".join(texts)[:MAX_CHARS],
                     "y": math.log(max(fut.std(), 1e-8))})
    rows.sort(key=lambda r: r["day"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st", action="store_true")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = assemble(a.horizon)
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    print(f"rows {len(rows)}  train={len(tr)}  test={len(te)}")

    enc = embed_st if a.st else hashed_embed
    print(f"embedding with {'all-MiniLM-L6-v2' if a.st else f'hashed n-gram ({HASH_DIM})'} ...")
    Etr, Ete = enc([r["text"] for r in tr]), enc([r["text"] for r in te])
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    Btr = np.array([r["lags"] + [r["n"], math.log1p(r["n"])] for r in tr])
    Bte = np.array([r["lags"] + [r["n"], math.log1p(r["n"])] for r in te])
    ybar = ytr.mean()

    lin = oos_r2(yte, np.hstack([np.ones((len(te), 1)), Bte])
                 @ ols_fit(np.hstack([np.ones((len(tr), 1)), Btr]), ytr), ybar)

    def gbm(Xtr, Xte, seed=0):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        m.fit(Xtr, ytr)
        return oos_r2(yte, m.predict(Xte), ybar)

    g_base = gbm(Btr, Bte, a.seed)
    g_text = gbm(np.hstack([Btr, Etr]), np.hstack([Bte, Ete]), a.seed)

    # Null control: shuffle the embedding rows against the targets. If shuffled
    # text "helps" as much as real text, any gain is variance/leakage not signal.
    rng = np.random.default_rng(a.seed + 99)
    perm = rng.permutation(len(tr))
    permte = rng.permutation(len(te))
    g_perm = gbm(np.hstack([Btr, Etr[perm]]), np.hstack([Bte, Ete[permte]]), a.seed)

    print(f"\n=== NON-LINEAR test, target=idio h={a.horizon} "
          f"(text dim={Etr.shape[1]}) ===")
    print(f"  {'model':34} {'OOS R2':>9}")
    print(f"  {'OLS(baseline)  [linear ref]':34} {lin:9.4f}")
    print(f"  {'GBM(baseline)':34} {g_base:9.4f}")
    print(f"  {'GBM(baseline + text)':34} {g_text:9.4f}")
    print(f"  {'GBM(baseline + SHUFFLED text)':34} {g_perm:9.4f}   <- null control")
    print(f"\n  incremental from text  (GBM_text - GBM_base) = {g_text - g_base:+.4f}")
    print(f"  permutation control    (GBM_perm - GBM_base) = {g_perm - g_base:+.4f}")
    print("\n  The text incremental must clearly exceed the permutation control to count.\n"
          "  Both models are non-linear, so this isolates the INFORMATION in the text\n"
          "  rather than the functional form — comparing GBM(text) against OLS(baseline)\n"
          "  would confound the two and manufacture a win.")


if __name__ == "__main__":
    main()
