# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sentence-transformers", "torch"]
# ///
"""
RAW-TEXT latent signal — skip the graph, embed the articles.

THE POINT. Every previous latent test embedded the extraction's 86-character
verbatim QUOTES and could only use (ticker, day) cells that survived the whole
graph funnel:

    57,489 documents -> causal edges -> resolved-security effect entity
                     -> 4,884 (ticker, day) cells -> 3,377 usable rows

That throws away most of the corpus AND most of each article. We hold 185,190
chunks totalling 133M characters of real article text. If an embedding can read
raw text, compressing to a quote first is a lossy step that also shrinks the
sample — and sample size was the measured binding constraint on the last test
(n_train=2,171 vs 384 dims, monotone overfitting from k=1).

So: use the graph ONLY as a name dictionary for ticker linking (cheap and
reliable), never as a filter. Embed the article text itself. This costs nothing
— no Groq, no re-extraction.

    ticker linking : company-name / alias match in headline+body
    text           : the document's chunks, concatenated (capped)
    representation : PLS on the embedding, TRAIN-ONLY fit, y residualised on the
                     price-history baseline first
    target         : idiosyncratic vol over the next h days
    baseline       : trailing idio-vol lags + article count

The comparison that matters is against the quote-based run (n_train=2,171,
incremental -0.0136 at k=1 with MiniLM). If raw text + a bigger sample turns
that positive, the graph was the bottleneck. If it stays negative with several
times the data, the constraint is redundancy, not representation or sample size.

Usage:
    uv run scripts/news_raw_latent.py                    # hashed n-gram
    uv run --with sentence-transformers --with torch \\
        scripts/news_raw_latent.py --st                  # semantic
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
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
SPLIT = "2011-12-15"
BETA_WIN = 250
HASH_DIM = 256
MAX_CHARS = 1200          # per document fed to the encoder
STOP_NAME = {"inc", "corp", "co", "ltd", "plc", "group", "holdings", "the", "company"}


def hashed_embed(texts, dim=HASH_DIM):
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


def embed_st(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_center(np.asarray(m.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=256)))


def pls_fit(X, y, k):
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
        Xc -= np.outer(t, p)
        yc -= t * float(t @ yc) / tt
        W.append(w)
    return (np.column_stack(W) if W else np.zeros((X.shape[1], 1))), mu, sd


def build_name_dictionary():
    """ticker -> set of lowercase name variants, from entity_symbol + classification.
    The graph is used ONLY as a dictionary here, never as a filter."""
    names = collections.defaultdict(set)
    p = G / "entity_symbol.jsonl"
    if p.exists():
        for l in open(p):
            j = json.loads(l)
            if j.get("kind") != "security":
                continue
            sym = (j.get("symbol") or "").upper()
            nm = (j.get("canonical_name") or "").lower().strip()
            if sym and len(nm) >= 4:
                names[sym].add(nm)
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = (j.get("resolved_ticker") or "").upper()
        nm = (j.get("name") or "").lower().strip()
        if tk and len(nm) >= 4:
            names[tk].add(nm)
    # Drop name variants too generic to be safe evidence of a mention.
    out = {}
    for tk, nms in names.items():
        keep = {n for n in nms
                if len(n) >= 5 and not set(n.split()) <= STOP_NAME}
        if keep:
            out[tk] = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st", action="store_true")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()

    # ---- raw text per document ----
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
    print(f"documents with a date: {len(docmeta)}, with body text: {len(body)}")

    names = build_name_dictionary()
    print(f"name dictionary: {len(names)} tickers, "
          f"{sum(len(v) for v in names.values())} name variants")

    # ---- link documents to tickers by NAME MENTION (no graph filtering) ----
    cells = collections.defaultdict(list)   # (ticker, day) -> [text]
    pat = {tk: re.compile("|".join(re.escape(n) for n in sorted(v, key=len, reverse=True)[:6]))
           for tk, v in names.items()}
    for doc, parts in body.items():
        day, head = docmeta[doc]
        text = head + " — " + " ".join(t for _, t in sorted(parts))
        low = text.lower()[:4000]
        for tk, rx in pat.items():
            if rx.search(low):
                cells[(tk, day)].append(text[:MAX_CHARS])
    print(f"(ticker, day) cells from RAW TEXT: {len(cells)}  "
          f"[quote-based pipeline gave 4,884]")

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

    tickers = {tk for tk, _ in cells}
    rets = {}
    for tk in tickers:
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
    print(f"tickers with residual series: {len(resid)}")

    h = a.horizon
    rows = []
    for (tk, d), texts in cells.items():
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
        rows.append({"day": d, "lags": lags, "n": float(len(texts)),
                     "text": " ".join(texts)[:MAX_CHARS],
                     "y": math.log(max(fut.std(), 1e-8))})
    rows.sort(key=lambda r: r["day"])
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    print(f"\nrows: {len(rows)}  train={len(tr)}  test={len(te)}   "
          f"[quote-based run: train=2,171 test=1,206]")
    if len(tr) < 200 or len(te) < 100:
        sys.exit("insufficient observations")

    enc = embed_st if a.st else hashed_embed
    print(f"embedding {len(rows)} documents with "
          f"{'all-MiniLM-L6-v2' if a.st else f'hashed n-gram ({HASH_DIM})'} ...")
    Etr, Ete = enc([r["text"] for r in tr]), enc([r["text"] for r in te])
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    Btr = np.array([[1.0] + r["lags"] + [r["n"], math.log1p(r["n"])] for r in tr])
    Bte = np.array([[1.0] + r["lags"] + [r["n"], math.log1p(r["n"])] for r in te])

    base = oos_r2(yte, Bte @ ols_fit(Btr, ytr), ytr.mean())
    print(f"\n=== RAW TEXT, target=idio h={h}  baseline OOS R2 = {base:.4f} ===")
    print(f"  {'k comps':>8} {'+latent R2':>11} {'incremental':>12}")
    btr = ols_fit(Btr, ytr)
    r_tr = ytr - Btr @ btr
    for k in (1, 2, 3, 5, 8, 12, 20):
        W, mu, sd = pls_fit(Etr, r_tr, k)
        Xtr = np.hstack([Btr, ((Etr - mu) / sd) @ W])
        Xte = np.hstack([Bte, ((Ete - mu) / sd) @ W])
        r2 = oos_r2(yte, Xte @ ols_fit(Xtr, ytr), ytr.mean())
        print(f"  {k:8d} {r2:11.4f} {r2 - base:12.4f}")

    print("\n  vs the quote-based run (MiniLM): incremental -0.0136 at k=1, monotone\n"
          "  decline thereafter. More data + fuller text turning this POSITIVE would\n"
          "  mean the graph funnel was the bottleneck; still negative means the\n"
          "  constraint is redundancy with price, not representation or sample size.")


if __name__ == "__main__":
    main()
