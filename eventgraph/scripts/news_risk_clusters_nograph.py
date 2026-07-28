# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "sentence-transformers", "torch"]
# ///
"""
RISK CLUSTERING FROM RAW NEWS TEXT — no graph, no extraction.

THE QUESTION. Clustering companies by shared risk exposure is the one thing news
has repeatedly been shown to do well here (the earlier graph-based embedding
basket beat a REAL GICS same-sector control on FOMC-day residual correlation,
+0.091 vs +0.003). But that used quotes harvested from extracted causal edges.
Does it still work with NO graph at all — just article text and name matching?

That matters for two reasons: the graph funnel costs ~90% of the sample
(4,884 cells vs 56,969 from raw text), and extraction costs money while
name-matching is free.

THE TEST. Build a company vector = mean embedding of every article mentioning
it. k-means into K clusters. Then ask whether cluster co-membership predicts
SHARED IDIOSYNCRATIC RISK — mean pairwise correlation of market-model residuals
within a cluster. Residuals, not raw returns, because raw returns co-move
through the market factor and would make any clustering look successful.

THREE COMPARISONS, because a number alone means nothing:
    news clusters   k-means on the news embedding
    GICS sectors    the real curated sector map (gics.py) — the benchmark to beat
    random clusters same sizes, shuffled membership — the noise floor

If news clusters beat random but not GICS, the embedding is rediscovering
industry. If they beat GICS too, the embedding is finding risk structure that
industry classification misses — which is the actual claim.

Usage:
    uv run scripts/news_risk_clusters_nograph.py           # hashed n-gram
    uv run --with sentence-transformers --with torch \\
        scripts/news_risk_clusters_nograph.py --st         # semantic
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
# SIC (SEC EDGAR), not GICS. GICS is proprietary so gics.py was ~140 tickers
# assigned BY HAND — 12% of the universe, unverifiable, and flat. SIC is public
# domain, assigned per filer by the SEC itself (74% coverage here), and genuinely
# hierarchical: division -> major group -> 4-digit industry, which is what the
# drill-down needs. SIC is an older and coarser taxonomy than GICS; it is used
# because it is real and checkable, which beats a better-shaped invented one.
from sic import sector as gics_sector, subsector, industry  # noqa: E402,F401
from news_raw_latent import build_name_dictionary  # noqa: E402
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
BETA_WIN = 250
RNG = np.random.default_rng(20260726)
MIN_ARTICLES = 8
MIN_COMMON_DAYS = 200


def hashed_embed(texts, dim=256):
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--st", action="store_true")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()

    # ---- raw articles, linked to tickers by NAME ONLY (no causal edges) ----
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
    per_ticker = collections.defaultdict(list)
    for doc, parts in body.items():
        _day, head = docmeta[doc]
        text = head + " — " + " ".join(t for _, t in sorted(parts))
        low = text.lower()[:4000]
        for tk, rx in pat.items():
            if rx.search(low):
                per_ticker[tk].append(text[:900])
    per_ticker = {tk: v for tk, v in per_ticker.items() if len(v) >= MIN_ARTICLES}
    print(f"tickers with >={MIN_ARTICLES} mentioning articles: {len(per_ticker)}")

    # ---- idiosyncratic residuals ----
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
    for tk in per_ticker:
        r = rets_of(tk.replace("/", "-"))
        if len(r) > 600 and tradeable(r):
            rets[tk] = r
    daycount = collections.Counter()
    for r in rets.values():
        daycount.update(r.keys())
    thresh = max(30, int(0.30 * len(rets)))
    alldays = sorted([d for d, c in daycount.items() if c >= thresh])
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])

    resid = {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < MIN_COMMON_DAYS:
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
        if np.isfinite(e).sum() >= MIN_COMMON_DAYS:
            resid[tk] = e
    tks = sorted(resid)
    print(f"tickers with idiosyncratic residual series: {len(tks)}")

    # ---- company vectors from raw text ----
    enc = embed_st if a.st else hashed_embed
    print(f"embedding company text with "
          f"{'all-MiniLM-L6-v2' if a.st else 'hashed n-gram'} ...")
    docs = [" ".join(per_ticker[tk][:40]) for tk in tks]
    V = enc(docs)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    # ---- pairwise residual correlation matrix ----
    M = np.vstack([resid[tk] for tk in tks])
    ok = np.isfinite(M)
    n = len(tks)
    Cm = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m = ok[i] & ok[j]
            if m.sum() < MIN_COMMON_DAYS:
                continue
            x, y = M[i][m], M[j][m]
            if x.std() > 0 and y.std() > 0:
                Cm[i, j] = Cm[j, i] = float(np.corrcoef(x, y)[0, 1])

    def within(labels):
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j] and labels[i] is not None and np.isfinite(Cm[i, j]):
                    vals.append(Cm[i, j])
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

    overall = float(np.nanmean(Cm))
    print(f"\nmean pairwise idiosyncratic correlation, ALL pairs: {overall:+.4f}")

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(V)
    news_lab = list(km.labels_)
    w_news, n_news = within(news_lab)

    gl = [gics_sector(tk) for tk in tks]
    gics_lab = [None if g == "UNK" else g for g in gl]
    w_gics, n_gics = within(gics_lab)

    # Random control: preserve the news cluster SIZE distribution exactly.
    rand_w = []
    for _ in range(200):
        lab = list(news_lab)
        RNG.shuffle(lab)
        rand_w.append(within(lab)[0])
    rand_w = np.array(rand_w)

    print(f"\n=== WITHIN-CLUSTER idiosyncratic correlation (k={a.k}) ===")
    print(f"  {'grouping':22} {'within-corr':>12} {'n pairs':>9} {'vs random':>11}")
    print(f"  {'news clusters (no graph)':22} {w_news:+12.4f} {n_news:9d} "
          f"{w_news - rand_w.mean():+11.4f}")
    print(f"  {'GICS sectors':22} {w_gics:+12.4f} {n_gics:9d} "
          f"{w_gics - rand_w.mean():+11.4f}")
    print(f"  {'random (same sizes)':22} {rand_w.mean():+12.4f} {'—':>9} "
          f"{'—':>11}   p95={np.percentile(rand_w, 95):+.4f}")
    # FAIR COMPARISON: GICS covers only ~140 curated (mostly mega-cap) names, so
    # the two groupings above are measured on different — and differently
    # correlated — universes. Restrict BOTH to the GICS-known subset.
    known = [i for i in range(n) if gics_lab[i] is not None]
    if len(known) >= 20:
        idx = {v: k for k, v in enumerate(known)}
        Cs = Cm[np.ix_(known, known)]
        def within_sub(labels):
            vals = []
            for ii in range(len(known)):
                for jj in range(ii + 1, len(known)):
                    if labels[ii] == labels[jj] and np.isfinite(Cs[ii, jj]):
                        vals.append(Cs[ii, jj])
            return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
        nl = [news_lab[i] for i in known]
        gl2 = [gics_lab[i] for i in known]
        wn, cn = within_sub(nl)
        wg, cg = within_sub(gl2)
        rw = []
        for _ in range(200):
            lab = list(nl); RNG.shuffle(lab); rw.append(within_sub(lab)[0])
        rw = np.array(rw)
        print(f"\n=== FAIR COMPARISON — same universe ({len(known)} GICS-known tickers, "
              f"mean corr {float(np.nanmean(Cs)):+.4f}) ===")
        print(f"  {'grouping':22} {'within-corr':>12} {'n pairs':>9}")
        print(f"  {'news clusters':22} {wn:+12.4f} {cn:9d}")
        print(f"  {'GICS sectors':22} {wg:+12.4f} {cg:9d}")
        print(f"  {'random (same sizes)':22} {rw.mean():+12.4f} {'—':>9}"
              f"   p95={np.percentile(rw,95):+.4f}")
        print(f"  news beats random: p={float(np.mean(rw >= wn)):.4f}   "
              f"news - GICS = {wn - wg:+.4f}")

    p = float(np.mean(rand_w >= w_news))
    print(f"\n  news-cluster p vs random: {p:.4f}")
    print(f"  news vs GICS: {w_news - w_gics:+.4f} "
          f"({'news finds structure GICS misses' if w_news > w_gics else 'GICS is stronger — the embedding is at best rediscovering industry'})")

    # What the clusters look like, so the result is inspectable.
    print("\n  cluster composition (top GICS sectors per news cluster):")
    for c in range(a.k):
        mem = [tks[i] for i in range(n) if news_lab[i] == c]
        if len(mem) < 3:
            continue
        secs = collections.Counter(gics_sector(t) for t in mem)
        top = ", ".join(f"{s}:{v}" for s, v in secs.most_common(3))
        print(f"    c{c:<2} n={len(mem):3d}  {top}   e.g. {', '.join(mem[:5])}")


if __name__ == "__main__":
    main()
