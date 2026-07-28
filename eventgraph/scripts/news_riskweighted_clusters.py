# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "sentence-transformers", "torch"]
# ///
"""
RISK-WEIGHTED CLUSTERING — synthesise the two things that actually worked.

WHAT WE KNOW, and why this design follows from it:

  1. We MEASURED which event classes carry immediate idiosyncratic risk
     (same-day |move| in sigma vs a random-day control of 0.712):
         earnings 1.62 | employment 1.39 | debt_issuance 1.32
         econ_indicator 1.26 | guidance 1.24 | legal_regulatory 1.15
         ... m_and_a 0.91 | credit_event 0.80 | election 0.78
     So news topics are NOT equally risk-bearing — a 2x spread, with a clean
     bootstrap separation from control for the top classes.

  2. UNSUPERVISED clustering on all news LOSES to GICS (+0.053 vs +0.080 within
     -cluster residual correlation on a matched universe): it merely
     rediscovers industry, worse.
     But an ANCHORED, EVENT-CONDITIONED basket BEAT a real GICS control
     (+0.091 vs +0.003) on FOMC-day residual correlation. The difference is
     supervision plus conditioning, not the embedding itself.

THE SYNTHESIS TESTED HERE: build the company vector from its HIGH-RISK news
only — the classes measured to carry risk — instead of from everything it is
mentioned in. Then evaluate within-cluster residual correlation ON HIGH-RISK
EVENT DAYS, the window where that shared exposure should express itself, and
compare against the same clusters measured on ALL days.

If risk-weighting and conditioning both matter, the ordering should be:
    high-risk-news clusters ON event days  >  same clusters on all days
    and ideally                            >  GICS on the same event days
Every comparison is on the SAME universe and the SAME day-set, because the
previous version of this test was confounded by GICS being scored on a smaller,
more-correlated curated subset.

Usage:
    uv run --with sentence-transformers --with torch \\
        scripts/news_riskweighted_clusters.py --st
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
# SIC (SEC EDGAR), not GICS. GICS is proprietary so gics.py was ~140 tickers
# assigned BY HAND — 12% of the universe, unverifiable, and flat. SIC is public
# domain, assigned per filer by the SEC itself (74% coverage here), and genuinely
# hierarchical: division -> major group -> 4-digit industry, which is what the
# drill-down needs. SIC is an older and coarser taxonomy than GICS; it is used
# because it is real and checkable, which beats a better-shaped invented one.
from sic import sector as gics_sector, subsector, industry  # noqa: E402,F401
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
BETA_WIN = 250
RNG = np.random.default_rng(20260726)
MIN_COMMON = 200

# Measured same-day |move| in sigma per class (news_immediate_risk.py); control
# = 0.712. Used both to SELECT high-risk classes and to WEIGHT the text.
RISK = {"earnings": 1.621, "employment": 1.386, "debt_issuance": 1.318,
        "econ_indicator": 1.257, "guidance": 1.242, "other": 1.161,
        "legal_regulatory": 1.153, "monetary_policy": 1.118,
        "rating_action": 1.063, "growth": 1.053, "m_and_a": 0.912,
        "credit_event": 0.797, "election": 0.779}
CONTROL = 0.712
HIGH_RISK = {k for k, v in RISK.items() if v >= 1.20 and k != "other"}


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
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    print(f"HIGH-RISK classes (measured |move| >= 1.20 sigma): {sorted(HIGH_RISK)}")

    # ---- graph: per (ticker, day) event classes + the quote text ----
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

    hi_text = collections.defaultdict(list)   # ticker -> quotes from HIGH-RISK news
    all_text = collections.defaultdict(list)  # ticker -> all quotes
    hi_days = collections.defaultdict(set)    # ticker -> days with a HIGH-RISK event
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        q = (j.get("quote") or "").strip()
        if e not in ent or not d or not q:
            continue
        tk = ent[e]
        types = doctypes.get(doc) or {"other"}
        all_text[tk].append(q)
        if types & HIGH_RISK:
            hi_text[tk].append(q)
            hi_days[tk].add(d)

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

    cands = [tk for tk in hi_text if len(hi_text[tk]) >= 5]
    rets = {}
    for tk in cands:
        r = rets_of(tk.replace("/", "-"))
        if len(r) > 600 and tradeable(r):
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
        if np.isfinite(e).sum() >= MIN_COMMON:
            resid[tk] = e
    tks = sorted(resid)
    n = len(tks)
    print(f"tickers with high-risk news + residual series: {n}")

    # ---- company vectors: HIGH-RISK news only vs ALL news ----
    enc = embed_st if a.st else hashed_embed
    print("embedding ...")
    V_hi = enc([" ".join(hi_text[tk][:60]) for tk in tks])
    V_all = enc([" ".join(all_text[tk][:60]) for tk in tks])
    for V in (V_hi, V_all):
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    # ---- day-sets: all days vs HIGH-RISK event days ----
    M = np.vstack([resid[tk] for tk in tks])
    ok = np.isfinite(M)
    hi_idx = {tk: {dayidx[d] for d in hi_days[tk] if d in dayidx} for tk in tks}

    def corr_matrix(day_filter):
        C = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                m = ok[i] & ok[j]
                if day_filter is not None:
                    sel = np.zeros(len(alldays), bool)
                    idxs = sorted(hi_idx[tks[i]] | hi_idx[tks[j]])
                    for t in idxs:
                        sel[max(t - 1, 0):t + 2] = True   # event day +/- 1
                    m = m & sel
                if m.sum() < (30 if day_filter is not None else MIN_COMMON):
                    continue
                x, y = M[i][m], M[j][m]
                if x.std() > 0 and y.std() > 0:
                    C[i, j] = C[j, i] = float(np.corrcoef(x, y)[0, 1])
        return C

    C_all = corr_matrix(None)
    C_ev = corr_matrix("hi")
    print(f"mean pairwise residual corr — all days {np.nanmean(C_all):+.4f}, "
          f"high-risk event windows {np.nanmean(C_ev):+.4f}")

    from sklearn.cluster import KMeans
    lab_hi = list(KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(V_hi).labels_)
    lab_all = list(KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(V_all).labels_)
    gl = [gics_sector(tk) for tk in tks]
    lab_gics = [None if g == "UNK" else g for g in gl]

    def within(C, labels, idx=None):
        rng = range(n) if idx is None else idx
        vals = [C[i, j] for i in rng for j in rng
                if i < j and labels[i] is not None and labels[i] == labels[j]
                and np.isfinite(C[i, j])]
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

    # Fair universe: score every grouping on the GICS-known subset.
    known = [i for i in range(n) if lab_gics[i] is not None]
    print(f"\n=== WITHIN-CLUSTER residual correlation "
          f"(k={a.k}, matched universe n={len(known)}) ===")
    print(f"  {'grouping':30} {'ALL days':>12} {'EVENT windows':>15}")
    for name, lab in (("clusters: HIGH-RISK news", lab_hi),
                      ("clusters: all news", lab_all),
                      ("GICS sectors", lab_gics)):
        wa, _ = within(C_all, lab, known)
        we, _ = within(C_ev, lab, known)
        print(f"  {name:30} {wa:+12.4f} {we:+15.4f}")
    rnd_a, rnd_e = [], []
    for _ in range(200):
        s = list(lab_hi); RNG.shuffle(s)
        rnd_a.append(within(C_all, s, known)[0])
        rnd_e.append(within(C_ev, s, known)[0])
    print(f"  {'random (same sizes)':30} {np.nanmean(rnd_a):+12.4f} "
          f"{np.nanmean(rnd_e):+15.4f}")

    wa_hi, _ = within(C_all, lab_hi, known)
    we_hi, _ = within(C_ev, lab_hi, known)
    we_g, _ = within(C_ev, lab_gics, known)
    print(f"\n  high-risk clusters: event-window lift over all-days = "
          f"{we_hi - wa_hi:+.4f}")
    print(f"  high-risk clusters vs GICS on event windows        = {we_hi - we_g:+.4f}")
    print(f"  p vs random (event windows) = "
          f"{float(np.mean(np.array(rnd_e) >= we_hi)):.4f}")
    print("\n  The claim being tested: risk-weighting the text AND conditioning on the\n"
          "  event window is what made the embedding beat GICS before. Unsupervised,\n"
          "  unconditional clustering already lost to GICS (+0.053 vs +0.080).")


if __name__ == "__main__":
    main()
