# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","scipy"]
# ///
"""
DOES STABLE CHARACTERISTIC TEXT PREDICT FUTURE CO-MOVEMENT? (the correctly-specified test)

THE DIAGNOSIS THIS TESTS. F53: recency decay HURT the news signal, so exposure
profiles behave like a stable characteristic. F55: news vocabulary is period topic
and does not transfer across time (OOS +0.004). Together: we were estimating a
STABLE CHARACTERISTIC with a TIME-VARYING EVENT STREAM. This swaps the input for
10-K Item 1 business descriptions — what a firm IS rather than what happened to it —
which is the document type the Hoberg-Phillips text-based-industry literature uses.

THE TEST IS OUT-OF-TIME BY CONSTRUCTION. Period A and period B are disjoint and
ordered. The question is whether text similarity predicts period-B correlation
INCREMENTAL to period-A correlation. Nothing about B is used to build the features,
and the text is a stable characteristic, so there is no window to leak through.

CONTROLS, because today's lesson is that the control set decides everything:
  same-sector   -- text similarity that is just an industry classifier is not news
  corr in A     -- the price-history baseline, the bar everything must clear
Returns are cross-sectionally demeaned each day (a clean one-factor market removal
with no beta estimation, hence no leakage).

Usage:
    uv run scripts/stable_text_covariance.py
    uv run scripts/stable_text_covariance.py --split 2020-01-01
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg_live2"
STOP = set("the a an and or of to in on for at by with from as is are was were be been it its this that "
           "these those has have had will would may might can could our we us their they which such "
           "other including include also any all more most than then so but not no if under over into "
           "through during company companies business operations products services inc corp plc ltd "
           "common stock shares report annual fiscal year years period date "
           # SEC FILING FURNITURE ONLY. The "Available Information" paragraph
           # paired firms at random (ACGL~DUK, JPM~UAL on practicable/furnished).
           # But the first attempt ALSO stripped securities/exchange/commission/
           # investor-relations, which are furniture for most filers and genuine
           # BUSINESS vocabulary for financials -- and the statistic fell 0.065 ->
           # 0.047. This list is restricted to words that are never a business
           # descriptor for anyone.
           "practicable furnished electronically herein hereto thereto hereby "
           "www http html aspx toll-free charge copies posted routinely website "
           "websites edgar proxy amendments amended incorporated reference "
           "stockholders shareholders filings filed 10-k 10-q 8-k forms "
           "table contents item items part page pages approximately".split())

TOK = re.compile(r"[a-z][a-z0-9-]{2,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2020-01-01")
    ap.add_argument("--min-bars", type=int, default=900)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--useful", action="store_true",
                    help="Translate the partial correlation into something usable: how much "
                         "the correlation FORECAST improves, and the concrete pairs where "
                         "price history and text DISAGREE -- which is the whole point.")
    ap.add_argument("--walk", action="store_true",
                    help="Walk the A/B split forward through disjoint regimes. A stable "
                         "characteristic should hold at similar magnitude everywhere; "
                         "look-ahead from a 2025-26 filing should DECAY the further back "
                         "period B sits. This is the anachronism check generalised.")
    ap.add_argument("--maxchars", type=int, default=2500,
                    help="Truncate each description. The stored 2500-char window often runs "
                         "PAST the business description into the availability/risk sections, "
                         "so a shorter early window is purer characteristic text.")
    ap.add_argument("--cut-boilerplate", action="store_true",
                    help="Cut each description at the first 'Available Information' marker "
                         "rather than by a fixed length -- targets the furniture where it "
                         "lives instead of stripping words globally.")
    ap.add_argument("--jackknife", action="store_true",
                    help="Drop each firm in turn and re-measure. Cutting 5 of 183 documents "
                         "moved the statistic 0.064 -> 0.034, which suggests a handful of "
                         "names carry it. This finds them.")
    ap.add_argument("--examples", action="store_true",
                    help="Print the highest-similarity PAIRS and the terms driving them. "
                         "F54's lesson: check what the signal is made of BEFORE optimising it.")
    a = ap.parse_args()

    bt = json.load(open(a.text_file if a.text_file else G / "business_text.json"))
    # ---- prices -> cross-sectionally demeaned returns -------------------------
    px = {}
    for f in sorted(os.listdir(G / "prices")):
        t = f[:-5]
        if not f.endswith(".json") or t not in bt:
            continue
        try:
            j = json.load(open(G / "prices" / f))
        except Exception:
            continue
        r = (j.get("chart") or {}).get("result")
        if not r:
            continue
        ts = r[0].get("timestamp") or []
        cl = r[0]["indicators"]["quote"][0].get("close") or []
        pts = [(dt.datetime.fromtimestamp(x, dt.timezone.utc).date().isoformat(), c)
               for x, c in zip(ts, cl) if c is not None]
        if len(pts) < a.min_bars:
            continue
        tail = [c for _, c in pts[-60:]]
        z = sum(1 for i in range(1, len(tail)) if abs(tail[i] - tail[i - 1]) < 1e-9) / max(1, len(tail) - 1)
        if z >= 0.15:
            continue
        px[t] = dict(pts)

    names = sorted(px)
    days = sorted({d for s in px.values() for d in s})
    ret = {t: {} for t in names}
    for t in names:
        ds = sorted(px[t])
        for i in range(1, len(ds)):
            p0, p1 = px[t][ds[i - 1]], px[t][ds[i]]
            if p0:
                ret[t][ds[i]] = (p1 - p0) / p0
    mkt = {}
    for d in days:
        v = [ret[t][d] for t in names if d in ret[t]]
        if len(v) >= 30:
            mkt[d] = float(np.mean(v))
    A = [d for d in days if d in mkt and d < a.split]
    B = [d for d in days if d in mkt and d >= a.split]
    print(f"{len(names)} names with business text + usable prices")
    print(f"period A {A[0]}..{A[-1]} ({len(A)} sessions)   period B {B[0]}..{B[-1]} ({len(B)} sessions)")

    def mat(win):
        M = np.full((len(names), len(win)), np.nan)
        for i, t in enumerate(names):
            for j, d in enumerate(win):
                if d in ret[t]:
                    M[i, j] = ret[t][d] - mkt[d]        # market removed, no beta fit
        return M

    def corrmat(M):
        Z = np.where(np.isfinite(M), M, np.nan)
        mu = np.nanmean(Z, axis=1, keepdims=True)
        sd = np.nanstd(Z, axis=1, keepdims=True)
        Zs = (Z - mu) / np.where(sd > 0, sd, 1)
        Zs = np.nan_to_num(Zs)
        n = (np.isfinite(Z)).astype(float) @ (np.isfinite(Z)).astype(float).T
        C = (Zs @ Zs.T) / np.maximum(n, 1)
        return C, n

    CA, nA = corrmat(mat(A))
    CB, nB = corrmat(mat(B))

    # ---- stable-text similarity (idf term overlap over 10-K Item 1) -----------
    BOIL = re.compile(r"available free of charge|as soon as reasonably practicable|"
                      r"investor relations|our website|sec\.gov|electronically with the|"
                      r"proxy statement|we (?:make|file) .{0,40}(?:available|with the)", re.I)
    def prep(x):
        if a.cut_boilerplate:
            m = BOIL.search(x)
            if m and m.start() > 250:
                x = x[:m.start()]
        return x[:a.maxchars].lower()
    docs = [prep(bt[t]["text"]) for t in names]
    print(f"description length: mean {np.mean([len(d) for d in docs]):.0f} chars")
    terms = [{w for w in TOK.findall(d) if w not in STOP} for d in docs]
    df = collections.Counter()
    for s in terms:
        df.update(s)
    vocab = {w: i for i, w in enumerate(df)}
    idf = np.array([np.log(1 + len(docs) / (1 + df[w])) for w in vocab], dtype=np.float32)
    rows, cols = [], []
    for i, s in enumerate(terms):
        for w in s:
            rows.append(i); cols.append(vocab[w])
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
    X = X.multiply(idf[None, :]).tocsr()
    nrm = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel(); nrm[nrm == 0] = 1
    X = sp.diags(1 / nrm) @ X
    S = (X @ X.T).toarray()
    print(f"business-text vocabulary {len(vocab)} terms")

    # ---- sector control -------------------------------------------------------
    sec = {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t in px and j.get("std_sector") not in (None, "UNK"):
            sec.setdefault(t, j["std_sector"])
    same = np.zeros_like(S)
    for i, ti in enumerate(names):
        for j, tj in enumerate(names):
            same[i, j] = 1.0 if (sec.get(ti) and sec.get(ti) == sec.get(tj)) else 0.0

    iu = np.triu_indices(len(names), 1)
    keep = (nA[iu] >= 250) & (nB[iu] >= 250)
    ca, cb, s_, sm = CA[iu][keep], CB[iu][keep], S[iu][keep], same[iu][keep]
    print(f"pairs used: {keep.sum()} of {len(iu[0])}\n")

    def pcorr(x, y, *ctrl):
        Xd = np.column_stack([np.ones(len(x))] + list(ctrl))
        rx = x - Xd @ np.linalg.lstsq(Xd, x, rcond=None)[0]
        ry = y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])

    if a.examples:
        inv={v:k for k,v in vocab.items()}
        Xd=np.asarray(X.todense())
        pairs=[(S[i,j],i,j) for i,j in zip(*iu) if keep[list(zip(*iu)).index((i,j))] if False]
        # rank pairs directly
        flat=[(S[i,j],i,j) for i,j in zip(*np.triu_indices(len(names),1))]
        flat.sort(reverse=True)
        print("=== HIGHEST-SIMILARITY PAIRS — what is the match made of?")
        for sc,i,j in flat[:12]:
            contrib=Xd[i]*Xd[j]
            top=np.argsort(-contrib)[:6]
            tw=[inv[k] for k in top if contrib[k]>0]
            print(f"  {names[i]:6} ~ {names[j]:6}  sim {sc:.2f}   corrA {CA[i,j]:+.2f} corrB {CB[i,j]:+.2f}   "
                  f"{', '.join(tw)}")
        print()
    if a.jackknife:
        base = pcorr(s_, cb, ca, sm)
        ii, jj = iu[0][keep], iu[1][keep]
        infl = []
        for k, t in enumerate(names):
            m = (ii != k) & (jj != k)
            if m.sum() < 500: continue
            infl.append((pcorr(s_[m], cb[m], ca[m], sm[m]) - base, t, int(m.sum())))
        infl.sort()
        print(f"=== JACKKNIFE BY FIRM (base {base:+.3f}) — drop one name, how far does it move?")
        print("  most INFLATING (dropping them lowers the result):")
        for d, t, n in infl[:8]:
            print(f"    {t:6} {d:+.4f}")
        print("  most DEFLATING:")
        for d, t, n in infl[-4:]:
            print(f"    {t:6} {d:+.4f}")
        tot = sum(abs(d) for d, _, _ in infl)
        top5 = sum(abs(d) for d, _, _ in infl[:5])
        print(f"  top-5 names account for {top5/tot:.0%} of total absolute influence")
        import sys as _s; _s.exit(0)
    if a.walk:
        SPLITS = [("2014-01-01","2016-07-01","2019-01-01"),
                  ("2016-07-01","2019-01-01","2021-07-01"),
                  ("2019-01-01","2021-07-01","2024-01-01"),
                  ("2021-07-01","2024-01-01","2026-08-01")]
        print("=== WALK-FORWARD SPLITS  (10-K filed 2025-26 throughout)")
        print(f"  {'period A':<24}{'period B':<24}{'pairs':>7}{'incr':>9}{'p':>8}")
        rng2 = np.random.default_rng(3)
        for a0,a1,b1 in SPLITS:
            wa=[d for d in days if d in mkt and a0<=d<a1]
            wb=[d for d in days if d in mkt and a1<=d<b1]
            if len(wa)<200 or len(wb)<200: continue
            Ca,na=corrmat(mat(wa)); Cb,nb=corrmat(mat(wb))
            k=(na[iu]>=180)&(nb[iu]>=180)
            if k.sum()<1000: continue
            x,y,c,sq=S[iu][k],Cb[iu][k],Ca[iu][k],same[iu][k]
            obs=pcorr(x,y,c,sq)
            nl=[]
            for _ in range(200):
                pm=rng2.permutation(len(names)); Sp=S[np.ix_(pm,pm)]
                nl.append(pcorr(Sp[iu][k],y,c,sq))
            pv=(np.sum(np.array(nl)>=obs)+1)/(len(nl)+1)
            print(f"  {a0}..{a1}   {a1}..{b1}   {k.sum():>6}  {obs:+.3f}  {pv:>7.3f}"
                  f"{'  *' if pv<0.05 else ''}")
        import sys as _s; _s.exit(0)
    if a.useful:
        # 1. forecast error, fit on A-half of pairs, applied out of sample by FIRM
        rng3=np.random.default_rng(5)
        half=set(rng3.choice(len(names),len(names)//2,replace=False).tolist())
        ii,jj=iu[0][keep],iu[1][keep]
        tr=np.array([ (i in half) and (j in half) for i,j in zip(ii,jj)])
        te=np.array([ (i not in half) and (j not in half) for i,j in zip(ii,jj)])
        def fit_pred(cols_tr,cols_te,y_tr,y_te):
            X=np.column_stack([np.ones(cols_tr[0].shape[0])]+list(cols_tr))
            b,*_=np.linalg.lstsq(X,y_tr,rcond=None)
            Xt=np.column_stack([np.ones(cols_te[0].shape[0])]+list(cols_te))
            return float(np.sqrt(np.mean((y_te-Xt@b)**2)))
        r0=fit_pred([ca[tr]],[ca[te]],cb[tr],cb[te])
        r1=fit_pred([ca[tr],s_[tr]],[ca[te],s_[te]],cb[tr],cb[te])
        print("=== IS THE OUTPUT USEFUL? (1) correlation-forecast error")
        print(f"  firms split in half; model fitted on one half's pairs, tested on the other's")
        print(f"  RMSE, price history only        {r0:.4f}")
        print(f"  RMSE, + business-text similarity {r1:.4f}")
        print(f"  improvement                      {(r0-r1)/r0*100:+.2f}%")

        # 2. the pairs where price history and text DISAGREE
        loA=np.percentile(ca,35); hiS=np.percentile(s_,97)
        trap=(ca<loA)&(s_>hiS)
        print(f"\n=== (2) DIVERSIFICATION TRAPS — history says unrelated, text says same business")
        print(f"  {trap.sum()} pairs: past corr below {loA:+.2f}, text sim above {hiS:.2f}")
        print(f"    their mean corr in A  {ca[trap].mean():+.3f}")
        print(f"    their mean corr in B  {cb[trap].mean():+.3f}   (change {cb[trap].mean()-ca[trap].mean():+.3f})")
        base=(ca<loA)
        print(f"  ALL low-past-corr pairs ({base.sum()}): {ca[base].mean():+.3f} -> {cb[base].mean():+.3f}"
              f"   (change {cb[base].mean()-ca[base].mean():+.3f})")
        print(f"  -> text-flagged pairs converge {((cb[trap].mean()-ca[trap].mean())-(cb[base].mean()-ca[base].mean())):+.3f} MORE")
        idx=np.where(trap)[0]
        order=idx[np.argsort(-(cb[idx]-ca[idx]))][:10]
        print("\n  the flagged pairs, ranked by how much they actually converged:")
        for k in order:
            print(f"    {names[ii[k]]:6} ~ {names[jj[k]]:6}  sim {s_[k]:.2f}   corr {ca[k]:+.2f} -> {cb[k]:+.2f}"
                  f"   ({cb[k]-ca[k]:+.2f})")
        import sys as _s; _s.exit(0)
    print("=== PREDICTING PERIOD-B CORRELATION")
    print(f"  corr A -> corr B  (price-history baseline)        {np.corrcoef(ca, cb)[0,1]:+.3f}")
    print(f"  same-sector -> corr B                             {np.corrcoef(sm, cb)[0,1]:+.3f}")
    print(f"  business-text sim -> corr B  (raw)                {np.corrcoef(s_, cb)[0,1]:+.3f}")
    print(f"\n  business-text sim -> corr B | corr A              {pcorr(s_, cb, ca):+.3f}   <- the test")
    print(f"  business-text sim -> corr B | corr A + same-sector {pcorr(s_, cb, ca, sm):+.3f}   <- vs an industry dummy")
    print(f"\n  for reference, the NEWS-text equivalent was OOS   +0.004  (F55)")

    # ---- ANACHRONISM CHECK ---------------------------------------------------
    # The 10-K is filed 2025-26, i.e. at the END of period B. If it describes what
    # each firm BECAME, it should predict late-B better than early-B. A stable
    # characteristic should predict both alike.
    print("\n=== ANACHRONISM CHECK — the filing post-dates period B")
    mid = "2023-01-01"
    for lbl, win in (("B early (2020-2022)", [d for d in B if d < mid]),
                     ("B late  (2023-2026)", [d for d in B if d >= mid])):
        Cw, nw = corrmat(mat(win))
        kw = (nA[iu] >= 250) & (nw[iu] >= 200)
        inc = pcorr(S[iu][kw], Cw[iu][kw], CA[iu][kw], same[iu][kw])
        print(f"  text sim -> corr in {lbl} | corr A + sector : {inc:+.3f}   ({kw.sum()} pairs)")

    # ---- PERMUTATION NULL ----------------------------------------------------
    # Shuffle firm labels on the TEXT matrix only. Preserves the return
    # correlation structure and the text structure; breaks only the pairing.
    # This is the right null for dyadic data, where a bare correlation's nominal
    # SE is badly overstated.
    rng = np.random.default_rng(7)
    obs = pcorr(s_, cb, ca, sm)
    null = []
    for _ in range(400):
        pm = rng.permutation(len(names))
        Sp = S[np.ix_(pm, pm)]
        null.append(pcorr(Sp[iu][keep], cb, ca, sm))
    null = np.array(null)
    pval = (np.sum(null >= obs) + 1) / (len(null) + 1)
    print(f"\n=== PERMUTATION NULL (400 firm-label shuffles of the text matrix)")
    print(f"  observed {obs:+.3f}   null mean {null.mean():+.3f}   null p95 {np.percentile(null,95):+.3f}"
          f"   p = {pval:.3f}{'  *' if pval<0.05 else ''}")


if __name__ == "__main__":
    main()
