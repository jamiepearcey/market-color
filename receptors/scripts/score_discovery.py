# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Score the driver-discovery eval on the CORRECTED metric: novel, load-bearing
drivers surfaced vs the baseline — a set-difference-then-verify count, not pooled
snippet-relevance.

For each arm, reconstruct its top-K ranked candidates from key_disc.json, attach the
content-review grades (rating_disc_<qid>.json: lid -> 0/1/2), and compute per query:

  discovery@K   # grade==2 (novel & load-bearing) in the arm's top-K   <- headline
  drivers@K     # grade>=1 (any genuine driver)     in the arm's top-K
  precision@K   drivers@K / K
  novel_yield   discovery@K / drivers@K   (share of real drivers that are net-new)

Macro-averaged over queries, with a bootstrap CI on the delta of discovery@K vs a
reference arm (default B = the current cosine-top-5 pipeline). The decisive question:
does C (graph-conditioned) or D (hybrid) beat B on discovery@K with a CI excluding 0?
If so, the graph earns its place as a hypothesis conditioner and RETRIEVAL_FINDINGS.md's
"no operator/graph needed" conclusion — which only ever tested the graph as a
first-stage retriever — needs revising.

Usage:
  uv run scripts/score_discovery.py --arms A,B,C,D --ref B --k 10
"""
import json, argparse
from pathlib import Path
import numpy as np

D = Path("data/eval")
RNG = np.random.default_rng(0)


def arm_ranked(kmap, arm, k):
    lids = [lid for lid, e in kmap.items() if arm in e["ranks"]]
    lids.sort(key=lambda lid: kmap[lid]["ranks"][arm])
    return lids[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--ref", default="B")
    ap.add_argument("--suffix", default="disc")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    K = a.k

    key = json.load(open(D / f"key_{a.suffix}.json"))
    # per-arm, per-query vectors of the four metrics
    M = {arm: {"disc": [], "drv": [], "prec": [], "yield": []} for arm in arms}
    qids_used = []
    for qid, kmap in key.items():
        rf = D / f"rating_{a.suffix}_{qid}.json"
        if not rf.exists():
            print(f"  MISSING ratings for {qid} ({rf.name})")
            continue
        grade = json.load(open(rf))
        qids_used.append(qid)
        for arm in arms:
            ranked = arm_ranked(kmap, arm, K)
            gr = [int(grade.get(lid, 0)) for lid in ranked]
            disc = sum(1 for g in gr if g == 2)
            drv = sum(1 for g in gr if g >= 1)
            n = len(gr) or 1
            M[arm]["disc"].append(disc)
            M[arm]["drv"].append(drv)
            M[arm]["prec"].append(drv / n)
            M[arm]["yield"].append(disc / drv if drv else 0.0)

    nq = len(qids_used)
    print(f"\nDriver-discovery eval — novel-load-bearing vs baseline — "
          f"{nq} queries, top-{K}\n")
    print(f"  {'arm':<10} {'discovery@K':>12} {'drivers@K':>10} "
          f"{'precision':>10} {'novel_yield':>12}")
    for arm in arms:
        d = M[arm]
        tag = "  (ref)" if arm == a.ref else ""
        print(f"  {arm:<10} {np.mean(d['disc']):>12.3f} {np.mean(d['drv']):>10.3f} "
              f"{np.mean(d['prec']):>10.3f} {np.mean(d['yield']):>12.3f}{tag}")

    if a.ref in M and nq > 0:
        ref = np.array(M[a.ref]["disc"], float)
        print(f"\n  discovery@K delta vs {a.ref} (bootstrap 95% CI over {nq} queries):")
        for arm in arms:
            if arm == a.ref:
                continue
            d = np.array(M[arm]["disc"], float) - ref
            bs = [np.mean(d[RNG.integers(0, len(d), len(d))]) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            w = int((d > 0).sum()); l = int((d < 0).sum())
            verdict = "SIGNAL" if lo > 0 else ("neg" if hi < 0 else "ns")
            print(f"    {arm:<10} Δ{np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"W{w}/L{l}  {verdict}")


if __name__ == "__main__":
    main()
