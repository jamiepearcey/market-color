# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Score the hypothesis-at-depth experiment. Per bucket (costop / cosdeep / reach)
aggregate judged relevance across all queries: mean grade, % relevant (>=1),
% strong-driver (==2). The decisive comparison is REACH vs COSDEEP — does the
method's cross-vocab reach find MORE genuine drivers than just reading deeper in
cosine? Also reports reach precision by union-rank tier (does the top of the reach
list — what an analyst sees first — hold up?)."""
import json
from pathlib import Path
D = Path("data/eval")
key = json.load(open(D / "key_reach.json"))

buckets = {}
reach_tiers = {"reach top1-4": [], "reach 5-12": []}
for qid, kmap in key.items():
    rf = D / f"rating_reach_{qid}.json"
    if not rf.exists():
        print(f"  MISSING {qid}"); continue
    grade = json.load(open(rf))
    for lid, e in kmap.items():
        g = grade.get(lid)
        if g is None: continue
        buckets.setdefault(e["bucket"], []).append(g)
        if e["bucket"] == "reach":
            ur = e.get("union_rank", 99)
            (reach_tiers["reach top1-4"] if ur < 4 else reach_tiers["reach 5-12"]).append(g)

def stat(gs):
    n = len(gs) or 1
    return len(gs), sum(gs)/n, 100*sum(1 for g in gs if g>=1)/n, 100*sum(1 for g in gs if g==2)/n

print("\nHypothesis-at-depth: judged relevance of the docs the METHOD surfaces vs controls\n")
print(f"  {'bucket':<16} {'n':>4} {'mean':>6} {'%relevant(>=1)':>15} {'%strong-driver(2)':>18}")
order = ["costop", "cosdeep", "reach"]
for b in order:
    if b in buckets:
        n, m, r, s = stat(buckets[b])
        print(f"  {b:<16} {n:>4} {m:>6.2f} {r:>14.0f}% {s:>17.0f}%")
print()
for b, gs in reach_tiers.items():
    n, m, r, s = stat(gs)
    print(f"  {b:<16} {n:>4} {m:>6.2f} {r:>14.0f}% {s:>17.0f}%")

# headline deltas
def mean(b): return (sum(buckets.get(b,[]))/len(buckets[b])) if buckets.get(b) else 0
print(f"\n  REACH vs COSDEEP mean-grade: {mean('reach'):.2f} vs {mean('cosdeep'):.2f}  (Δ{mean('reach')-mean('cosdeep'):+.2f})")
print(f"  REACH vs COSTOP  mean-grade: {mean('reach'):.2f} vs {mean('costop'):.2f}  (Δ{mean('reach')-mean('costop'):+.2f})")
