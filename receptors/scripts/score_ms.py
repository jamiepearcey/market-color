# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Score the multi-step isolation. Per bucket: mean grade, %relevant(>=1), %strong(==2).
Decisive: chainreach (PURE 2-hop) vs onehopreach (1-hop operator) vs cosdeep (read deeper)."""
import json
from pathlib import Path
D = Path("data/eval")
key = json.load(open(D / "key_ms.json"))
buckets = {}
chain_tiers = {"chainreach top1-4": [], "chainreach 5+": []}
for qid, kmap in key.items():
    rf = D / f"rating_ms_{qid}.json"
    if not rf.exists():
        print(f"  MISSING {qid}"); continue
    grade = json.load(open(rf))
    for lid, e in kmap.items():
        g = grade.get(lid)
        if g is None: continue
        buckets.setdefault(e["bucket"], []).append(g)
        if e["bucket"] == "chainreach":
            # rank among this query's chainreach docs by chain_rank
            (chain_tiers["chainreach top1-4"] if e.get("chain_rank",99) < 30 else chain_tiers["chainreach 5+"]).append(g)

def stat(gs):
    n = len(gs) or 1
    return len(gs), sum(gs)/n, 100*sum(1 for g in gs if g>=1)/n, 100*sum(1 for g in gs if g==2)/n

print("\nMULTI-STEP ISOLATION — judged relevance by source\n")
print(f"  {'bucket':<18} {'n':>4} {'mean':>6} {'%rel(>=1)':>10} {'%strong(2)':>11}")
for b in ["costop","cosdeep","onehopreach","chainreach"]:
    if b in buckets:
        n,m,r,s = stat(buckets[b])
        print(f"  {b:<18} {n:>4} {m:>6.2f} {r:>9.0f}% {s:>10.0f}%")
print()
for b,gs in chain_tiers.items():
    if gs:
        n,m,r,s = stat(gs)
        print(f"  {b:<18} {n:>4} {m:>6.2f} {r:>9.0f}% {s:>10.0f}%")
def mean(b): return (sum(buckets.get(b,[]))/len(buckets[b])) if buckets.get(b) else 0
print(f"\n  PURE 2-HOP (chainreach) vs 1-HOP reach: {mean('chainreach'):.2f} vs {mean('onehopreach'):.2f}  (Δ{mean('chainreach')-mean('onehopreach'):+.2f})")
print(f"  PURE 2-HOP (chainreach) vs read-deeper cosine: {mean('chainreach'):.2f} vs {mean('cosdeep'):.2f}  (Δ{mean('chainreach')-mean('cosdeep'):+.2f})")
