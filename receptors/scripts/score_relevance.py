# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Score the pooled blind-relevance experiment. For each arm reconstruct its
ranked list from key.json (per-lid per-arm rank), attach the judge grades from
rating_<qid>.json, and compute per-arm: mean grade@10, #relevant(>=1)@10,
#strong(==2)@10, and nDCG@10 (ideal = best ordering of the query's pooled grades).
Macro-averaged over queries.
"""
import json, math, sys
from pathlib import Path

D = Path("data/eval")
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cosine", "old", "new"]
key = json.load(open(D / f"key{suffix}.json"))
K = 10
def rpath(qid): return D / f"rating{suffix}_{qid}.json"

def dcg(gr):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gr))

agg = {a: {"grade": [], "rel": [], "strong": [], "ndcg": []} for a in arms}
per_q = {}
for qid, kmap in key.items():
    rf = D / f"rating_{qid}.json"
    if not rf.exists():
        print(f"  MISSING ratings for {qid}"); continue
    grade = json.load(open(rf))
    all_g = sorted((grade.get(lid, 0) for lid in kmap), reverse=True)
    idcg = dcg(all_g[:K]) or 1.0
    per_q[qid] = {}
    for a in arms:
        ranked = sorted([lid for lid, e in kmap.items() if a in e["ranks"]],
                        key=lambda lid: kmap[lid]["ranks"][a])[:K]
        gr = [grade.get(lid, 0) for lid in ranked]
        n = len(gr) or 1
        m = {"grade": sum(gr) / n,
             "rel": sum(1 for g in gr if g >= 1),
             "strong": sum(1 for g in gr if g == 2),
             "ndcg": dcg(gr) / idcg}
        per_q[qid][a] = m
        for k in m: agg[a][k].append(m[k])

print(f"\nPooled blind-relevance IR eval — {len(per_q)} queries, top-{K}, grades 0/1/2\n")
print(f"  {'arm':<8} {'mean-grade':>11} {'#relevant':>10} {'#strong(2)':>11} {'nDCG@10':>8}")
for a in arms:
    n = len(agg[a]["grade"]) or 1
    print(f"  {a:<8} {sum(agg[a]['grade'])/n:>11.3f} {sum(agg[a]['rel'])/n:>10.2f} "
          f"{sum(agg[a]['strong'])/n:>11.2f} {sum(agg[a]['ndcg'])/n:>8.3f}")

# per-query nDCG table
print(f"\n  per-query nDCG@10:")
print("  " + f"{'qid':<6}" + "".join(f"{a:>11}" for a in arms))
for qid in per_q:
    print("  " + f"{qid:<6}" + "".join(f"{per_q[qid][a]['ndcg']:>11.3f}" for a in arms))

# pairwise win/tie/loss on nDCG for any pairs passed as argv[3] "a:b,c:d"
import sys as _s
pairs = _s.argv[3].split(",") if len(_s.argv) > 3 else []
for pr in pairs:
    x, y = pr.split(":")
    w = sum(1 for q in per_q if per_q[q][x]['ndcg'] > per_q[q][y]['ndcg'] + 1e-6)
    l = sum(1 for q in per_q if per_q[q][x]['ndcg'] < per_q[q][y]['ndcg'] - 1e-6)
    dx = sum(per_q[q][x]['ndcg'] for q in per_q)/len(per_q)
    dy = sum(per_q[q][y]['ndcg'] for q in per_q)/len(per_q)
    print(f"\n  {x} vs {y} (nDCG@10): win {w} / tie {len(per_q)-w-l} / loss {l}   (mean {dx:.3f} vs {dy:.3f}, Δ{dx-dy:+.3f})")
