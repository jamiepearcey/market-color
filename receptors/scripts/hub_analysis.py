# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "numpy"]
# ///
"""
Hub detection + a per-month HUB-DOMINATION INDEX from the extracted causal graph.

A "hub" = an entity with disproportionate CAUSE IN-DEGREE (how many distinct docs
attribute it as a cause). No LLM needed -- it's pure graph structure over
facts.parquet. The month-level concentration predicts when the transport operator
will struggle (hub-dominated months collapse the cause->effect asymmetry).

Usage:  uv run receptors/scripts/hub_analysis.py receptors/data_bloomberg/facts.parquet
"""
import sys, collections, math
import numpy as np
import pyarrow.parquet as pq

path = sys.argv[1] if len(sys.argv) > 1 else "receptors/data_bloomberg/facts.parquet"
t = pq.read_table(path).to_pylist()

# group edges by month (YYYY-MM) using the date column
by_month = collections.defaultdict(list)
for r in t:
    ym = (r.get("date") or "")[:7]
    by_month[ym].append(r)

def gini(x):
    x = np.sort(np.asarray(x, float));  n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    return (2*np.sum((np.arange(1, n+1))*x)/(n*x.sum())) - (n+1)/n

print(f"{'month':8} {'docs':>5} {'edges':>6} {'entities':>8} {'HHI':>6} {'Gini':>6} {'top10%':>7} {'#hubs':>6}  top hubs (cause-in-degree)")
for ym in sorted(by_month):
    rows = by_month[ym]
    docs = {r["doc_id"] for r in rows}
    # cause in-degree = # DISTINCT docs blaming an entity as a cause
    cause_docs = collections.defaultdict(set)
    for r in rows:
        cause_docs[r["cause"]].add(r["doc_id"])
    indeg = {e: len(ds) for e, ds in cause_docs.items()}
    n_causal_docs = len(docs)
    total = sum(indeg.values())
    shares = np.array(list(indeg.values()), float) / max(total, 1)
    hhi = float((shares**2).sum())                       # 0=diffuse, 1=one entity
    g = gini(list(indeg.values()))
    top = sorted(indeg.items(), key=lambda kv: -kv[1])
    top10_share = sum(v for _, v in top[:10]) / max(total, 1)
    # hub rule: blamed in >= 5% of causal docs (tune) -> a genuine over-represented cause
    thr = max(3, 0.05 * n_causal_docs)
    hubs = [(e, d) for e, d in top if d >= thr]
    hub_str = ", ".join(f"{e}({d})" for e, d in hubs[:6])
    print(f"{ym:8} {n_causal_docs:>5} {len(rows):>6} {len(indeg):>8} {hhi:>6.3f} {g:>6.3f} "
          f"{top10_share:>6.1%} {len(hubs):>6}  {hub_str}")

# overall (all months pooled)
print("\n=== interpretation ===")
print("HHI/Gini/top10% high  -> causal attribution concentrates on a few entities")
print("                         = hub-dominated = transport operator will underperform.")
print("#hubs = entities blamed in >=5% of that month's causal docs (a tunable threshold).")
