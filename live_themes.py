# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "scikit-learn", "numpy"]
# ///
"""LIVE sector/macro OVERVIEW — the non-idiosyncratic mode. Cluster the WHOLE corpus by topic (embeddings,
where they earn their keep), then rank clusters by VOLUME x BREADTH (how many firms/sectors each spans).
This is aggregation, not per-firm retrieval — the broad 'what's driving the market' digest."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from qdrant_client import QdrantClient

HERE = Path(__file__).resolve().parent
TICKER_DOCS = json.load(open(HERE / "data" / "tmp" / "live_ticker_docs.json"))
SECTOR = json.load(open(HERE / "data" / "eg_runs" / "eg_live2" / "sector.json"))
K = 24

# invert: doc -> tickers -> sectors
doc_tickers = {}
for tk, docs in TICKER_DOCS.items():
    for d in docs:
        doc_tickers.setdefault(d, set()).add(tk)

C = QdrantClient(url="http://localhost:6333")
print("[themes] fetching corpus vectors …")
ids, vecs, titles, dates = [], [], [], []
nxt = None
while True:
    pts, nxt = C.scroll("market_color", limit=2000, offset=nxt, with_payload=True, with_vectors=True)
    for p in pts:
        if p.vector:
            ids.append(p.payload.get("doc_id")); vecs.append(p.vector)
            titles.append(p.payload.get("title") or ""); dates.append(p.payload.get("published_date") or "")
    if nxt is None:
        break
X = np.asarray(vecs, dtype=np.float32)
print(f"[themes] {len(X)} docs → KMeans k={K}")
km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(X)
lab = km.labels_

rows = []
for c in range(K):
    idx = np.where(lab == c)[0]
    if len(idx) == 0:
        continue
    tks, secs = set(), Counter()
    for i in idx:
        for tk in doc_tickers.get(ids[i], ()):
            tks.add(tk)
            if tk in SECTOR:
                secs[SECTOR[tk]] += 1
    # representative headline = doc nearest to centroid
    cen = km.cluster_centers_[c]
    near = idx[np.argmax(X[idx] @ cen)]
    rows.append({"n": len(idx), "tickers": len(tks), "sectors": len(secs),
                 "rep": titles[near][:78], "top_tk": [t for t, _ in Counter(
                     tk for i in idx for tk in doc_tickers.get(ids[i], ())).most_common(5)]})

# breadth score = docs * (1 + distinct sectors); broad macro/sector themes float up, single-name noise sinks
rows.sort(key=lambda r: -(r["n"] * (1 + r["sectors"])))
print(f"\n{'vol':>4}{'firms':>6}{'sect':>5}   THEME (representative headline)   ·   top tickers")
print("-" * 96)
for r in rows:
    kind = "BROAD" if r["sectors"] >= 3 else ("sector" if r["sectors"] == 2 else "narrow")
    print(f"{r['n']:>4}{r['tickers']:>6}{r['sectors']:>5}  [{kind:6}] {r['rep']}")
    if r["top_tk"]:
        print(f"{'':17}      · {', '.join(r['top_tk'])}")
