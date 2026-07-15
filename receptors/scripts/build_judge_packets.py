# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pool the candidate arms per query into a BLIND judge packet (TREC-style pooled
relevance). For each query: union the doc_ids across arms, dedupe, order
deterministically (by doc_id hash seeded on qid so arm identity leaks nothing),
and emit an anonymised list the judge rates 0-2. A separate key file records which
arm ranked each doc where — never shown to the judge.

Outputs:
  data/eval/packets.json  : {qid: {effect, docs:[{lid,title,snippet}]}}   (to judges)
  data/eval/key.json      : {qid: {lid: {doc_id, ranks:{arm:rank}}}}       (for scoring)
"""
import json, hashlib, sys
from pathlib import Path

D = Path("data/eval")
queries = json.load(open(D / "queries.json"))
# args: [suffix] [arm1,arm2,...]   (default = the original MiniLM 3-arm pool)
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cosine", "old", "new"]
cand = {a: json.load(open(D / f"cand_{a}.json")) for a in arms}

packets, key = {}, {}
for q in queries:
    qid = q["id"]
    pool = {}  # doc_id -> {title, snippet, ranks}
    for a in arms:
        for r in cand[a].get(qid, []):
            e = pool.setdefault(r["doc_id"], {"title": r["title"], "snippet": r["snippet"], "ranks": {}})
            e["ranks"][a] = r["rank"]
    # deterministic anonymised order: hash(qid|doc_id)
    items = sorted(pool.items(), key=lambda kv: hashlib.md5(f"{qid}|{kv[0]}".encode()).hexdigest())
    docs, kmap = [], {}
    for i, (did, e) in enumerate(items):
        lid = f"D{i+1:02d}"
        docs.append({"lid": lid, "title": e["title"], "snippet": e["snippet"]})
        kmap[lid] = {"doc_id": did, "ranks": e["ranks"]}
    packets[qid] = {"effect": q["effect"], "docs": docs}
    key[qid] = kmap

json.dump(packets, open(D / f"packets{suffix}.json", "w"), indent=1)
json.dump(key, open(D / f"key{suffix}.json", "w"), indent=1)
for qid, p in packets.items():
    json.dump(p, open(D / f"packet{suffix}_{qid}.json", "w"), indent=1)
tot = sum(len(p["docs"]) for p in packets.values())
print(f"wrote packets{suffix}.json ({len(packets)} queries, {tot} pooled docs, arms={arms}) + per-query files")
