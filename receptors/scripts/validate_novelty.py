# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Validate the novelty score: are low-novelty chunks really restatements of a
prior story, are high-novelty chunks really new, and which docs are the 'origin'
stories that everyone else echoes?"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

rows = [json.loads(l) for l in (D / "novelty.jsonl").read_text().splitlines() if l.strip()]
ids = sorted({r["doc_id"] for r in rows} | {r["nn_doc"] for r in rows})
con = duckdb.connect()
con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
meta = {r[0]: {"t": (r[1] or "?"), "d": (r[2] or "?"), "s": (r[3] or "?")}
        for r in con.execute(f"""SELECT doc_id, any_value(title), any_value(published_date),
                                 any_value(source_name) FROM '{CORPUS}' JOIN want USING(doc_id)
                                 GROUP BY doc_id""").fetchall()}
def t(d): return meta.get(d, {}).get("t", "?")[:62]
def src(d): return meta.get(d, {}).get("s", "?")

# day-bucketed trend (does the corpus get more repetitive as history accumulates?)
byday = defaultdict(list)
for r in rows:
    byday[meta.get(r["doc_id"], {}).get("d", "?")].append(r["novelty"])
print("=== mean novelty by DAY (as prior history accumulates) ===")
for day in sorted(byday):
    v = byday[day]
    print(f"  {day} | n={len(v):4} | mean-novelty {sum(v)/len(v):.3f}")

# de-dup to one lowest/highest-novelty chunk per doc
best_per_doc = {}
for r in rows:
    d = r["doc_id"]
    if d not in best_per_doc or r["novelty"] < best_per_doc[d]["novelty"]:
        best_per_doc[d] = r
docs = list(best_per_doc.values())

print("\n=== LEAST novel (echoes): is it a restatement of the matched prior doc? ===")
for r in sorted(docs, key=lambda r: r["novelty"])[:8]:
    print(f"  nov={r['novelty']:.3f}  [{src(r['doc_id'])}] {t(r['doc_id'])}")
    print(f"            echoes -> [{src(r['nn_doc'])}] {t(r['nn_doc'])}")

print("\n=== MOST novel (new information) ===")
for r in sorted(docs, key=lambda r: -r["novelty"])[:8]:
    print(f"  nov={r['novelty']:.3f}  [{src(r['doc_id'])}] {t(r['doc_id'])}")

print("\n=== 'origin' stories most echoed by later chunks (nn_doc frequency) ===")
for d, c in Counter(r["nn_doc"] for r in rows).most_common(10):
    print(f"  echoed {c:4}x  [{src(d)}] {t(d)}")
