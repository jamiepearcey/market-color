# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Build blind evaluation bundles: for each question, a NAIVE (cosine) context
and a FUSION (cosine ∪ union, RRF) context, as text for an independent LLM."""
import json
from pathlib import Path
import duckdb

DATA = Path(__file__).resolve().parents[1] / "data"
CORPUS = (Path(__file__).resolve().parents[2] / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()
K = 5

res = json.loads((DATA / "ask_results.json").read_text())

def rrf(*lists, take=K):
    sc = {}
    for lst in lists:
        for r, d in enumerate(lst):
            sc[d] = sc.get(d, 0.0) + 1.0 / (60.0 + r)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])][:take]

ids = sorted({d for r in res for k in ("cosine", "union") for d in r.get(k, [])})
con = duckdb.connect()
con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
rows = con.execute(f"""SELECT doc_id, any_value(title) t, any_value(source_name) s,
                       any_value(published_date) d, any_value(body_text) b
                       FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id""").fetchall()
meta = {r[0]: {"title": r[1], "src": r[2], "date": r[3], "body": (r[4] or "")} for r in rows}

def block(doc_id, n=130):
    m = meta.get(doc_id, {})
    body = " ".join((m.get("body") or "").split()[:n])
    return f"[{m.get('date','?')} | {m.get('src','?')}] {m.get('title','?')}\n{body}"

def ctx(doc_ids):
    return "\n\n".join(f"({i+1}) {block(d)}" for i, d in enumerate(doc_ids[:K]))

batches = [[], [], [], []]
for qi, r in enumerate(res):
    naive = r["cosine"][:K]
    fusion = rrf(r["cosine"], r["union"])
    entry = {"n": qi + 1, "question": r["question"], "naive": ctx(naive), "fusion": ctx(fusion)}
    batches[qi % 4].append(entry)

for bi, batch in enumerate(batches, 1):
    lines = []
    for e in batch:
        lines.append("=" * 80)
        lines.append(f"QUESTION {e['n']}: {e['question']}")
        lines.append("\n### CONTEXT SET A (naive) ###\n" + e["naive"])
        lines.append("\n### CONTEXT SET B (fusion) ###\n" + e["fusion"])
        lines.append("")
    (DATA / f"eval_batch_{bi}.txt").write_text("\n".join(lines))
    print(f"wrote eval_batch_{bi}.txt with {len(batch)} questions")
