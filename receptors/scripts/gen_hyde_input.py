# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Assemble first-pass context for LLM hypothesis distillation: each question +
titles/snippets of its naive-cosine top results. The LLM will read these and
distill concrete hypothesis search-queries."""
import json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

res = json.loads((D / "ask_results.json").read_text())
ids = sorted({d for r in res for d in r["cosine"][:5]})
con = duckdb.connect(); con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
meta = {r[0]: (r[1] or "?", (r[2] or "")) for r in con.execute(
    f"SELECT doc_id, any_value(title), any_value(body_text) FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id").fetchall()}

lines = []
for i, r in enumerate(res):
    lines.append("=" * 80)
    lines.append(f"QUESTION {i}: {r['question']}")
    lines.append("First-pass retrieved snippets:")
    for d in r["cosine"][:5]:
        t, b = meta.get(d, ("?", ""))
        snip = " ".join(b.split()[:45])
        lines.append(f"  - {t[:70]} :: {snip}")
    lines.append("")
(D / "hyde_input.txt").write_text("\n".join(lines))
print(f"wrote hyde_input.txt for {len(res)} questions")
