# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Assemble a research dossier per question: topical evidence + distilled
hypotheses with their validation verdicts, so an analyst agent can write a report
under REPORT_PROTOCOL and a red-team can stress the PROCESS."""
import json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()
PICK = [0, 3]

res = json.loads((D / "ask_results.json").read_text())
hyps = {json.loads(l)["i"]: json.loads(l)["hyps"] for l in (D/"hypotheses.jsonl").read_text().splitlines() if l.strip()}
verd = {}
for f in ("hyde_valid_verdicts_1.jsonl", "hyde_valid_verdicts_2.jsonl"):
    for l in (D/f).read_text().splitlines():
        if l.strip():
            o = json.loads(l); verd[o["h"]] = o

ids = sorted({d for i in PICK for d in res[i]["cosine"][:6]})
con = duckdb.connect(); con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(x,) for x in ids])
meta = {r[0]: (r[1] or '?', r[2] or '', r[3] or '?') for r in con.execute(
    f"SELECT doc_id, any_value(title), any_value(body_text), any_value(source_name) FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id").fetchall()}

for i in PICK:
    L = [f"RESEARCH QUESTION: {res[i]['question']}", "", "=== TOPICAL EVIDENCE (retrieved) ==="]
    for j, d in enumerate(res[i]["cosine"][:6], 1):
        t, b, s = meta.get(d, ('?','','?'))
        L.append(f"[E{j}] ({s}) {t[:70]}")
        L.append(f"      {' '.join(b.split()[:55])}")
    L += ["", "=== CANDIDATE HYPOTHESES (LLM-distilled) + VALIDATION VERDICT ==="]
    for k, h in enumerate(hyps[i]):
        v = verd.get(f"{i}.{k}", {})
        L.append(f"[H{k}] {h}")
        L.append(f"      VALIDATION: {v.get('verdict','?').upper()} — unconfirmed: {v.get('unsupported_part','?')}")
    (D / f"dossier_{i}.txt").write_text("\n".join(L))
    print(f"wrote dossier_{i}.txt")
