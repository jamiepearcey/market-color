# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""
Rich, per-FACT export for the second-generation receptors (atlas / impact /
latency / source / consensus / regime).

Unlike export_for_receptors.py (which aggregates one row per *document*), this
keeps one row per *atomic fact* and carries every free label already present in
facts.parquet: predicate, signed direction, confidence, parsed magnitude,
source, desk(s), entities, cause_entities, and the publish epoch. The claim text
is embedded with the same model the whole project uses
(sentence-transformers/all-MiniLM-L6-v2, 384-d, L2-normalised).

Outputs (into receptors/data/):
  facts.npy    (N, 384) float32, L2-normalised, row i <-> facts.jsonl line i
  facts.jsonl  N lines: {fact_id, doc_id, published_epoch, date, predicate,
                         direction (+1/-1/0 signed), confidence, magnitude,
                         source_name, desk, desks[], entities[], cause_entities[]}

Row order is deterministic: (published_epoch, fact_id).
"""
import json, re, datetime as dt
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # research/market-color
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)
FACTS = ROOT / "facts_work" / "facts.parquet"

ALIASES = {
    "us": "united states", "u.s.": "united states", "usa": "united states",
    "fed": "federal reserve", "the fed": "federal reserve",
    "uk": "united kingdom", "ecb": "european central bank",
    "boj": "bank of japan", "rbnz": "reserve bank of new zealand",
}

def norm(e: str) -> str:
    e = (e or "").strip().lower()
    return ALIASES.get(e, e)

def to_epoch(ts: str) -> int:
    if not ts:
        return 0
    try:
        return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except Exception:
        try:
            return int(dt.datetime.fromisoformat(ts + "T00:00:00+00:00").timestamp())
        except Exception:
            return 0

_num = re.compile(r"[-+]?\d[\d,]*\.?\d*")
def parse_mag(s: str):
    """Best-effort first numeric magnitude (unit-agnostic). NaN if none."""
    if not s:
        return float("nan")
    m = _num.search(s.replace(",", ""))
    if not m:
        return float("nan")
    try:
        v = float(m.group())
    except Exception:
        return float("nan")
    low = s.lower()
    if "bps" in low or "basis point" in low:
        v /= 100.0            # bps -> pct-points, rough common scale
    return v

def dir_sign(d: str) -> float:
    d = (d or "").strip().lower()
    if d == "up":
        return 1.0
    if d == "down":
        return -1.0
    return 0.0

con = duckdb.connect()
print("loading facts ...")
rows = con.execute(f"""
    SELECT fact_id, doc_id, published_utc, published_date, claim,
           predicate, direction, confidence, magnitude,
           source_name, desk, desks, entities, cause_entities
    FROM '{FACTS.as_posix()}'
    WHERE claim IS NOT NULL AND length(trim(claim)) > 0
""").fetchall()

facts = []
for (fid, did, put, pdate, claim, pred, direction, conf, mag,
     src, desk, desks, ents, causes) in rows:
    ents = sorted({norm(e) for e in (ents or []) if e and e.strip()})
    causes = sorted({norm(e) for e in (causes or []) if e and e.strip()})
    desks = sorted({d for d in (desks or []) if d and d.strip()})
    facts.append({
        "fact_id": fid,
        "doc_id": did,
        "published_epoch": to_epoch(put or (pdate + "T00:00:00+00:00" if pdate else "")),
        "date": (pdate or (put or "")[:10]),
        "claim": claim,
        "predicate": (pred or "other"),
        "direction": dir_sign(direction),
        "confidence": float(conf or 0.0),
        "magnitude": parse_mag(mag),
        "source_name": (src or ""),
        "desk": (desk or "other"),
        "desks": desks,
        "entities": ents,
        "cause_entities": causes,
    })

facts.sort(key=lambda f: (f["published_epoch"], f["fact_id"]))
print(f"  {len(facts)} facts")

print("embedding claims (all-MiniLM-L6-v2) ...")
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
vecs = np.array(list(model.embed([f["claim"] for f in facts])), dtype=np.float32)
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
print(f"  embeddings {vecs.shape}")

np.save(OUT / "facts.npy", vecs)
with open(OUT / "facts.jsonl", "w") as f:
    for r in facts:
        r = dict(r)
        # keep a short claim for readable interpretability at the extremes
        r["claim"] = (r.get("claim") or "")[:240]
        # json can't encode NaN by default in strict readers; emit null instead
        if r["magnitude"] != r["magnitude"]:
            r["magnitude"] = None
        f.write(json.dumps(r) + "\n")
print(f"wrote {OUT/'facts.npy'} and {OUT/'facts.jsonl'}")
