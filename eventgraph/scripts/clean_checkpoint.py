# /// script
# requires-python = ">=3.10"
# ///
"""
Coerce numeric fields that the 8B sometimes emits as STRINGS ("25" -> 25) back to
numbers in an extractions.jsonl checkpoint, so the strict Rust `ingest` checkpoint
reader (serde_json::from_value::<DocExtraction>) accepts them instead of dropping
the whole doc to a mock extraction. Idempotent; writes <in>.clean.jsonl (or --out).

Usage: uv run clean_checkpoint.py --in /tmp/eg100k/extractions.jsonl
"""
import json, argparse

# (array-key, [f64 fields], [i64 fields]); plus top-level sentiment_overall (f64)
F64 = {
    "events": ["expected", "actual", "prior"],
    "causal_edges": ["magnitude_value"],
    "sensitivities": ["magnitude_value"],
    "sentiments": ["polarity"],
    "propositions": ["probability"],
    "figures": ["value", "deal_value"],
    "relations": ["deal_value"],
}
I64 = {
    "sensitivities": ["sign"],
    "events": ["evidence_chunk"],
    "causal_edges": ["evidence_chunk"],
    "sensitivities_": ["evidence_chunk"],  # noqa (evidence handled per-array below)
    "sentiments": ["evidence_chunk"],
    "propositions": ["evidence_chunk"],
    "figures": ["evidence_chunk"],
    "relations": ["evidence_chunk"],
}

def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip().replace(",", "").rstrip("%")
        try:
            f = float(s)
            return int(f) if f.is_integer() and "." not in s else f
        except ValueError:
            return None
    return None

def as_int(v):
    n = num(v)
    return int(n) if isinstance(n, (int, float)) else None

def clean(ex):
    if not isinstance(ex, dict):
        return ex
    if "sentiment_overall" in ex:
        ex["sentiment_overall"] = num(ex.get("sentiment_overall"))
    for key, items in list(ex.items()):
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            for fld in F64.get(key, []):
                if fld in it:
                    it[fld] = num(it[fld])
            if key == "sensitivities" and "sign" in it:
                it["sign"] = as_int(it["sign"])
            if "evidence_chunk" in it:
                it["evidence_chunk"] = as_int(it["evidence_chunk"])
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or (a.inp.rsplit(".", 1)[0] + ".clean.jsonl")
    n = fixed = 0
    with open(out, "w") as w:
        for l in open(a.inp):
            d = json.loads(l)
            before = json.dumps(d.get("ex"))
            d["ex"] = clean(d.get("ex"))
            if json.dumps(d.get("ex")) != before:
                fixed += 1
            w.write(json.dumps(d) + "\n")
            n += 1
    print(f"cleaned {n} rows ({fixed} had coerced numerics) -> {out}")

if __name__ == "__main__":
    main()
