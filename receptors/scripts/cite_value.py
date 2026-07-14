# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Quantify the VALUE of low-cosine ("mis-aligned"/WEAK) citations.

The gate flags a cite WEAK when claim<->passage COSINE < 0.40. But the receptor
thesis is: low cosine + HIGH CAUSAL score = the cross-vocabulary reach cosine cannot
make. So a WEAK cite is not necessarily a mis-cite — it may be a valuable causal link
a topical retriever would never surface. This script computes, for every cite in every
report, BOTH scores and splits the WEAK cites into:
  REACH  = low cosine (<0.40) but HIGH causal (>= --cthr)  -> cross-vocabulary value
  NOISE  = low cosine AND low causal                        -> genuine mis-cite
and measures whether REACH cites land on corroborated / counter-evidence claims (i.e.
whether they are load-bearing on the hard claims a plain-RAG system could not make).

Usage: uv run scripts/cite_value.py [--cthr 0.45]
"""
import json, glob, argparse
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--cthr", type=float, default=0.45, help="causal-score threshold for REACH")
ap.add_argument("--wthr", type=float, default=0.40, help="cosine below this = WEAK (gate default)")
a = ap.parse_args()

ch = np.load(D / "chunks.npy").astype(np.float32)
ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
T = np.load(D / "chunks_transported.npy").astype(np.float32)  # normalize(chunk·W)
meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
chunks_by_doc = {}
for i, m in enumerate(meta):
    chunks_by_doc.setdefault(m["doc_id"], []).append(i)
# resolve short (8-char) ids the ledgers use
full_ids = list(chunks_by_doc)
def resolve(did):
    if did in chunks_by_doc: return did
    hits = [f for f in full_ids if f.startswith(did)]
    return hits[0] if len(hits) == 1 else None

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def analyze(arm):
    reports = sorted(glob.glob(f"data/ab/gate_q*_{arm}.json"),
                     key=lambda p: int(p.split("_q")[1].split(f"_{arm}")[0]))
    # collect (claim_text, status, doc_id) for every cite
    rows = []
    for f in reports:
        for c in json.load(open(f)):
            for ci in c["cites"]:
                if ci["status"] == "MISSING":
                    continue
                rows.append((c["claim"], c["status"], ci["doc_id"]))
    claims = [r[0] for r in rows]
    # embed unique claims once
    uniq = list(dict.fromkeys(claims))
    cv = np.array(list(model.embed(uniq)), dtype=np.float32)
    cv /= (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-9)
    cvec = {t: cv[i] for i, t in enumerate(uniq)}
    n_cos_weak = n_reach = n_noise = 0
    reach_on_hard = reach_total = 0
    cos_all = []; caus_all = []
    weak_caus = []
    n_lift = lift_on_hard = lift_total = 0   # cites where causal out-ranks cosine by >=0.10
    lifts = []
    for claim, status, did in rows:
        rid = resolve(did)
        if rid is None:
            continue
        idx = chunks_by_doc[rid]
        q = cvec[claim]
        cos = float(np.max(ch[idx] @ q))          # topical alignment (gate's 'align')
        caus = float(np.max(T[idx] @ q))          # causal (W) alignment
        cos_all.append(cos); caus_all.append(caus)
        lift = caus - cos; lifts.append(lift)     # how much W out-ranks topical cosine
        if lift >= 0.10:                          # causal materially exceeds topical
            n_lift += 1; lift_total += 1
            if status in ("corroborated", "contradicted"):
                lift_on_hard += 1
        if cos < a.wthr:                          # WEAK topically
            n_cos_weak += 1
            weak_caus.append(caus)
            if caus >= a.cthr:
                n_reach += 1
                reach_total += 1
                if status in ("corroborated", "contradicted"):
                    reach_on_hard += 1
            else:
                n_noise += 1
    tot = len(cos_all)
    return dict(arm=arm, cites=tot, weak=n_cos_weak,
                reach=n_reach, noise=n_noise,
                reach_pct_of_weak=(100*n_reach/n_cos_weak if n_cos_weak else 0),
                reach_on_hard=reach_on_hard, reach_total=reach_total,
                mean_cos=float(np.mean(cos_all)), mean_caus=float(np.mean(caus_all)),
                mean_weak_caus=(float(np.mean(weak_caus)) if weak_caus else 0),
                reach_share_all=(100*n_reach/tot if tot else 0),
                n_lift=n_lift, lift_pct=(100*n_lift/tot if tot else 0),
                lift_on_hard=lift_on_hard, lift_total=lift_total,
                mean_lift=float(np.mean(lifts)) if lifts else 0)

for arm in ["base", "full", "graph"]:
    r = analyze(arm)
    print(f"\n== {arm.upper()} ==")
    print(f"  cites resolved: {r['cites']}   WEAK (cos<{a.wthr}): {r['weak']} ({100*r['weak']/r['cites']:.1f}%)")
    print(f"  of WEAK: REACH (causal>={a.cthr}) = {r['reach']} ({r['reach_pct_of_weak']:.0f}% of weak)   NOISE = {r['noise']}")
    print(f"  REACH cites as share of ALL cites: {r['reach_share_all']:.1f}%")
    print(f"  REACH cites landing on corroborated/counter-evidence (hard) claims: {r['reach_on_hard']}/{r['reach_total']}")
    print(f"  CAUSAL-LIFT cites (causal-cosine>=0.10, W out-ranks topical): {r['n_lift']} ({r['lift_pct']:.1f}% of cites)")
    print(f"    of those, landing on hard (corroborated/counter-evidence) claims: {r['lift_on_hard']}/{r['lift_total']}")
    print(f"  mean cosine {r['mean_cos']:.3f}  mean causal {r['mean_caus']:.3f}  mean causal-of-WEAK {r['mean_weak_caus']:.3f}  mean lift {r['mean_lift']:+.3f}")
