#!/usr/bin/env python3
"""Definitive weighted-walk pilot — the walk's best configuration, tested
properly.

PRE-REGISTERED ANALYSIS (fixed before any judging):
  Primary: across all verified questions, decisive double-judged verdicts
  (win = both A/B orders) — one-sided exact binomial test of walk wins vs
  v2 wins. No thresholds, no subgroups in the primary.
  Secondary (reported, not confirmatory): outcomes by g1 tercile.

Walk candidate scoring (the weighted priors, replacing raw cosine):
  score = cos(driver, candidate) * conf_w * temporal_w * graph_w
    conf_w     = 0.5 + confidence/2
    temporal_w = 1.0 if event_time(cand) <= event_time(parent) + 1 day
                 0.7 if either event time unknown (soft — parsing is approx)
                 0.2 if candidate's event postdates the parent's (soft veto)
    graph_w    = 1.2 if candidate entities intersect the parent's
                 cause_entities, else 1.0
  Event time = dateutil-parsed `time` field anchored on published_date
  (84.4% coverage measured).

Question factory: auto-mined canonical-exclusion chains, then a VERIFICATION
pass (haiku): well-posed? answer not leaked by the query? non-trivial chain?
Only KEEPs enter the benchmark.

Stages: mine -> verify -> prep -> (OUT=ww_out run_ablation.sh gen) ->
judge-prep -> (OUT=ww_out run_ablation.sh judge) -> score.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import GEN_TMPL, JUDGE_TMPL, _fact_line  # noqa: E402
from strat_blind import _embed_all, _v2_context  # noqa: E402

OUT = HERE / "ww_out"
CLAUDE = "/opt/homebrew/bin/claude"
N_MINE = 70
N_KEEP = 40
WALK_SEEDS = 8
WALK_PER = 2

MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"

# dateutil drops regional tznames with a warning — exactly the EM sources we
# care about (KST=Yonhap, IST=ET/Mint, ...). Offsets in seconds.
TZINFOS = {"KST": 9 * 3600, "JST": 9 * 3600, "IST": 19800, "SGT": 8 * 3600,
           "HKT": 8 * 3600, "CST": 8 * 3600, "WAT": 3600, "SAST": 2 * 3600,
           "BRT": -3 * 3600, "GST": 4 * 3600, "AEST": 10 * 3600,
           "EST": -5 * 3600, "EDT": -4 * 3600, "BST": 3600}


def norm_event_time(t, pub):
    from dateutil import parser as dp
    if not t or str(t).lower() in ("none", "null", "na"):
        return None
    s = str(t).strip()
    low = s.lower()
    try:
        pubd = dp.parse(str(pub), tzinfos=TZINFOS) if pub else None
    except Exception:
        pubd = None
    if re.search(r"\b(today|this week|recent|now|currently)\b", low) and pubd:
        return pubd.date()
    if re.search(r"\byesterday\b", low) and pubd:
        return (pubd - dt.timedelta(days=1)).date()
    if re.fullmatch(r"(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", low) and pubd:
        return pubd.date()
    if re.search(rf"\b({MONTHS})\b", low) or re.search(r"\b20\d\d\b", low) \
            or re.search(r"\d{1,2}[/-]\d{1,2}", low):
        try:
            return dp.parse(s, default=pubd, fuzzy=True,
                            tzinfos=TZINFOS).date()
        except Exception:
            return None
    if re.search(r"\b(q[1-4]|h[12]|first half|second half|(first|second|third|fourth) quarter)\b", low) and pubd:
        return pubd.date()
    return None


def _hay(f):
    return (str(f.get("claim", "")) + " " + str(f.get("cause", ""))).lower()


def cmd_mine(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    used_subjects = set()
    for qf in ("blind_questions_gate.jsonl",):
        p = HERE / qf
        if p.exists():
            for l in open(p):
                used_subjects.add(json.loads(l).get("endpoint_subject", ""))
    seen_subj, out = set(used_subjects), []
    desk_count = collections.Counter()
    for f in rows:
        cause = str(f.get("cause") or "")
        if not cause or cause.lower() in ("none", "null") or len(cause.split()) < 5:
            continue
        subj = str(f.get("subject") or "").lower().strip()
        claim = str(f.get("claim") or "").strip()
        ce = [e for e in (f.get("cause_entities") or []) if len(e) > 3]
        if not subj or len(subj) < 4 or not ce or subj in seen_subj or len(claim) < 40:
            continue
        end_docs = {g["doc_id"] for g in rows if subj in _hay(g)} - {f["doc_id"]}
        root_docs = {g["doc_id"] for g in rows if any(e in _hay(g) for e in ce)} - {f["doc_id"]}
        if len(end_docs) >= 2 and len(root_docs) >= 3 and desk_count[f.get("desk")] < 6:
            seen_subj.add(subj)
            desk_count[f.get("desk")] += 1
            out.append({
                "id": f"W-{len(out):02d}-{re.sub(r'[^a-z0-9]+', '-', subj)[:16]}",
                "query": f"Explain why this happened: {claim.rstrip('.')}.",
                "canonical_claim": claim, "canonical_cause": cause,
                "exclude_doc_ids": [f["doc_id"]], "endpoint_subject": subj,
            })
            if len(out) >= N_MINE:
                break
    OUT.mkdir(exist_ok=True)
    (OUT / "mined.jsonl").write_text("\n".join(json.dumps(q) for q in out) + "\n")
    print(f"mined {len(out)} candidates")


VERIFY_TMPL = """You are auditing an auto-generated benchmark question for a causal-inference test.

QUESTION shown to systems: {query}
GROUND-TRUTH CAUSE (from a document that will be HIDDEN from all systems): {cause}

Answer three checks:
1. WELL_POSED: is the question meaningful and answerable in principle? (yes/no)
2. LEAKED: does the question text itself already state or strongly imply the ground-truth cause? (yes/no)
3. TRIVIAL: is the cause a circular restatement of the question rather than a distinct upstream event? (yes/no)

Reply with exactly one line: KEEP or DROP
(KEEP only if WELL_POSED=yes, LEAKED=no, TRIVIAL=no.) Then one short reason."""


def cmd_verify(args):
    """Resume-safe: one verdict file per candidate; rerun continues."""
    mined = [json.loads(l) for l in open(OUT / "mined.jsonl") if l.strip()]
    kept = []
    for q in mined:
        if len(kept) >= N_KEEP:
            break
        vf = OUT / f"verify_{q['id']}.txt"
        if not vf.exists():
            prompt = VERIFY_TMPL.format(query=q["query"], cause=q["canonical_cause"])
            r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                               input=prompt, capture_output=True, text=True, timeout=120)
            vf.write_text(r.stdout or "ERROR")
        verdict = "KEEP" in vf.read_text().split("\n")[0].upper()
        print(f"  {q['id']:<24} {'KEEP' if verdict else 'DROP'}", file=sys.stderr)
        if verdict:
            kept.append(q)
    (HERE / "blind_questions_ww.jsonl").write_text(
        "\n".join(json.dumps(q) for q in kept) + "\n")
    print(f"verified: kept {len(kept)} (target {N_KEEP})")


def cmd_prep(args):
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    for f in facts:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
    extracted_docs = {f["doc_id"] for f in facts}
    by_doc: dict[str, list] = {}
    for f in facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
    print(f"[ww] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_ww.jsonl") if l.strip()]
    manifest = []
    for q in questions:
        excluded = set(q["exclude_doc_ids"])
        items, budget = _v2_context(q["query"], facts, fmat, client, by_doc,
                                    extracted_docs, excluded)
        (OUT / f"gen_{q['id']}_v2.txt").write_text(GEN_TMPL.format(query=q["query"], items=items))

        qv = _embed_all([q["query"]])[0]
        sims = fmat @ qv
        order = [int(i) for i in np.argsort(-sims)
                 if facts[int(i)]["doc_id"] not in excluded][:50]
        fact_hits = [(float(sims[i]), facts[i]) for i in order]
        ctx_vecs = fmat[order]
        drivers = [(f, str(f["cause"])) for _, f in fact_hits[:WALK_SEEDS]
                   if f.get("cause") and str(f["cause"]).lower() not in ("none", "null")]
        walk_lines, used_fids = [], set()
        g1 = 0.0
        if drivers:
            dvecs = _embed_all([c for _, c in drivers])
            g1 = float(max(1 - float(np.max(ctx_vecs @ dv)) for dv in dvecs))
            for (parent, _), dv in zip(drivers, dvecs, strict=True):
                p_et = parent["_etime"]
                p_ce = set(parent.get("cause_entities") or [])
                cand_sims = fmat @ dv
                scored = []
                for i in np.argsort(-cand_sims)[:200]:
                    f = facts[int(i)]
                    if (f["fact_id"] in used_fids or f["doc_id"] in excluded
                            or f["doc_id"] == parent["doc_id"]):
                        continue
                    conf_w = 0.5 + float(f.get("confidence") or 0.5) / 2
                    c_et = f["_etime"]
                    if p_et and c_et:
                        temporal_w = 1.0 if c_et <= p_et + dt.timedelta(days=1) else 0.2
                    else:
                        temporal_w = 0.7
                    graph_w = 1.2 if p_ce & set(f.get("entities") or []) else 1.0
                    scored.append((float(cand_sims[i]) * conf_w * temporal_w * graph_w, f))
                scored.sort(key=lambda t: -t[0])
                for _, f in scored[:WALK_PER]:
                    cause = (f" DRIVER: {f['cause']}"
                             if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else "")
                    walk_lines.append(
                        f"[W{len(walk_lines)}] UPSTREAM FACT: {f.get('claim')} "
                        f"(source: {f.get('source_name')}, {f.get('published_date')}){cause}")
                    used_fids.add(f["fact_id"])
        walk_chars = sum(len(l) for l in walk_lines)
        base_lines = items.split("\n")
        kept, used = [], 0
        for line in base_lines:
            if used + len(line) > budget - walk_chars and kept:
                break
            kept.append(line)
            used += len(line)
        renum = [re.sub(r"^\[\w+\]", f"[{i + 1}]", l)
                 for i, l in enumerate(kept + walk_lines)]
        (OUT / f"gen_{q['id']}_wwalk.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(renum)))
        manifest.append({"id": q["id"], "query": q["query"], "g1": round(g1, 3),
                         "walk_lines": len(walk_lines)})
        print(f"  {q['id']:<24} g1={g1:.3f} walk_lines={len(walk_lines)}", file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} questions")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        a_p = OUT / f"ans_{m['id']}_wwalk.txt"
        b_p = OUT / f"ans_{m['id']}_v2.txt"
        if not a_p.exists() or not b_p.exists():
            continue
        a, b = a_p.read_text().strip(), b_p.read_text().strip()
        (OUT / f"judge_{m['id']}_o1.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=a, b=b))
        (OUT / f"judge_{m['id']}_o2.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=b, b=a))
        n += 1
    print(f"prepared {n} pairs x 2 orders")


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = []
    for m in manifest:
        vs = []
        for order, walk_is_a in (("o1", True), ("o2", False)):
            vp = OUT / f"verdict_{m['id']}_{order}.txt"
            mt = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text()) if vp.exists() else None
            vs.append(None if not mt or mt.group(1) == "TIE"
                      else (mt.group(1) == "A") == walk_is_a)
        both = [v for v in vs if v is not None]
        res = ("W" if len(both) == 2 and all(both) else
               "L" if len(both) == 2 and not any(both) else "T")
        rows.append((m, res))
        print(f"  {m['id']:<24} {res}  g1={m['g1']}")
    W = sum(1 for _, r in rows if r == "W")
    L = sum(1 for _, r in rows if r == "L")
    T = sum(1 for _, r in rows if r == "T")
    n = W + L
    p = sum(math.comb(n, k) for k in range(W, n + 1)) / 2**n if n else 1.0
    # secondary: by g1 tercile
    g1s = sorted(m["g1"] for m, _ in rows)
    t1, t2 = g1s[len(g1s) // 3], g1s[2 * len(g1s) // 3]
    terc = {"low": [], "mid": [], "high": []}
    for m, r in rows:
        terc["low" if m["g1"] <= t1 else "high" if m["g1"] > t2 else "mid"].append(r)
    print(json.dumps({
        "PRIMARY": {"walk_wins": W, "v2_wins": L, "ties": T,
                    "one_sided_binomial_p": round(p, 4)},
        "SECONDARY_by_g1_tercile": {k: {"W": v.count("W"), "L": v.count("L"),
                                        "T": v.count("T")} for k, v in terc.items()},
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mine", "verify", "prep", "judge-prep", "score"])
    args = p.parse_args()
    {"mine": cmd_mine, "verify": cmd_verify, "prep": cmd_prep,
     "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
