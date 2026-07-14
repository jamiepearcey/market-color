#!/usr/bin/env python3
"""Out-of-sample validation of the walk gate, with structural features.

PRE-REGISTERED primary test: fire the walk iff g1 (max unexplained-driver,
1 − max-cos(driver, context items)) >= 0.69 — threshold FROZEN from the
in-sample analysis before this data existed. Success = fired questions show
walk wins at a better rate than unfired ones.

Also recorded per question, for the learned-gate design (exploratory):
  g1        semantic uncoveredness of the worst driver
  g_temp    1 if earlier-dated upstream evidence exists for that driver
  g_graph   entity-graph hop distance from driver entities to context entities
            (sparse co-occurrence adjacency, BFS; 9 = disconnected)
  g_spec    driver specificity (word count of the driver text)

Walk candidates are TEMPORALLY FILTERED (candidate.published_epoch <=
endpoint's) — causality's arrow of time, per the domain-structure proposal.

Stages: mine (auto-generate fresh exclusion chains, excluding subjects used
in ANY previous eval), prep, judge-prep, score.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import GEN_TMPL, JUDGE_TMPL, _fact_line  # noqa: E402
from strat_blind import _embed_all, _v2_context  # noqa: E402

OUT = HERE / "gate_out"
G1_FROZEN = 0.69
WALK_SEEDS = 8
WALK_PER = 2
N_QUESTIONS = 20

USED_SUBJECT_TOKENS = set("""
boj bank_of_japan bitcoin gas tajikistan rba inflation ether coinbase dollar
russia crude usd/jpy japan copper nasdaq won hang_seng sable micron guinea
hormuz german sweden stablecoin korea fomc gbp aerovironment blue_origin
ukraine cook silver gold riot oil lng jet semis att amex ucits eni
""".split())


def _hay(f):
    return (str(f.get("claim", "")) + " " + str(f.get("cause", ""))).lower()


def cmd_mine(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    seen_subj = set()
    out = []
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
        if any(tok.replace("_", " ") in subj for tok in USED_SUBJECT_TOKENS):
            continue
        end_docs = {g["doc_id"] for g in rows if subj in _hay(g)} - {f["doc_id"]}
        root_docs = {g["doc_id"] for g in rows
                     if any(e in _hay(g) for e in ce)} - {f["doc_id"]}
        if len(end_docs) >= 2 and len(root_docs) >= 3 and desk_count[f.get("desk")] < 4:
            seen_subj.add(subj)
            desk_count[f.get("desk")] += 1
            q = claim.rstrip(".")
            out.append({
                "id": f"G-{len(out):02d}-{re.sub(r'[^a-z0-9]+', '-', subj)[:18]}",
                "query": f"Explain why this happened: {q}.",
                "exclude_doc_ids": [f["doc_id"]],
                "endpoint_subject": subj,
            })
            if len(out) >= N_QUESTIONS:
                break
    OUT.mkdir(exist_ok=True)
    with open(HERE / "blind_questions_gate.jsonl", "w") as fh:
        for q in out:
            fh.write(json.dumps(q) + "\n")
    print(f"mined {len(out)} fresh exclusion chains")


def _graph_distance(driver_ents, ctx_ents, adj, max_hops=4):
    if not driver_ents or not ctx_ents:
        return 9
    frontier = set(driver_ents)
    seen = set(frontier)
    for hop in range(max_hops + 1):
        if frontier & ctx_ents:
            return hop
        nxt = set()
        for e in frontier:
            nxt |= adj.get(e, set())
        frontier = nxt - seen
        seen |= frontier
        if not frontier:
            break
    return 9


def cmd_prep(args):
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    extracted_docs = {f["doc_id"] for f in facts}
    by_doc: dict[str, list] = {}
    adj: dict[str, set] = {}
    for f in facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
        es = list(set(f.get("entities") or []))
        for i, a in enumerate(es):
            for b in es[i + 1:]:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    print(f"[gate] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_gate.jsonl") if l.strip()]
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
        ctx_ents = set()
        for _, f in fact_hits[:20]:
            ctx_ents |= set(f.get("entities") or [])

        drivers = [(f, str(f["cause"])) for _, f in fact_hits[:WALK_SEEDS]
                   if f.get("cause") and str(f["cause"]).lower() not in ("none", "null")]
        g1 = g_temp = 0.0
        g_graph, g_spec = 9, 0
        walk_lines, used_fids = [], set()
        if drivers:
            dvecs = _embed_all([c for _, c in drivers])
            uncov = [1 - float(np.max(ctx_vecs @ dv)) for dv in dvecs]
            g1 = float(max(uncov))
            worst = int(np.argmax(uncov))
            worst_f, worst_c = drivers[worst]
            g_spec = len(worst_c.split())
            g_graph = _graph_distance(
                set(worst_f.get("cause_entities") or []), ctx_ents, adj)
            for (parent, _), dv in zip(drivers, dvecs, strict=True):
                p_epoch = parent.get("published_epoch") or 0
                sims_d = fmat @ dv
                added = 0
                for i in np.argsort(-sims_d):
                    f = facts[int(i)]
                    f_epoch = f.get("published_epoch") or 0
                    if (f["fact_id"] in used_fids or f["doc_id"] in excluded
                            or f["doc_id"] == parent["doc_id"]
                            or (p_epoch and f_epoch and f_epoch > p_epoch)):  # arrow of time
                        continue
                    cause = (f" DRIVER: {f['cause']}"
                             if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else "")
                    walk_lines.append(
                        f"[W{len(walk_lines)}] UPSTREAM FACT: {f.get('claim')} "
                        f"(source: {f.get('source_name')}, {f.get('published_date')}){cause}")
                    used_fids.add(f["fact_id"])
                    if parent is worst_f:
                        g_temp = 1.0
                    added += 1
                    if added >= WALK_PER:
                        break

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
        (OUT / f"gen_{q['id']}_v2walk.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(renum)))
        manifest.append({"id": q["id"], "query": q["query"], "budget": budget,
                         "g1": round(g1, 3), "g_temp": g_temp, "g_graph": g_graph,
                         "g_spec": g_spec, "fires": g1 >= G1_FROZEN,
                         "walk_lines": len(walk_lines)})
        print(f"  {q['id']:<26} g1={g1:.3f} fires={g1 >= G1_FROZEN} "
              f"g_graph={g_graph} g_temp={g_temp}", file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    fired = sum(1 for m in manifest if m["fires"])
    print(f"prepared {len(manifest)} questions; gate fires on {fired}")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        a_p = OUT / f"ans_{m['id']}_v2walk.txt"
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
        print(f"  {m['id']:<26} {res}  fires={m['fires']} g1={m['g1']} "
              f"g_graph={m['g_graph']} g_temp={m['g_temp']}")
    def rec(sel):
        return {"W": sum(1 for m, r in rows if sel(m) and r == "W"),
                "L": sum(1 for m, r in rows if sel(m) and r == "L"),
                "T": sum(1 for m, r in rows if sel(m) and r == "T")}
    print(json.dumps({
        "PRIMARY_fired_at_frozen_0.69": rec(lambda m: m["fires"]),
        "unfired": rec(lambda m: not m["fires"]),
        "all": rec(lambda m: True),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mine", "prep", "judge-prep", "score"])
    args = p.parse_args()
    {"mine": cmd_mine, "prep": cmd_prep, "judge-prep": cmd_judge_prep,
     "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
