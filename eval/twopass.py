#!/usr/bin/env python3
"""Two-pass (breadth -> nuance-request -> synthesis) answering, evaluated by
ANSWER-KEY GRADING: canonical-exclusion questions carry ground truth (the
excluded fact's cause), so a haiku grader scores answers 0/1/2 against it —
objective, cheap, no pairwise judging.

  grade-existing  retro-grade eval/ww_out answers (v2, wwalk) -> calibration
  prep            pass-1 breadth survey (haiku over 50 atomic facts ->
                  nuance requests), fetch parent passages for requested
                  items, write pass-2 synthesis prompts (gen_*_twopass.txt)
  (OUT=tp_out run_ablation.sh gen)   pass-2 synthesis via opus
  grade           grade twopass answers with the same answer key

Grading scale: 2 = names the specific root event/actor; 1 = generic gesture
in the right direction; 0 = misses or wrong.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import CHUNK_COLLECTION, _fact_line  # noqa: E402
from strat_blind import _embed_all  # noqa: E402

OUT = HERE / "tp_out"
WW = HERE / "ww_out"
CLAUDE = "/opt/homebrew/bin/claude"

GRADE_TMPL = """Grade an answer against ground truth.

QUESTION: {query}
GROUND-TRUTH ROOT CAUSE (from a source hidden from the answerer): {cause}

ANSWER:
{answer}

Score how well the answer identifies the ground-truth root cause:
2 = names the specific event/actor/decision in the ground truth (wording may differ)
1 = gestures at the right general direction but stays generic
0 = misses it or attributes a different cause

Reply with exactly one line: SCORE: 0 or SCORE: 1 or SCORE: 2
Then one short reason."""

SURVEY_TMPL = """You are preparing to answer a market question. Below is a BROAD survey of numbered atomic facts. Do NOT answer yet.

QUESTION: {query}

FACTS:
{items}

List the 3-5 fact numbers whose SOURCE DETAIL you most need to see to answer confidently — prioritize facts whose stated DRIVER needs verification or whose specifics (figures, actors, sequence) would change the answer. Reply with exactly one line:
NEED: <comma-separated fact numbers>"""

SYNTH_TMPL = """You are a market analyst. Form your answer from the two evidence layers below: (A) the BROAD PICTURE — numbered atomic facts surveying all the evidence, with computed CONSENSUS/DISSENT annotations where sources conflict; (B) NUANCE — full source passages for the facts flagged as decision-relevant.

Structure your answer as an analyst note (3-8 sentences):
1. The consensus explanation, weighed from the broad picture.
2. Where sources DISAGREE (direction or attributed cause), say so explicitly with the sources.
3. Any non-consensus or tail signal from the evidence that might matter more than the consensus, and why.
Cite item numbers in brackets. News is aggregated collective reasoning — distinguishing what consensus believes from what might actually be relevant is the value of the note.

QUESTION: {query}

(A) BROAD PICTURE:
{facts}

(B) NUANCE PASSAGES:
{passages}
"""


def _dissent_annotations(hits):
    """Mechanical disagreement detection over the fact head: same subject,
    conflicting direction across sources, or materially different causes."""
    import collections as _c
    by_subj = _c.defaultdict(list)
    for i, f in enumerate(hits):
        s = str(f.get("subject") or "").lower().strip()
        if len(s) >= 4:
            by_subj[s].append((i + 1, f))
    lines = []
    for subj, group in by_subj.items():
        if len(group) < 2:
            continue
        dirs = _c.defaultdict(set)
        for idx, f in group:
            d = str(f.get("direction") or "na")
            if d in ("up", "down"):
                dirs[d].add((idx, str(f.get("source_name") or "?")))
        if dirs.get("up") and dirs.get("down"):
            ups = ", ".join(f"[{i}] {s}" for i, s in sorted(dirs["up"]))
            downs = ", ".join(f"[{i}] {s}" for i, s in sorted(dirs["down"]))
            lines.append(f"DISSENT on '{subj}' direction: UP per {ups} vs DOWN per {downs}")
        causes = [(idx, str(f.get("source_name") or "?"), str(f.get("cause")))
                  for idx, f in group
                  if f.get("cause") and str(f["cause"]).lower() not in ("none", "null")]
        srcs = {c[1] for c in causes}
        if len(causes) >= 2 and len(srcs) >= 2:
            toks = [set(str(c[2]).lower().split()) for c in causes]
            overlap = len(toks[0] & toks[-1]) / max(len(toks[0] | toks[-1]), 1)
            if overlap < 0.2:
                a, b = causes[0], causes[-1]
                lines.append(f"DIVERGING CAUSES for '{subj}': [{a[0]}] {a[1]}: "
                             f"\"{a[2][:70]}\" vs [{b[0]}] {b[1]}: \"{b[2][:70]}\"")
    return lines[:6]


def _questions():
    return [json.loads(l) for l in open(HERE / "blind_questions_ww.jsonl") if l.strip()]


def _grade(query, cause, answer):
    prompt = GRADE_TMPL.format(query=query, cause=cause, answer=answer)
    r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                       input=prompt, capture_output=True, text=True, timeout=120)
    m = re.search(r"SCORE:\s*([012])", r.stdout or "")
    return int(m.group(1)) if m else None


def _grade_arm(questions, path_fn, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores = {}

    def job(q):
        cf = cache_dir / f"{q['id']}.txt"
        if cf.exists() and cf.read_text().strip().isdigit():
            return q["id"], int(cf.read_text().strip())
        p = path_fn(q)
        if not p.exists():
            return q["id"], None
        s = _grade(q["query"], q["canonical_cause"], p.read_text().strip())
        if s is not None:
            cf.write_text(str(s))
        return q["id"], s

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for qid, s in ex.map(job, questions):
            scores[qid] = s
    return scores


def cmd_grade_existing(args):
    qs = _questions()
    OUT.mkdir(exist_ok=True)
    for arm, path_fn in (("v2", lambda q: WW / f"ans_{q['id']}_v2.txt"),
                         ("wwalk", lambda q: WW / f"ans_{q['id']}_wwalk.txt")):
        sc = _grade_arm(qs, path_fn, OUT / f"grades_{arm}")
        vals = [v for v in sc.values() if v is not None]
        print(f"{arm}: n={len(vals)} mean={sum(vals)/len(vals):.2f} "
              f"dist={{0: {vals.count(0)}, 1: {vals.count(1)}, 2: {vals.count(2)}}}")


def cmd_prep(args):
    import pyarrow.parquet as pq
    from qdrant_client.http import models
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    print("[tp] embedding facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    qs = _questions()
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in qs:
        excluded = set(q["exclude_doc_ids"])
        qv = _embed_all([q["query"]])[0]
        sims = fmat @ qv
        order = [int(i) for i in np.argsort(-sims)
                 if facts[int(i)]["doc_id"] not in excluded][:50]
        hits = [facts[i] for i in order]
        dissent = _dissent_annotations(hits)
        fact_block = "\n".join(_fact_line(i + 1, f) for i, f in enumerate(hits))
        if dissent:
            fact_block += "\n--- COMPUTED CONSENSUS/DISSENT ---\n" + "\n".join(dissent)

        # pass 1: breadth survey -> nuance requests (haiku, cached)
        sf = OUT / f"survey_{q['id']}.txt"
        if not sf.exists():
            r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                               input=SURVEY_TMPL.format(query=q["query"], items=fact_block),
                               capture_output=True, text=True, timeout=180)
            sf.write_text(r.stdout or "")
        m = re.search(r"NEED:\s*([\d,\s]+)", sf.read_text())
        need = []
        if m:
            need = [int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= len(hits)][:5]
        if not need:
            need = [1, 2, 3]

        passages, seen_docs = [], set()
        for idx in need:
            f = hits[idx - 1]
            if f["doc_id"] in seen_docs:
                continue
            seen_docs.add(f["doc_id"])
            cv = _embed_all([str(f.get("claim") or "")])[0]
            pts = client.query_points(
                collection_name=CHUNK_COLLECTION, query=cv, limit=1, with_payload=True,
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=f["doc_id"]))])).points
            if pts:
                pl = pts[0].payload or {}
                passages.append(f"[P{len(passages) + 1} — detail for fact {idx}] "
                                f"{pl['text']} (source: {pl.get('source_name')}, "
                                f"{pl.get('published_date')})")
        (OUT / f"gen_{q['id']}_twopass.txt").write_text(
            SYNTH_TMPL.format(query=q["query"], facts=fact_block,
                              passages="\n".join(passages) or "(none requested)"))
        manifest.append({"id": q["id"], "query": q["query"], "need": need,
                         "passages": len(passages), "dissent_lines": len(dissent)})
        print(f"  {q['id']:<24} need={need} passages={len(passages)} "
              f"dissent={len(dissent)}", file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} two-pass prompts")


DISSENT_CHECK_TMPL = """The evidence for this question contained genuine source disagreement:
{dissent}

Does the ANSWER below explicitly surface any disagreement or non-consensus view (rather than presenting a single unified explanation)?

ANSWER:
{answer}

Reply with exactly one line: DISSENT_FLAGGED: yes or DISSENT_FLAGGED: no"""


def cmd_grade(args):
    qs = _questions()
    sc = _grade_arm(qs, lambda q: OUT / f"ans_{q['id']}_twopass.txt", OUT / "grades_twopass")
    vals = [v for v in sc.values() if v is not None]
    print(f"twopass: n={len(vals)} mean={sum(vals)/len(vals):.2f} "
          f"dist={{0: {vals.count(0)}, 1: {vals.count(1)}, 2: {vals.count(2)}}}")
    # dissent-awareness: on questions with computed disagreement, did the
    # answer surface it? (checked for twopass AND v2 for comparison)
    manifest = {m["id"]: m for m in json.loads((OUT / "manifest.json").read_text())}
    for arm, path_fn in (("twopass", lambda q: OUT / f"ans_{q['id']}_twopass.txt"),
                         ("v2", lambda q: WW / f"ans_{q['id']}_v2.txt")):
        flagged = total = 0
        for q in qs:
            m = manifest.get(q["id"])
            if not m or not m.get("dissent_lines"):
                continue
            p = path_fn(q)
            if not p.exists():
                continue
            cf = OUT / f"dissent_{arm}_{q['id']}.txt"
            if not cf.exists():
                sf = OUT / f"survey_{q['id']}.txt"  # dissent lines live in gen prompt
                gen = (OUT / f"gen_{q['id']}_twopass.txt").read_text()
                dm = re.search(r"--- COMPUTED CONSENSUS/DISSENT ---\n(.*?)\n\n",
                               gen, re.S)
                dtext = dm.group(1) if dm else "(see context)"
                r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                                   input=DISSENT_CHECK_TMPL.format(
                                       dissent=dtext, answer=p.read_text().strip()),
                                   capture_output=True, text=True, timeout=120)
                cf.write_text(r.stdout or "")
            total += 1
            flagged += "DISSENT_FLAGGED: YES" in cf.read_text().upper()
        if total:
            print(f"dissent-awareness [{arm}]: {flagged}/{total} answers "
                  f"surfaced disagreement where it existed")
    # paired comparison vs v2 grades
    v2_dir = OUT / "grades_v2"
    if v2_dir.exists():
        better = worse = same = 0
        for q in qs:
            a = OUT / f"grades_twopass/{q['id']}.txt"
            b = v2_dir / f"{q['id']}.txt"
            if a.exists() and b.exists():
                d = int(a.read_text()) - int(b.read_text())
                better += d > 0
                worse += d < 0
                same += d == 0
        print(json.dumps({"paired_vs_v2": {"twopass_better": better,
                                           "v2_better": worse, "same": same}}))


CHUNKNOTE_TMPL = """You are a market analyst. Below are retrieved source passages. Structure your answer as an analyst note (3-8 sentences):
1. The consensus explanation, weighed from all the evidence.
2. Where sources DISAGREE (direction or attributed cause), say so explicitly with the sources.
3. Any non-consensus or tail signal from the evidence that might matter more than the consensus, and why.
Cite item numbers in brackets. News is aggregated collective reasoning — distinguishing what consensus believes from what might actually be relevant is the value of the note.

QUESTION: {query}

PASSAGES:
{items}
"""


def cmd_prep_chunks(args):
    """Chunk-RAG arm with the SAME analyst-note demands (fair fight):
    matched budget, exclusions respected, no computed annotations (chunks
    have no structured fields to compute them from)."""
    from qdrant_client.http import models
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    qs = _questions()
    for q in qs:
        excluded = set(q["exclude_doc_ids"])
        qv = _embed_all([q["query"]])[0]
        sims = fmat @ qv
        order = [int(i) for i in np.argsort(-sims)
                 if facts[int(i)]["doc_id"] not in excluded][:50]
        budget = len("\n".join(_fact_line(i + 1, facts[i])
                               for i, i2 in enumerate(order) for i in [i2]))
        pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                                  limit=60, with_payload=True).points
        lines, used = [], 0
        for p in pts:
            pl = p.payload or {}
            if pl.get("doc_id") in excluded:
                continue
            line = (f"[{len(lines) + 1}] {pl['text']} "
                    f"(source: {pl.get('source_name')}, {pl.get('published_date')})")
            if used + len(line) > budget and lines:
                break
            lines.append(line)
            used += len(line)
        (OUT / f"gen_{q['id']}_chunknote.txt").write_text(
            CHUNKNOTE_TMPL.format(query=q["query"], items="\n".join(lines)))
        print(f"  {q['id']:<24} chunks={len(lines)}", file=sys.stderr)
    print(f"prepared {len(qs)} chunk-note prompts")


def cmd_grade_chunks(args):
    qs = _questions()
    sc = _grade_arm(qs, lambda q: OUT / f"ans_{q['id']}_chunknote.txt",
                    OUT / "grades_chunknote")
    vals = [v for v in sc.values() if v is not None]
    print(f"chunknote: n={len(vals)} mean={sum(vals)/len(vals):.2f} "
          f"dist={{0: {vals.count(0)}, 1: {vals.count(1)}, 2: {vals.count(2)}}}")
    manifest = {m["id"]: m for m in json.loads((OUT / "manifest.json").read_text())}
    flagged = total = 0
    for q in qs:
        m = manifest.get(q["id"])
        if not m or not m.get("dissent_lines"):
            continue
        p = OUT / f"ans_{q['id']}_chunknote.txt"
        if not p.exists():
            continue
        cf = OUT / f"dissent_chunknote_{q['id']}.txt"
        if not cf.exists():
            gen = (OUT / f"gen_{q['id']}_twopass.txt").read_text()
            dm = re.search(r"--- COMPUTED CONSENSUS/DISSENT ---\n(.*?)\n\n", gen, re.S)
            dtext = dm.group(1) if dm else "(see context)"
            r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                               input=DISSENT_CHECK_TMPL.format(
                                   dissent=dtext, answer=p.read_text().strip()),
                               capture_output=True, text=True, timeout=120)
            cf.write_text(r.stdout or "")
        total += 1
        flagged += "DISSENT_FLAGGED: YES" in cf.read_text().upper()
    if total:
        print(f"dissent-awareness [chunknote]: {flagged}/{total}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["grade-existing", "prep", "grade",
                                     "prep-chunks", "grade-chunks"])
    args = p.parse_args()
    {"grade-existing": cmd_grade_existing, "prep": cmd_prep, "grade": cmd_grade,
     "prep-chunks": cmd_prep_chunks, "grade-chunks": cmd_grade_chunks}[args.stage](args)


if __name__ == "__main__":
    main()
