#!/usr/bin/env python3
"""Post-drain measurements.

  cause-quality  compare cause specificity: old batches (000-053, codex, old
                 prompt) vs new batches (054+, haiku, specific-cause prompt).
  judge-prep     blind judge prompts: hybrid v2 answers regenerated on the
                 ENLARGED corpus (eval/blind_out_hybrid2/) vs the ORIGINAL
                 unrestricted chunk-RAG answers (eval/blind_out_alldocs/) —
                 the baseline that beat facts 5-3 purely on coverage.
  score          verdicts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chunk_baseline import JUDGE_TMPL  # noqa: E402

H2 = HERE / "blind_out_hybrid2"
FULLCHUNKS = HERE / "blind_out_alldocs"
D = HERE / "blind_h2_vs_fullchunks"

GENERIC = re.compile(r"^(the )?(conflict|war|crisis|situation|tensions?|uncertainty|"
                     r"market conditions|geopolitical (risk|tensions?))\.?$", re.I)


def cmd_cause_quality(args):
    def stats(paths):
        n = wc = short = generic = null = 0
        for fp in paths:
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                for f in (json.loads(line).get("facts") or []):
                    if not isinstance(f, dict):
                        continue
                    n += 1
                    c = f.get("cause")
                    if not c or str(c).lower() in ("none", "null"):
                        null += 1
                        continue
                    words = len(str(c).split())
                    wc += words
                    short += words <= 3
                    generic += bool(GENERIC.match(str(c).strip()))
        with_cause = n - null
        return {"facts": n, "with_cause_pct": round(100 * with_cause / n, 1),
                "avg_cause_words": round(wc / max(with_cause, 1), 1),
                "cause_<=3_words_pct": round(100 * short / max(with_cause, 1), 1),
                "generic_label_pct": round(100 * generic / max(with_cause, 1), 1)}

    b = str(ROOT / "facts_work" / "batches")
    old = [p for p in glob.glob(f"{b}/all_facts_*.jsonl")
           if int(re.search(r"(\d+)", Path(p).stem).group(1)) <= 53]
    new = [p for p in glob.glob(f"{b}/all_facts_*.jsonl")
           if int(re.search(r"(\d+)", Path(p).stem).group(1)) >= 54]
    print(json.dumps({"old_prompt_codex": stats(old),
                      "new_prompt_haiku": stats(new)}, indent=2))


def cmd_judge_prep(args):
    manifest = json.loads((H2 / "manifest.json").read_text())
    D.mkdir(exist_ok=True)
    n = 0
    for m in manifest:
        mine = H2 / f"ans_{m['id']}_hybrid2.txt"
        theirs = FULLCHUNKS / f"ans_{m['id']}_chunks.txt"
        if not mine.exists() or not theirs.exists():
            print(f"missing: {m['id']}", file=sys.stderr)
            continue
        mine_is_a = int(hashlib.sha256(f"{m['id']}|h2after|full".encode()).hexdigest(), 16) % 2 == 0
        a, b = ((mine.read_text().strip(), theirs.read_text().strip()) if mine_is_a
                else (theirs.read_text().strip(), mine.read_text().strip()))
        (D / f"judge_{m['id']}.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=a, b=b))
        m["h2_is_a_vs_fullchunks"] = mine_is_a
        n += 1
    (H2 / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {n} judge prompts (h2-after vs FULL-corpus chunks)")


def cmd_score(args):
    manifest = json.loads((H2 / "manifest.json").read_text())
    wins = losses = ties = 0
    for m in manifest:
        key = "h2_is_a_vs_fullchunks"
        vp = D / f"verdict_{m['id']}.txt"
        if key not in m or not vp.exists():
            continue
        match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
        if not match:
            continue
        v = match.group(1)
        won = (v == "A") == m[key] if v != "TIE" else None
        print(f"  {m['id']:<22} " + {True: "H2_WIN", False: "FULLCHUNKS_WIN",
                                      None: "TIE"}[won])
        if won is True:
            wins += 1
        elif won is False:
            losses += 1
        else:
            ties += 1
    print(json.dumps({"h2_wins": wins, "fullchunk_wins": losses, "ties": ties}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["cause-quality", "judge-prep", "score"])
    args = p.parse_args()
    {"cause-quality": cmd_cause_quality, "judge-prep": cmd_judge_prep,
     "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
