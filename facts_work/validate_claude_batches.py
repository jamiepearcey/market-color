#!/usr/bin/env python3
"""Validate claude-drained batch outputs: strip markdown fences, require
parseable JSONL with doc_ids from the input batch, rewrite cleaned files.
Invalid outputs are DELETED so run_claude_batches.sh retries them."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "batches"))

ok = bad = docs = facts = 0
for out in sorted(glob.glob("all_facts_*.jsonl")):
    i = out[len("all_facts_"):-len(".jsonl")]
    inp = f"all_batch_{i}.jsonl"
    if not os.path.exists(inp):
        continue
    want_ids = {json.loads(l)["doc_id"] for l in open(inp) if l.strip()}
    lines = []
    valid = True
    for line in open(out):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            valid = False
            break
        if o.get("doc_id") not in want_ids or not isinstance(o.get("facts"), list):
            valid = False
            break
        lines.append(o)
    got_ids = {o["doc_id"] for o in lines}
    if not valid or not lines or len(got_ids) < len(want_ids) * 0.8:
        print(f"INVALID {out} (valid={valid}, docs {len(got_ids)}/{len(want_ids)}) — deleted",
              file=sys.stderr)
        os.remove(out)
        bad += 1
        continue
    with open(out, "w") as fh:
        for o in lines:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    ok += 1
    docs += len(lines)
    facts += sum(len(o["facts"]) for o in lines)
print(json.dumps({"valid_outputs": ok, "invalid_deleted": bad,
                  "docs": docs, "facts": facts}))
