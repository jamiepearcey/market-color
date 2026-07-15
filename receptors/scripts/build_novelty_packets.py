# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build NOVELTY-AWARE content-review judge packets for the driver-discovery eval.

The flaw the recent findings diagnosed: blind pooled snippet-relevance (0/1/2 for
topical fit) is structurally blind to novelty — a redundant doc and a genuinely novel
driver both score 2. "Drivers not in the baseline" is a set-difference-then-verify
question that requires READING the candidate against what the baseline already knows.

So this packet shows the judge, per query:
  * the EFFECT,
  * the BASELINE driver set (cosine-on-effect docs) as read-only context, and
  * an anonymised, arm-blind pool of CANDIDATE docs (union of the arms) to grade.

Grades (see EVAL_DISCOVERY_PROTOCOL.md for the operational rubric):
  0  not a driver of the effect (off-topic, or an effect/restatement, not a cause)
  1  genuine driver, but ALREADY represented in the baseline set (redundant)
  2  genuine driver NOT represented in the baseline set (NOVEL & load-bearing)

Grade 2 is the discovery target: real, upstream, and net-new over the baseline.
Arm identity never reaches the judge (order = hash(qid|doc_id)); the key records
which arm ranked each doc where, for scoring.

Outputs:
  data/eval/packets_disc.json      {qid: {effect, baseline:[...], candidates:[...]}}
  data/eval/packet_disc_<qid>.json per-query
  data/eval/key_disc.json          {qid: {lid: {doc_id, ranks:{arm:rank}}}}

Usage:
  uv run scripts/build_novelty_packets.py --baseline baseline --arms A,B,C,D
"""
import json, hashlib, argparse
from pathlib import Path

D = Path("data/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline")   # cand_<baseline>.json (cosine-on-effect)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--suffix", default="disc")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    queries = json.load(open(D / "queries.json"))
    base = json.load(open(D / f"cand_{a.baseline}.json"))
    cand = {arm: json.load(open(D / f"cand_{arm}.json")) for arm in arms}

    packets, key = {}, {}
    for q in queries:
        qid = q["id"]
        base_ids = {r["doc_id"] for r in base.get(qid, [])}
        baseline_view = [{"title": r["title"], "snippet": r["snippet"]}
                         for r in base.get(qid, [])]

        pool = {}   # doc_id -> {title, snippet, ranks}
        for arm in arms:
            for r in cand[arm].get(qid, []):
                # baseline docs are shown as context, not graded — skip from the pool
                if r["doc_id"] in base_ids:
                    continue
                e = pool.setdefault(r["doc_id"], {"title": r["title"],
                                                  "snippet": r["snippet"], "ranks": {}})
                e["ranks"][arm] = r["rank"]

        items = sorted(pool.items(),
                       key=lambda kv: hashlib.md5(f"{qid}|{kv[0]}".encode()).hexdigest())
        cands, kmap = [], {}
        for i, (did, e) in enumerate(items):
            lid = f"D{i+1:02d}"
            cands.append({"lid": lid, "title": e["title"], "snippet": e["snippet"]})
            kmap[lid] = {"doc_id": did, "ranks": e["ranks"]}
        packets[qid] = {"effect": q["effect"], "baseline": baseline_view, "candidates": cands}
        key[qid] = kmap

    json.dump(packets, open(D / f"packets_{a.suffix}.json", "w"), indent=1)
    json.dump(key, open(D / f"key_{a.suffix}.json", "w"), indent=1)
    for qid, p in packets.items():
        json.dump(p, open(D / f"packet_{a.suffix}_{qid}.json", "w"), indent=1)
    tot = sum(len(p["candidates"]) for p in packets.values())
    print(f"wrote packets_{a.suffix}.json ({len(packets)} q, {tot} pooled candidates, "
          f"baseline={a.baseline}, arms={arms}) + per-query files")


if __name__ == "__main__":
    main()
