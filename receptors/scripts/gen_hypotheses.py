# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Generate upstream-hypothesis queries per eval query, per SOURCE arm (A/B/C/D).

This is the discovery engine's first stage. Each arm conditions on a different
context (hyp_sources.assemble_context) but is otherwise identical downstream, so a
head-to-head isolates the ONE variable the recent findings never controlled: where
the hypotheses come from.

Two composer modes:

  --compose llm       Assemble one prompt packet per (arm, query) for an external
                      LLM (codex exec / claude), matching the repo's offline-LLM
                      convention (cf. gen_hyde_input.py). Emits:
                        data/eval/hypgen_<arm>.jsonl   (prompts, one row per query)
                      You run the LLM over those, then `ingest` its JSON replies.

  --compose template  Deterministic, NO-LLM composer — turns the graph drivers /
                      cosine entities directly into hypothesis phrases. Reproducible
                      lower bound + lets the whole harness run (and smoke-test)
                      without a judge or an LLM in the loop. Emits the final
                        data/eval/hyps_<arm>.json  directly.

Both write hyps_<arm>.json = {qid: {effect, hypotheses:[str,...], meta:{...}}}.

Point-in-time: pass --asof (per-query 'asof' epoch in queries.json is used when
present) so hypothesis context only sees docs knowable as-of T.

Usage:
  uv run scripts/gen_hypotheses.py --compose template --arms A,B,C,D
  uv run scripts/gen_hypotheses.py --compose llm      --arms B,C,D
  uv run scripts/gen_hypotheses.py ingest --arm C --replies data/eval/hypgen_C_out.jsonl
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, "scripts")
import hyp_sources as hs

DEVAL = Path("data/eval")


def _asof_epoch(q):
    v = q.get("asof") or q.get("asof_epoch")
    return int(v) if v else None


# ---- template (no-LLM) composer -----------------------------------------------

def _template_hypotheses(arm, effect, ctx):
    """Deterministic hypothesis phrases from the arm context. Not as good as an LLM
    (that's the point — it's the floor), but grounded in the same substrate."""
    arm = arm.upper()
    hyps = [effect]  # every arm always includes the bare effect (baseline seed)
    seen = {effect.lower()}

    def add(p):
        if p and p.lower() not in seen:
            seen.add(p.lower())
            hyps.append(p)

    if arm in ("C", "D"):
        for d in ctx.get("graph_drivers", []):
            e = d["entity"]
            # two framings so cosine has a directional and an input-cost angle
            add(f"{e} impact on {effect}")
            add(f"{e} as an upstream driver of {effect}")
    if arm in ("B", "D"):
        # salient entities from the cosine neighbourhood (co-topical, weaker)
        ents = []
        for s in ctx.get("cosine_snippets", []):
            ents += s.get("entities", [])
        for e in list(dict.fromkeys(ents))[:8]:
            add(f"{e} and {effect}")
    return hyps


# ---- LLM prompt assembly ------------------------------------------------------

_PROMPT = """You are decomposing a market EFFECT into its plausible UPSTREAM CAUSAL \
DRIVERS, then writing a short search query for each so we can retrieve evidence.

Rules:
- Name drivers that are UPSTREAM of the effect (inputs, policy, FX, supply, demand \
shocks), not restatements of the effect itself.
- Prefer drivers whose vocabulary does NOT overlap the effect phrase — those are the \
ones plain topical search misses.
- 4-8 queries. Each query one line, concrete, retrieval-shaped (nouns, not questions).
- Output STRICT JSON: {"hypotheses": ["...", "..."]}  and nothing else.

EFFECT: %(effect)s
%(context)s"""


def _context_block(ctx):
    lines = []
    for s in ctx.get("cosine_snippets", []):
        lines.append(f"- (topical) {s['title'][:90]}")
    gd = ctx.get("graph_drivers", [])
    if gd:
        lines.append("Candidate upstream drivers from the causal graph "
                     "(entity, mass, example upstream docs):")
        for d in gd:
            ups = "; ".join(u["title"][:50] for u in d["upstream"]) or "-"
            lines.append(f"  * {d['entity']} (mass {d['mass']}) :: {ups}")
    if not lines:
        return "CONTEXT: (none — rely on world knowledge)"
    return "CONTEXT:\n" + "\n".join(lines)


def cmd_generate(a):
    queries = json.load(open(DEVAL / "queries.json"))
    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if a.compose == "template":
            out = {}
            for q in queries:
                ctx = hs.assemble_context(arm, q["effect"], before_epoch=_asof_epoch(q))
                out[q["id"]] = {"effect": q["effect"],
                                "hypotheses": _template_hypotheses(arm, q["effect"], ctx),
                                "meta": {"arm": arm, "compose": "template",
                                         "source": hs.ARMS[arm]}}
            DEVAL.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(DEVAL / f"hyps_{arm}.json", "w"), indent=1)
            n = sum(len(v["hypotheses"]) for v in out.values())
            print(f"[template] wrote hyps_{arm}.json  ({len(out)} q, {n} hypotheses)")
        else:
            rows = []
            for q in queries:
                ctx = hs.assemble_context(arm, q["effect"], before_epoch=_asof_epoch(q))
                prompt = _PROMPT % {"effect": q["effect"], "context": _context_block(ctx)}
                rows.append({"qid": q["id"], "effect": q["effect"], "prompt": prompt})
            DEVAL.mkdir(parents=True, exist_ok=True)
            p = DEVAL / f"hypgen_{arm}.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"[llm] wrote {p}  ({len(rows)} prompts) — run the LLM per row, "
                  f"then: gen_hypotheses.py ingest --arm {arm} --replies <replies.jsonl>")


def cmd_ingest(a):
    """Ingest LLM replies (jsonl rows: {qid, hypotheses:[...]} OR {qid, reply:'<json>'})
    into hyps_<arm>.json."""
    queries = {q["id"]: q["effect"] for q in json.load(open(DEVAL / "queries.json"))}
    out = {}
    for line in Path(a.replies).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["qid"]
        hyps = row.get("hypotheses")
        if hyps is None and "reply" in row:
            hyps = json.loads(row["reply"]).get("hypotheses", [])
        eff = queries.get(qid, "")
        hyps = [h for h in ([eff] + list(hyps or [])) if h]           # always seed the effect
        deduped = list(dict.fromkeys(hyps))
        out[qid] = {"effect": eff, "hypotheses": deduped,
                    "meta": {"arm": a.arm.upper(), "compose": "llm"}}
    json.dump(out, open(DEVAL / f"hyps_{a.arm.upper()}.json", "w"), indent=1)
    print(f"[ingest] wrote hyps_{a.arm.upper()}.json  ({len(out)} queries)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--compose", choices=["template", "llm"], default="template")
    g.add_argument("--arms", default="A,B,C,D")
    i = sub.add_parser("ingest")
    i.add_argument("--arm", required=True)
    i.add_argument("--replies", required=True)
    # default subcommand = generate
    if len(sys.argv) > 1 and sys.argv[1] not in ("generate", "ingest"):
        sys.argv.insert(1, "generate")
    a = ap.parse_args()
    if a.cmd == "ingest":
        cmd_ingest(a)
    else:
        cmd_generate(a)


if __name__ == "__main__":
    main()
