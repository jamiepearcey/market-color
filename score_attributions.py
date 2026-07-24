# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "httpx"]
# ///
"""SCORE hybrid attributions with an adversarial, source-grounded LLM judge.

Not human ground truth — but the judge is (a) skeptical by default, (b) shown the VERBATIM cited
source (not the hypothesis's own summary), and (c) cross-checked against mechanical firm/sign signals.
It answers the questions the eyeball spot-check can't quantify:
  - named-driver precision (correct / non-abstaining)
  - CONFIDENCE CALIBRATION: is "high" actually more accurate than "med"/"low"?
  - how many are sign-inconsistent (positive driver on a negative move) or generic-macro-on-idio.

  export GROQ_API_KEY=...
  uv run score_attributions.py --in /tmp/hybrid_40.json
"""
import argparse
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import httpx

GROQ = "https://api.groq.com/openai/v1/chat/completions"
KEY = os.environ.get("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
DEFAULT_GRAPH = Path(__file__).resolve().parent / "data" / "eg_runs" / "eg100k_graph"
COLLECTION = "market_color_events"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def hydrate(graph: Path, needed: set[str]) -> dict[str, str]:
    out = {}
    if not needed:
        return out
    for line in open(graph / "lake" / "chunk.jsonl"):
        j = json.loads(line)
        cid = j.get("chunk_id")
        if cid in needed:
            out[cid] = j.get("text") or ""
            if len(out) == len(needed):
                break
    return out


def judge(rec, sources_text, cited_tickers):
    firm, t = rec["firm"], rec["t"]
    tot = rec["total"]
    direction = "UP" if tot >= 0 else "DOWN"
    h = rec["hypothesis"]
    src = "\n\n".join(f"[{cid}] (tickers={cited_tickers.get(cid)}):\n{txt[:1400]}"
                      for cid, txt in sources_text.items()) or "(no source cited)"
    route = rec.get("route", "?")
    dc = rec.get("decomp", {})
    sysmsg = (
        "You are an ADVERSARIAL auditor of a stock-move attribution. Default to skeptical: only accept a "
        "claim the VERBATIM SOURCE actually supports, and reject any sign-inconsistent claim (a driver whose "
        "described effect would push the stock the OPPOSITE way to the observed move). "
        "Judge correctness RELATIVE TO THE ROUTED CHANNEL, which comes from a rigorous return decomposition: "
        "- macro-dominant move: the correct driver is a real MARKET/MACRO event (eurozone crisis, Fed/rates, "
        "commodity/risk-off) that is sign-consistent and would move a macro-sensitive stock; it need NOT mention "
        "the firm by name. "
        "- idiosyncratic move: require a FIRM-SPECIFIC driver that a cited source ties to this firm. "
        "- sector move: require a catalyst binding this firm's sector/peers. "
        "- mixed: either a firm-specific OR a macro driver is acceptable if sign-consistent. "
        "verdict = 'correct' (source supports a channel-appropriate, sign-consistent driver), 'weak' (related "
        "but thin/indirect), or 'wrong' (unsupported, sign-inconsistent, or wrong channel — e.g. a firm-specific "
        "claim with no firm evidence on an idiosyncratic move). "
        'Return ONLY JSON {"firm_specific":bool,"sign_consistent":bool,"channel_appropriate":bool,'
        '"driver_type":"driver|context","verdict":"correct|weak|wrong","why":str}')
    user = (f"FIRM: {firm} ({t})\nMOVE: {tot:+.1%} ({direction}) in {rec['month']}\n"
            f"ROUTED CHANNEL: {route}  (decomposition macro {dc.get('macro',0):+.1%} / "
            f"sector {dc.get('sector',0):+.1%} / idio {dc.get('idio',0):+.1%})\n"
            f"CLAIMED DRIVER: {h.get('primary_driver')}\nMODEL CONFIDENCE: {h.get('confidence')}\n"
            f"CITED VERBATIM SOURCE(S):\n{src}")
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]}
    for a in range(3):
        try:
            r = httpx.post(GROQ, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
            if r.status_code != 200:
                time.sleep(2 * (a + 1)); continue
            txt = r.json()["choices"][0]["message"]["content"]
            x, y = txt.find("{"), txt.rfind("}")
            return json.loads(txt[x:y + 1])
        except Exception:
            time.sleep(2 * (a + 1))
    return {"verdict": "error"}


ABSTAIN_MARK = ("unexplained", "insufficient", "unknown", "no clear", "no specific", "unresolved")


def is_abstain(rec):
    pd = (rec["hypothesis"].get("primary_driver") or "").lower()
    return any(k in pd for k in ABSTAIN_MARK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    client = QdrantClient(url=args.url)
    recs = json.load(open(args.inp))

    # hydrate all cited chunks + fetch their tickers from qdrant
    needed = {c["chunk_id"] for r in recs for c in (r.get("citations") or []) if c.get("chunk_id")}
    text = hydrate(args.graph, needed)
    tickers = {}
    if needed:
        pts = client.retrieve(COLLECTION, ids=[point_id(c) for c in needed], with_payload=True)
        by_pid = {p.id: p.payload for p in pts}
        for c in needed:
            tickers[c] = (by_pid.get(point_id(c)) or {}).get("tickers")

    scored = []
    for r in recs:
        cids = [c["chunk_id"] for c in (r.get("citations") or []) if c.get("chunk_id")]
        v = judge(r, {c: text.get(c, "") for c in cids}, tickers) if KEY else {"verdict": "no-key"}
        # mechanical cross-check: is the firm actually in any cited chunk's tickers?
        firm_in_src = any(r["t"] in (tickers.get(c) or []) for c in cids)
        scored.append({**r, "judge": v, "firm_in_src": firm_in_src, "abstain": is_abstain(r)})
        print(f"  {r['t']:6} {r['month']} {r['total']:+.0%} [{r['route']:6}] "
              f"conf={r['hypothesis'].get('confidence'):4} -> {v.get('verdict'):7} "
              f"(firm_specific={v.get('firm_specific')}, sign={v.get('sign_consistent')}, "
              f"firm_in_src={firm_in_src})")

    # ---- aggregate ----
    n = len(scored)
    named = [s for s in scored if not s["abstain"]]
    vd = Counter(s["judge"].get("verdict") for s in scored)
    print(f"\n==== N={n}  ({len(named)} named, {n - len(named)} abstained) ====")
    print("verdict:", dict(vd))
    correct = vd.get("correct", 0)
    print(f"named-driver precision (correct/named): {correct}/{len(named)} = "
          f"{correct / max(len(named), 1):.0%}")
    print(f"firm_specific: {sum(1 for s in scored if s['judge'].get('firm_specific'))}/{n}  |  "
          f"sign_consistent: {sum(1 for s in scored if s['judge'].get('sign_consistent'))}/{n}  |  "
          f"firm_in_cited_source: {sum(1 for s in scored if s['firm_in_src'])}/{n}")

    print("\n-- CONFIDENCE CALIBRATION (judge 'correct' rate within each stated confidence) --")
    by_conf = defaultdict(list)
    for s in scored:
        by_conf[s["hypothesis"].get("confidence")].append(s["judge"].get("verdict") == "correct")
    for c in ("high", "med", "low"):
        b = by_conf.get(c, [])
        if b:
            print(f"  {c:4}: {sum(b)}/{len(b)} correct = {sum(b) / len(b):.0%}")

    print("\n-- FLAGGED (wrong or sign-inconsistent) --")
    for s in scored:
        v = s["judge"]
        if v.get("verdict") == "wrong" or v.get("sign_consistent") is False:
            print(f"  {s['t']:6} {s['total']:+.0%} conf={s['hypothesis'].get('confidence')} "
                  f":: {(s['hypothesis'].get('primary_driver') or '')[:55]}  <- {(v.get('why') or '')[:70]}")

    if args.out:
        args.out.write_text(json.dumps(scored, indent=1))
        print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
