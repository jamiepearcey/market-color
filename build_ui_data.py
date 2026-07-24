# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15"]
# ///
"""Assemble the demo dashboard payload: hypotheses + channel-aware judge verdicts + the cited
evidence (qdrant payload + VERBATIM source text) for each activation. Output = one self-contained JSON
that the static HTML dashboard embeds."""
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

GRAPH = Path(__file__).resolve().parent / "data" / "eg_runs" / "eg100k_graph"
COLLECTION = "market_color_events"
HYP = Path("/tmp/hybrid_40_gated.json")
JUDGE = Path("/tmp/scored_40_fair.json")
OUT = Path("/tmp/ui_data.json")
ABSTAIN = ("unexplained", "insufficient", "unknown", "no clear", "no specific", "none identified", "unresolved")


def pid(cid): return str(uuid.uuid5(uuid.NAMESPACE_URL, str(cid)))


def is_abstain(h):
    return any(k in (h.get("primary_driver") or "").lower() for k in ABSTAIN)


hyp = json.load(open(HYP))
judged = {(r["t"], r["month"]): r.get("judge", {}) for r in json.load(open(JUDGE))}

cited = {c["chunk_id"] for r in hyp for c in (r.get("citations") or []) if c.get("chunk_id")}
client = QdrantClient(url="http://localhost:6333")
payloads = {p.id: p.payload for p in client.retrieve(COLLECTION, ids=[pid(c) for c in cited], with_payload=True)}
# verbatim text, one streaming pass
verbatim = {}
for line in open(GRAPH / "lake" / "chunk.jsonl"):
    j = json.loads(line)
    if j.get("chunk_id") in cited:
        verbatim[j["chunk_id"]] = (j.get("text") or "").strip()
        if len(verbatim) == len(cited):
            break

acts = []
for r in hyp:
    h = r["hypothesis"]
    ev = []
    for c in (r.get("citations") or []):
        cid = c.get("chunk_id")
        pl = payloads.get(pid(cid), {}) if cid else {}
        ev.append({
            "chunk_id": cid, "url": c.get("url"),
            "tickers": pl.get("tickers") or [], "is_macro": pl.get("is_macro"),
            "published_date": pl.get("published_date"), "headline": pl.get("headline"),
            "firm_specific": r["t"] in (pl.get("tickers") or []),
            "verbatim": (verbatim.get(cid) or "")[:520],
        })
    j = judged.get((r["t"], r["month"]), {})
    acts.append({
        "firm": r["firm"], "t": r["t"], "sec": r["sec"], "month": r["month"], "total": r["total"],
        "decomp": r["decomp"], "route": r["route"], "scope": r.get("scope"),
        "driver": h.get("primary_driver"), "confidence": h.get("confidence"),
        "reasoning": h.get("reasoning"), "gate_notes": h.get("gate_notes") or [],
        "sign_violation": bool(h.get("sign_violation")), "abstain": is_abstain(h),
        "verdict": j.get("verdict"), "channel_appropriate": j.get("channel_appropriate"),
        "evidence": ev,
    })

acts.sort(key=lambda a: -abs(a["total"]))
named = [a for a in acts if not a["abstain"]]
def rate(bucket):
    b = [a for a in acts if a["confidence"] == bucket]
    c = sum(1 for a in b if a["verdict"] == "correct")
    return {"n": len(b), "correct": c}
stats = {
    "n": len(acts),
    "named": len(named),
    "precision_correct": sum(1 for a in named if a["verdict"] == "correct"),
    "conf": {k: rate(k) for k in ("high", "med", "low")},
    "routes": {k: sum(1 for a in acts if a["route"] == k) for k in ("idio", "sector", "macro", "mixed")},
    "sign_caught": sum(1 for a in acts if a["sign_violation"]),
}
OUT.write_text(json.dumps({"stats": stats, "activations": acts}, indent=1))
print(f"wrote {OUT}: {len(acts)} activations, {len(cited)} cited chunks hydrated")
print("stats:", json.dumps(stats))
