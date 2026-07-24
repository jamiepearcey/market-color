# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "fastembed>=0.3", "httpx", "numpy"]
# ///
"""HYBRID attribution — the full loop.

  decompose (macro/sector/idio)  ->  ROUTE + retrieve scoped Qdrant evidence (recall)
    ->  LLM reasoning over the RETRIEVED chunks (not the graph-only event list)
    ->  structured hypothesis WITH CITATIONS (chunk_id + url).

vs `hypothesis_engine.py`: same decomposition + confidence discipline, but the evidence now
comes from the entity-scoped `market_color_events` collection, so the reasoner sees the article
the causal-edge extractor missed (BP dividend cut, ACGBY Huijin buy) and can cite it.

  export GROQ_API_KEY=...   # from data/tmp/groq.env; without it, deterministic fallback
  uv run hybrid_attribute.py --demo 8 --peers data/eg_runs/eg100k_graph/peer_subsets.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import httpx

from retrieval_router import build_plan, load_attribution, search, macro_centroid, DEFAULT_GRAPH, DEFAULT_URL

GROQ = "https://api.groq.com/openai/v1/chat/completions"
KEY = os.environ.get("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_NOISE = 0.0107  # idio noise baseline (kept in sync with hypothesis_engine deterministic rule)


def hydrate_chunks(graph: Path, needed: set[str]) -> dict[str, str]:
    """Resolve chunk_id -> VERBATIM original chunk text via the back-link (one streaming pass)."""
    out: dict[str, str] = {}
    if not needed:
        return out
    for line in open(graph / "lake" / "chunk.jsonl"):
        j = json.loads(line)
        cid = j.get("chunk_id")
        if cid in needed:
            out[cid] = (j.get("text") or "")
            if len(out) == len(needed):
                break
    return out


def evidence_block(hits, fulltext: dict[str, str], topk: int, max_chars: int) -> str:
    """High-importance hits (top-k by score) carry VERBATIM source text; the rest stay light."""
    lines = []
    for i, h in enumerate(hits):
        pl = h.payload
        cid = pl.get("chunk_id")
        head = (f"  - [{h.score:.2f}|{','.join(pl.get('mechanisms') or []) or '?'}] "
                f"{pl.get('published_date')} tickers={pl.get('tickers')} macro={pl.get('is_macro')}")
        if i < topk and cid in fulltext:
            lines.append(f"{head}\n    SOURCE (verbatim): \"{fulltext[cid][:max_chars].strip()}\"  (cite {cid})")
        else:
            lines.append(f"{head}: {(pl.get('headline') or '').strip()[:120]} — "
                         f"\"{(pl.get('quote') or '')[:100]}\"  (cite {cid})")
    return "\n".join(lines) or "  (no scoped evidence surfaced)"


def reason_llm(firm, sector, mv, channel, shares, evid):
    sysmsg = (
        "You attribute a stock move to its driver. You get a rigorous RESIDUAL DECOMPOSITION "
        "(which channel carried the move) and channel-ROUTED, entity-scoped news evidence retrieved "
        "for that firm/peers/macro in the move's month. High-importance items include the VERBATIM "
        "SOURCE text — base your driver on that source text, not on the tags or a summary. Rules: weigh "
        "channels BY THE DECOMPOSITION shares; prefer driver-type news (earnings/guidance/restructuring/"
        "M&A/dividend/rating) over context (generic factor/market mentions); the retrieval score is "
        "semantic relevance, not proof. Do NOT invent causes not in the evidence; if the evidence is thin "
        "or generic, say so and abstain rather than name from noise. Cite the chunk_id(s) you rely on. "
        'Return ONLY JSON {"channel":"macro|sector|idiosyncratic|mixed","primary_driver":str,'
        '"reasoning":str(<=60 words),"confidence":"high|med|low","citations":[chunk_id,...]}')
    user = (
        f"{firm} ({sector}) moved {mv['tot']:+.1%} in {mv['m']}.\n"
        f"DECOMPOSITION: macro {mv['macro']:+.1%}, sector {mv['sector']:+.1%}, idiosyncratic {mv['idio']:+.1%} "
        f"(routed channel: {channel}).\nSCOPED RETRIEVED EVIDENCE:\n{evid}")
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1500,  # reasoning model: leave room for reasoning + JSON
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]}
    for a in range(2):
        try:
            r = httpx.post(GROQ, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
            if r.status_code != 200:
                time.sleep(2 * (a + 1)); continue
            t = r.json()["choices"][0]["message"]["content"]
            x, y = t.find("{"), t.rfind("}")
            h = json.loads(t[x:y + 1]); h["method"] = "llm"
            return h
        except Exception:
            time.sleep(2 * (a + 1))
    return {}


def reason_deterministic(channel, mv, hits):
    """Fallback: top scored scoped hit is the driver if relevant enough, else abstain."""
    top = hits[0] if hits else None
    chan = {"idio": "idiosyncratic"}.get(channel, channel)
    if not top or top.score < 0.40:
        return {"channel": chan, "primary_driver": "unexplained by scoped news",
                "reasoning": f"{channel} channel routed but no retrieved evidence clears the relevance bar — abstaining",
                "confidence": "low", "citations": [], "method": "deterministic"}
    pl = top.payload
    driver = (pl.get("mechanisms") or ["move"])[0]
    conf = "high" if top.score >= 0.55 else "med" if top.score >= 0.45 else "low"
    if channel == "mixed" and conf == "high":
        conf = "med"
    return {"channel": chan, "primary_driver": f"{driver}: {(pl.get('quote') or '')[:60]}",
            "reasoning": f"{channel} channel; top scoped hit {top.score:.2f} ({pl.get('published_date')}, "
                         f"tickers={pl.get('tickers')})",
            "confidence": conf, "citations": [pl.get("chunk_id")], "method": "deterministic"}


_CONF_RANK = {"low": 0, "med": 1, "high": 2}
_RANK_CONF = {0: "low", 1: "med", 2: "high"}
_DIRV = {"up": 1, "down": -1, "widen": -1, "tighten": 1}


def _cap(conf, ceiling):
    return _RANK_CONF[min(_CONF_RANK.get(conf, 0), _CONF_RANK[ceiling])]


def apply_gate(h, firm_t, channel, mv, hits):
    """Deterministic grounding gate: a confident FIRM-level driver must be backed by a cited source that
    names the firm (idio) or >=2 peers (sector); macro can't masquerade as a firm driver; and the cited
    evidence's direction must be consistent with the move sign. Ungrounded claims are downgraded/abstained."""
    cited_ids = set(h.get("citations") or [])
    cited = [hit.payload for hit in hits if hit.payload.get("chunk_id") in cited_ids]
    move_sign = 1 if mv["tot"] >= 0 else -1
    firm_named = any(firm_t in (pl.get("tickers") or []) for pl in cited)
    peer_named = len({t for pl in cited for t in (pl.get("tickers") or []) if t != firm_t})
    notes = []

    if channel == "idio":
        if not firm_named:
            h["primary_driver"] = "unexplained by firm-specific sources"
            h["confidence"] = "low"; notes.append("idio:no-firm-source->abstain")
    elif channel == "sector":
        if peer_named < 2:
            h["confidence"] = _cap(h.get("confidence"), "low"); notes.append("sector:<2-peer-sources")
    elif channel == "macro":
        h["confidence"] = _cap(h.get("confidence"), "med"); notes.append("macro:cap-med")
        if not firm_named:
            notes.append("macro:generic-not-firm-driver")
    else:  # mixed
        if not firm_named and peer_named < 2:
            h["confidence"] = _cap(h.get("confidence"), "low"); notes.append("mixed:ungrounded->cap-low")

    # sign consistency: if every directional cue in the cited evidence contradicts the move, it can't be the driver
    signs = [_DIRV[d] for pl in cited for d in (pl.get("effect_dirs") or []) if d in _DIRV]
    if signs and all(s != move_sign for s in signs):
        h["confidence"] = _cap(h.get("confidence"), "low")
        h["sign_violation"] = True; notes.append("sign-violation->cap-low")

    h["gate_notes"] = notes
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--peers", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--demo", type=int, default=8)
    ap.add_argument("--fulltext-topk", type=int, default=3,
                    help="feed VERBATIM source text for this many top hits per activation")
    ap.add_argument("--fulltext-chars", type=int, default=1600)
    ap.add_argument("--lam", type=float, default=0.0,
                    help="macro-centroid penalty weight (0 = off; measured neutral at 0.35, kept optional)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from fastembed import TextEmbedding

    firms, firms_by_sector = load_attribution(args.graph)
    peers = json.load(open(args.peers)) if args.peers else None
    client = QdrantClient(url=args.url)
    embedder = TextEmbedding(model_name=FASTEMBED_MODEL)
    print(f"[hybrid] reasoning mode: {'LLM (' + MODEL + ')' if KEY else 'deterministic (no GROQ_API_KEY)'}")

    # Phase 1 — retrieve scoped evidence for every activation (macro-centroid re-rank on non-macro routes).
    plans = []
    needed: set[str] = set()
    centroid_cache: dict[str, object] = {}
    for f in list(firms.values())[:args.demo]:
        mv = max(f["moves"], key=lambda x: abs(x["tot"]))
        channel, shares, query, qfilter, scope = build_plan(
            f["t"], f["n"], f["sec"], mv, firms_by_sector, peers)
        cen = None
        if args.lam > 0 and channel != "macro":  # don't penalize macro news on a macro-routed move
            if mv["m"] not in centroid_cache:
                centroid_cache[mv["m"]] = macro_centroid(client, mv["m"])
            cen = centroid_cache[mv["m"]]
        hits = search(client, embedder, query, qfilter, args.limit, centroid=cen, lam=args.lam)
        plans.append((f, mv, channel, shares, scope, hits))
        for h in hits[:args.fulltext_topk]:
            if h.payload.get("chunk_id"):
                needed.add(h.payload["chunk_id"])

    # Phase 2 — hydrate ONLY the high-importance chunks to verbatim source (one streaming pass).
    print(f"[hybrid] hydrating {len(needed)} high-importance chunks to verbatim source ...")
    fulltext = hydrate_chunks(args.graph, needed)

    # Phase 3 — synthesise over the verbatim source (not summaries).
    out = []
    for f, mv, channel, shares, scope, hits in plans:
        evid = evidence_block(hits, fulltext, args.fulltext_topk, args.fulltext_chars)
        h = (reason_llm(f["n"], f["sec"], mv, channel, shares, evid) if KEY else {}) \
            or reason_deterministic(channel, mv, hits)
        h = apply_gate(h, f["t"], channel, mv, hits)
        # resolve citation chunk_ids -> urls for display
        by_cid = {p.payload.get("chunk_id"): p.payload for p in hits}
        cites = [{"chunk_id": c, "url": (by_cid.get(c) or {}).get("url")} for c in (h.get("citations") or [])]
        rec = {"firm": f["n"], "t": f["t"], "sec": f["sec"], "month": mv["m"], "total": mv["tot"],
               "decomp": {"macro": mv["macro"], "sector": mv["sector"], "idio": mv["idio"]},
               "route": channel, "scope": scope, "hypothesis": h, "citations": cites}
        out.append(rec)
        print(f"\n=== {f['n']} ({f['t']}) {mv['m']}  {mv['tot']:+.1%}  "
              f"[macro {mv['macro']:+.1%} | sector {mv['sector']:+.1%} | idio {mv['idio']:+.1%}] ===")
        print(f"  route: {channel} ({scope})  | method: {h.get('method')}")
        print(f"  channel: {h.get('channel')}  driver: {h.get('primary_driver')}  conf: {h.get('confidence')}")
        print(f"  reasoning: {h.get('reasoning')}")
        for c in cites:
            print(f"  cite: {c['chunk_id']}  {c['url']}")

    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
        print(f"\n-> wrote {args.out} ({len(out)} activations)")


if __name__ == "__main__":
    main()
