# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "fastembed>=0.3", "numpy"]
# ///
"""HYBRID retrieval router: the decomposition PLANS the Qdrant search.

This is the piece that sits between the residual decomposition (macro/sector/idio) and
the entity-enriched `market_color_events` collection. Instead of one channel-blind
semantic search, it SCOPES the query by what carried the move:

  idiosyncratic-dominant -> tickers = {firm}, point-in-time window, driver-type query
  sector-dominant        -> tickers in the beta-correlated PEER SUBSET (not the whole
                            sector), window, is_macro=false
  macro-dominant         -> is_macro=true, window, monetary/rate query
  mixed                  -> union of the relevant scoped searches

Qdrant then maximises recall WITHIN the scope (and can surface the right article the
causal-edge extractor missed). Every hit back-links via chunk_id -> source chunk.

Demo (needs the populated collection from index_events.py):
  uv run retrieval_router.py --demo 6
Single firm:
  uv run retrieval_router.py --firm BP --month 2010-06
"""
import argparse
import calendar
import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE / "data" / "eg_runs" / "eg100k_graph"
DEFAULT_URL = "http://localhost:6333"
COLLECTION = "market_color_events"
FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Channel-conditioned query priors (what KIND of news to surface).
IDIO_TERMS = "earnings results guidance profit warning restructuring merger acquisition CEO"
MACRO_TERMS = "central bank interest rate monetary policy credit crisis sovereign debt recession"


def month_window(month: str) -> tuple[int, int]:
    """'2011-09' -> (ordinal of first day, ordinal of last day)."""
    y, m = int(month[:4]), int(month[5:7])
    lo = dt.date(y, m, 1).toordinal()
    hi = dt.date(y, m, calendar.monthrange(y, m)[1]).toordinal()
    return lo, hi


def dominant_channel(mv: dict) -> tuple[str, dict]:
    """Return (channel, shares). 'mixed' when the runner-up carries >1/3 of the move."""
    shares = {"macro": abs(mv["macro"]), "sector": abs(mv["sector"]), "idio": abs(mv["idio"])}
    tot = sum(shares.values()) or 1e-9
    ordered = sorted(shares.items(), key=lambda x: -x[1])
    channel = "mixed" if ordered[1][1] / tot > 0.33 else ordered[0][0]
    return channel, shares


def peer_subset(firm_ticker: str, sector: str, firms_by_sector: dict, precomputed: dict | None) -> list[str]:
    """Beta-correlated peers if supplied upstream, else fall back to the same-sector set."""
    if precomputed and firm_ticker in precomputed:
        return precomputed[firm_ticker]
    return [t for t in firms_by_sector.get(sector, []) if t != firm_ticker]


def build_plan(firm_ticker: str, firm_name: str, sector: str, mv: dict,
               firms_by_sector: dict, precomputed_peers: dict | None):
    from qdrant_client.http import models

    channel, shares = dominant_channel(mv)
    lo, hi = month_window(mv["m"])
    window = models.FieldCondition(key="published_ordinal", range=models.Range(gte=lo, lte=hi))

    if channel == "idio":
        query = f"{firm_name} {IDIO_TERMS}"
        must = [window, models.FieldCondition(key="tickers", match=models.MatchValue(value=firm_ticker))]
        scope = f"tickers={firm_ticker}"
    elif channel == "sector":
        peers = peer_subset(firm_ticker, sector, firms_by_sector, precomputed_peers)
        query = f"{sector} sector {firm_name} peers"
        must = [window,
                models.FieldCondition(key="tickers", match=models.MatchAny(any=peers or [firm_ticker])),
                models.FieldCondition(key="is_macro", match=models.MatchValue(value=False))]
        scope = f"peers({len(peers)})={peers[:6]}{'…' if len(peers) > 6 else ''}"
    elif channel == "macro":
        query = MACRO_TERMS
        must = [window, models.FieldCondition(key="is_macro", match=models.MatchValue(value=True))]
        scope = "is_macro=true"
    else:  # mixed -> firm's own OR macro, within the window
        query = f"{firm_name} {IDIO_TERMS} OR {MACRO_TERMS}"
        must = [window]
        should = [models.FieldCondition(key="tickers", match=models.MatchValue(value=firm_ticker)),
                  models.FieldCondition(key="is_macro", match=models.MatchValue(value=True))]
        return channel, shares, query, models.Filter(must=must, should=should), "firm-own OR macro"

    return channel, shares, query, models.Filter(must=must), scope


def _norm(v):
    import numpy as np
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    return a / n if n else a


def macro_centroid(client, month: str, min_n: int = 15, cap: int = 400):
    """Mean (normalized) embedding of is_macro chunks that month = the 'generic macro tape' direction."""
    import numpy as np
    from qdrant_client.http import models
    pts, _ = client.scroll(
        COLLECTION,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="is_macro", match=models.MatchValue(value=True)),
            models.FieldCondition(key="month", match=models.MatchValue(value=month))]),
        limit=cap, with_vectors=True, with_payload=False)
    vecs = [p.vector for p in pts if p.vector]
    if len(vecs) < min_n:
        return None
    return _norm(np.mean([_norm(v) for v in vecs], axis=0))


def search(client, embedder, query: str, qfilter, limit: int,
           centroid=None, lam: float = 0.35, overfetch: int = 5):
    """Retrieve; if a macro centroid is given, over-fetch and RE-RANK by (score - lam * cos-to-macro),
    demoting generic-macro chunks so the firm-specific residual surfaces. No-op when centroid is None."""
    import numpy as np
    vec = list(map(float, next(iter(embedder.embed([query])))))
    k = limit * overfetch if centroid is not None else limit
    pts = client.query_points(COLLECTION, query=vec, query_filter=qfilter, limit=k,
                              with_payload=True, with_vectors=centroid is not None).points
    if centroid is None:
        return pts[:limit]
    c = _norm(centroid)
    scored = []
    for p in pts:
        gen = float(np.dot(_norm(p.vector), c)) if p.vector is not None else 0.0
        scored.append((p.score - lam * gen, gen, p))
    scored.sort(key=lambda x: -x[0])
    out = []
    for adj, gen, p in scored[:limit]:
        p.payload["_generic"] = round(gen, 3)
        out.append(p)
    return out


def load_attribution(graph: Path):
    from collections import defaultdict
    d = json.load(open(graph / "attribution.json"))
    firms_by_sector = defaultdict(list)
    firms = {}
    for f in d["firms"]:
        firms_by_sector[f["sec"]].append(f["t"])
        firms[f["t"]] = f
    return firms, dict(firms_by_sector)


def run_one(client, embedder, f: dict, mv: dict, firms_by_sector, peers, limit: int):
    channel, shares, query, qfilter, scope = build_plan(
        f["t"], f["n"], f["sec"], mv, firms_by_sector, peers)
    hits = search(client, embedder, query, qfilter, limit)
    print(f"\n=== {f['n']} ({f['t']}) {mv['m']}  {mv['tot']:+.1%}  "
          f"[macro {mv['macro']:+.1%} | sector {mv['sector']:+.1%} | idio {mv['idio']:+.1%}] ===")
    print(f"  ROUTE: {channel}-dominant  scope: {scope}")
    print(f"  query: {query!r}")
    if not hits:
        print("  (no scoped evidence surfaced in this window)")
    for h in hits:
        pl = h.payload
        print(f"  · {h.score:.3f} [{pl.get('published_date')}] tickers={pl.get('tickers')} "
              f"macro={pl.get('is_macro')} mech={pl.get('mechanisms')}")
        print(f"      quote: {(pl.get('quote') or '')[:120]!r}")
        print(f"      back-link: chunk_id={pl.get('chunk_id')}  doc_id={pl.get('doc_id')}  {pl.get('url')}")
    return {"firm": f["t"], "month": mv["m"], "channel": channel,
            "hits": [{"chunk_id": h.payload.get("chunk_id"), "score": h.score,
                      "tickers": h.payload.get("tickers")} for h in hits]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--peers", type=Path, default=None, help="JSON {ticker: [peer,...]} beta-correlated subsets")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--demo", type=int, default=0, help="run biggest move of first N firms")
    ap.add_argument("--firm", default=None)
    ap.add_argument("--month", default=None)
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from fastembed import TextEmbedding

    firms, firms_by_sector = load_attribution(args.graph)
    peers = json.load(open(args.peers)) if args.peers else None
    client = QdrantClient(url=args.url, api_key=args.api_key)
    embedder = TextEmbedding(model_name=FASTEMBED_MODEL)

    out = []
    if args.firm:
        f = firms[args.firm]
        mv = next(m for m in f["moves"] if m["m"] == args.month) if args.month \
            else max(f["moves"], key=lambda x: abs(x["tot"]))
        out.append(run_one(client, embedder, f, mv, firms_by_sector, peers, args.limit))
    else:
        n = args.demo or 6
        for f in list(firms.values())[:n]:
            mv = max(f["moves"], key=lambda x: abs(x["tot"]))
            out.append(run_one(client, embedder, f, mv, firms_by_sector, peers, args.limit))
    print(f"\n-> routed {len(out)} activations")


if __name__ == "__main__":
    main()
