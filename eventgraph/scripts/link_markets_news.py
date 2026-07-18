# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""
Link prediction markets <-> news, reusing the structure we ALREADY extract.

A market (polymarket_ingest / kalshi_ingest proposition) and a news item are relevant
to each other on two channels:

  STRUCTURAL (deterministic, high-precision): they share a resolved SYMBOL. The news
    graph resolves effect entities -> tickers (entity_symbol.jsonl via causal edges);
    markets resolve to the same ticker vocabulary. Their intersection is an exact link
    ("this Fed market <-> every news edge whose effect is ^TNX").

  SEMANTIC (recall, topical): cosine similarity between the market question and the
    news text (headline / causal-edge quote) in MiniLM space -- catches relevance the
    coarse symbol proxy misses ("Will the Fed cut 50bps?" <-> "traders price a jumbo
    cut after soft CPI" even when both only map to ^TNX, or when neither resolved).

Combined score = semantic cosine, boosted when a symbol is shared. Emits
lake/market_news_link.jsonl {prop_id, unit, ref, score, via, symbols, market_q, news}.
News unit = document headline by default; --edges also links against causal-edge quotes
(finer, entity-anchored) when a causal_event_edge.jsonl is present.

Views:
  build              compute + write all links.
  market  <prop|slug|q>   a market and its most-relevant news.
  news    <doc|substr>    a news item and the markets it's relevant to.

Usage:
  uv run eventgraph/scripts/link_markets_news.py build  --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/link_markets_news.py build  --graph-dir eventgraph/data/eg_live --edges
  uv run eventgraph/scripts/link_markets_news.py market "Fed decrease" --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/link_markets_news.py news   "Netflix"      --graph-dir eventgraph/data/eg_live
"""
import argparse, json, collections
from pathlib import Path
import numpy as np


def load_markets(lake):
    p = lake / "proposition.jsonl"
    if not p.exists():
        raise SystemExit(f"no {p} -- run polymarket_ingest.py / kalshi_ingest.py first")
    out = []
    for l in open(p):
        if not l.strip():
            continue
        m = json.loads(l)
        txt = " ".join(x for x in [m.get("event_title"), m.get("question"),
                                   (m.get("resolution_criteria") or "")[:160]] if x)
        out.append({"prop_id": m["prop_id"], "source": m["source"], "q": m.get("question") or m.get("event_title") or "",
                    "text": txt, "symbols": set(m.get("symbols") or []),
                    "date": m.get("resolution_date"), "category": m.get("category")})
    return out


def load_news(lake, use_edges):
    """news units: documents (headline) + optionally causal edges (quote, entity-anchored)."""
    docs = {}
    dp = lake / "document.jsonl"
    if dp.exists():
        for l in open(dp):
            if l.strip():
                d = json.loads(l); docs[d["doc_id"]] = d
    # entity -> symbol (for the structural channel)
    esym = {}
    ep = lake.parent / "entity_symbol.jsonl"
    if ep.exists():
        for l in open(ep):
            j = json.loads(l)
            if j.get("symbol"):
                esym[j["entity_id"]] = j["symbol"]
    # doc -> symbols (via causal edges' effect entities)
    doc_syms = collections.defaultdict(set)
    edges = []
    cp = lake / "causal_event_edge.jsonl"
    if cp.exists():
        for l in open(cp):
            if not l.strip():
                continue
            e = json.loads(l); did = e.get("doc_id"); eff = e.get("effect_entity")
            if eff in esym:
                doc_syms[did].add(esym[eff])
            if use_edges and (e.get("quote") or "").strip():
                edges.append({"unit": "edge", "ref": f'{did}:{eff}', "doc_id": did,
                              "text": e["quote"], "symbols": {esym[eff]} if eff in esym else set(),
                              "headline": (docs.get(did) or {}).get("headline", ""),
                              "date": (docs.get(did) or {}).get("published_at", "")[:10]})
    units = []
    for did, d in docs.items():
        h = d.get("headline") or ""
        if h:
            units.append({"unit": "doc", "ref": did, "doc_id": did, "text": h,
                          "symbols": doc_syms.get(did, set()), "headline": h,
                          "date": (d.get("published_at") or "")[:10]})
    units += edges
    return units


def embed(texts):
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    V = np.array(list(model.embed(texts)), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V


def compute_links(markets, news, topk, min_cos, sym_boost):
    if not markets or not news:
        return []
    Mtxt = [m["text"] for m in markets]
    Ntxt = [n["text"] for n in news]
    print(f"embedding {len(Mtxt)} markets + {len(Ntxt)} news units (MiniLM) ...")
    allv = embed(Mtxt + Ntxt)
    MV, NV = allv[:len(Mtxt)], allv[len(Mtxt):]
    links = []
    for i, m in enumerate(markets):
        cos = NV @ MV[i]                                   # (Nnews,)
        boosted = cos.copy()
        if m["symbols"]:
            for j, n in enumerate(news):
                if n["symbols"] & m["symbols"]:
                    boosted[j] += sym_boost                # structural agreement lifts the pair
        order = np.argsort(-boosted)[: topk * 3]
        picked = 0
        for j in order:
            shared = bool(news[j]["symbols"] & m["symbols"])
            if boosted[j] < min_cos and not shared:
                break
            via = ("both" if shared and cos[j] >= min_cos else
                   "symbol" if shared else "semantic")
            links.append({"prop_id": m["prop_id"], "source": m["source"], "unit": news[j]["unit"],
                          "ref": news[j]["ref"], "doc_id": news[j]["doc_id"],
                          "score": round(float(boosted[j]), 3), "cos": round(float(cos[j]), 3),
                          "via": via, "symbols": sorted(m["symbols"] & news[j]["symbols"]),
                          "market_q": m["q"][:120], "news": news[j]["headline"][:120],
                          "market_date": m["date"], "news_date": news[j]["date"]})
            picked += 1
            if picked >= topk:
                break
    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "market", "news"])
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--edges", action="store_true", help="also link against causal-edge quotes")
    ap.add_argument("--topk", type=int, default=8, help="max news links kept per market")
    ap.add_argument("--min-cos", type=float, default=0.35, help="semantic cosine floor")
    ap.add_argument("--sym-boost", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"
    lp = lake / "market_news_link.jsonl"

    if a.cmd == "build":
        markets = load_markets(lake); news = load_news(lake, a.edges)
        print(f"{len(markets)} markets, {len(news)} news units")
        links = compute_links(markets, news, a.topk, a.min_cos, a.sym_boost)
        with open(lp, "w") as f:
            for x in links:
                f.write(json.dumps(x) + "\n")
        via = collections.Counter(x["via"] for x in links)
        mk = len({x["prop_id"] for x in links})
        print(f"wrote {len(links)} links ({mk} markets linked) -> {lp}")
        print("by channel:", dict(via))
        return

    if not lp.exists():
        raise SystemExit("no links yet -- run `build` first")
    links = [json.loads(l) for l in open(lp) if l.strip()]

    if a.cmd == "market":
        markets = load_markets(lake)
        ql = a.query.lower()
        hits = [m for m in markets if ql in m["prop_id"].lower() or ql in m["q"].lower()]
        if not hits:
            print(f"no market matching '{a.query}'"); return
        for m in hits[:5]:
            ml = sorted([x for x in links if x["prop_id"] == m["prop_id"]], key=lambda x: -x["score"])
            print(f"\n=== [{m['source']}/{m['category']}] {m['q'][:90]}  (res {m['date']}, {sorted(m['symbols']) or '-'})")
            if not ml:
                print("    (no linked news)"); continue
            for x in ml[:a.top]:
                mark = "★" if x["via"] == "both" else ("=" if x["via"] == "symbol" else "~")
                print(f"    {mark} {x['score']:.2f} [{x['via']}{'/'+','.join(x['symbols']) if x['symbols'] else ''}] "
                      f"{x['news_date']}  {x['news'][:80]}")
        return

    # news
    ql = a.query.lower()
    hits = [x for x in links if ql in (x["news"] or "").lower() or ql in (x["doc_id"] or "").lower()]
    if not hits:
        print(f"no linked news matching '{a.query}'"); return
    bydoc = collections.defaultdict(list)
    for x in hits:
        bydoc[x["doc_id"]].append(x)
    for did, xs in list(bydoc.items())[:5]:
        print(f"\n=== news: {xs[0]['news'][:90]}  ({xs[0]['news_date']})")
        for x in sorted(xs, key=lambda z: -z["score"])[:a.top]:
            mark = "★" if x["via"] == "both" else ("=" if x["via"] == "symbol" else "~")
            print(f"    {mark} {x['score']:.2f} [{x['source']}/{x['via']}] res {x['market_date']}  {x['market_q'][:80]}")


if __name__ == "__main__":
    main()
