# /// script
# requires-python = ">=3.10"
# dependencies = ["python-igraph"]
# ///
"""
Rich, time-stamped graph.json for the React app: causal + sensitivity edges (each
carrying its article date/ts), factor nodes, and a per-entity sentiment timeline.
The app filters by a time window, toggles edge kinds, and can color by sentiment.

Usage: uv run eventgraph/scripts/export_app_json.py --graph-dir /tmp/eg_6k --min-degree 2
"""
import argparse, csv, json, math, re, datetime as dt
from pathlib import Path
import igraph as ig

PAL = {"company":"#4c9be8","bank":"#56B4E9","central_bank":"#E69F00","sovereign":"#22c99a",
       "country":"#22c99a","commodity":"#D55E00","currency":"#CC79A7","equity_index":"#F0E442",
       "person":"#9aa0aa","regulator":"#b07aa1","exchange":"#76b7b2","rate_or_bond":"#ff9da7",
       "sector":"#59a14f","factor":"#e15759","economic_indicator":"#c9a227","other":"#6b7280"}
SLUG2F = {"oil":"oil","crude_oil":"oil","crude":"oil","oil_prices":"oil","crude_oil_prices":"oil","brent":"oil",
          "brent_crude":"oil","energy":"oil","rates":"rates","interest_rate":"rates","interest_rates":"rates",
          "rate":"rates","yields":"rates","treasury_yields":"rates","bond":"rates","bonds":"rates","inflation":"rates",
          "credit":"credit","credit_spread":"credit","credit_spreads":"credit","high_yield":"credit","usd":"usd",
          "dollar":"usd","us_dollar":"usd","u_s_dollar":"usd","currency":"usd","gold":"gold","equity":"market",
          "equities":"market","equity_beta":"market","stocks":"market","market":"market","em":"em","emerging_markets":"em"}

def epoch(s):
    if not s: return None
    try: return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        try: return int(dt.datetime.fromisoformat(s[:10]).timestamp())
        except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--min-degree", type=int, default=2)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"

    # entities (name,type) + tickers
    epat = re.compile(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','((?:[^']|'')*)','([^']*)'")
    ent = {m.group(1): (m.group(2).replace("''", "'"), m.group(3))
           for m in (epat.search(l) for l in open(gd / "pg_upsert.sql")) if m}
    tick = {}
    if (gd / "entity_symbol.jsonl").exists():
        for l in open(gd / "entity_symbol.jsonl"):
            j = json.loads(l); tick[j["entity_id"]] = j["symbol"]

    # doc -> date
    docdate = {}
    for l in open(lake / "document.jsonl"):
        j = json.loads(l); docdate[j["doc_id"]] = j.get("published_at")

    links = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); c, e = j.get("cause_entity"), j.get("effect_entity")
        if not c or not e or c == e: continue
        d = docdate.get(j.get("doc_id"))
        links.append({"source": c, "target": e, "kind": "causal", "ts": epoch(d), "date": (d or "")[:10],
                      "mechanism": j.get("mechanism") or "other", "direction": j.get("effect_dir") or "unchanged",
                      "modality": j.get("modality") or "happened", "quote": (j.get("quote") or "")[:200]})
    # measured betas (weighted posterior) keyed by (asset, canonical factor)
    meas = {}
    rs = lake / "realised_sensitivity.jsonl"
    if rs.exists():
        for l in open(rs):
            m = json.loads(l); meas[(m["entity_id"], m["factor"])] = m
    seen_sens = {}
    for l in open(lake / "sensitivity_edge.jsonl"):
        j = json.loads(l); asset, fid = j.get("asset_entity"), (j.get("factor_id") or "").lower()
        if not asset or not fid: continue
        can = SLUG2F.get(fid, fid)
        key = (asset, can)
        if key in seen_sens: continue          # dedup to one edge per (asset, canonical factor)
        seen_sens[key] = 1
        d = docdate.get(j.get("doc_id")); mm = meas.get(key)
        bstar = mm["beta_star"] if mm else None
        links.append({"source": asset, "target": "factor:" + can, "kind": "sensitivity",
                      "ts": epoch(d), "date": (d or "")[:10], "sign": j.get("sign"),
                      "basis": j.get("basis") or "", "quote": (j.get("quote") or "")[:200],
                      "beta_star": bstar, "r2": mm["r2"] if mm else None,
                      "precision": mm["post_precision"] if mm else None, "agrees": mm["agrees"] if mm else None,
                      "direction": "up" if (bstar if bstar is not None else (j.get("sign") or 0)) > 0 else "down"})

    # degree over combined graph -> prune -> giant component
    deg = {}
    for l in links: deg[l["source"]] = deg.get(l["source"], 0) + 1; deg[l["target"]] = deg.get(l["target"], 0) + 1
    keep = {n for n, d in deg.items() if d >= a.min_degree}
    links = [l for l in links if l["source"] in keep and l["target"] in keep]
    ids0 = sorted({x for l in links for x in (l["source"], l["target"])}); idx0 = {n: i for i, n in enumerate(ids0)}
    g = ig.Graph(); g.add_vertices(len(ids0)); g.add_edges([(idx0[l["source"]], idx0[l["target"]]) for l in links])
    comps = g.connected_components(); big = comps.sizes().index(max(comps.sizes()))
    mem = {ids0[i] for i, c in enumerate(comps.membership) if c == big}
    links = [l for l in links if l["source"] in mem and l["target"] in mem]
    ids = sorted({x for l in links for x in (l["source"], l["target"])})
    mx = max(deg[n] for n in ids)

    def node_of(n):
        if n.startswith("factor:"):
            nm = n[7:]; ty = "factor"; tk = ""
        else:
            nm, ty = ent.get(n, (n, "other")); tk = tick.get(n, "")
        return {"id": n, "name": nm, "type": ty, "ticker": tk, "degree": deg[n],
                "color": PAL.get(ty, "#6b7280"), "val": 1.5 + 8 * math.sqrt(deg[n] / mx)}
    nodes = [node_of(n) for n in ids]

    # sentiment timeline for the nodes we kept (entity, ts, polarity)
    sent = []
    sp = lake / "sentiment_annotation.jsonl"
    if sp.exists():
        for l in open(sp):
            j = json.loads(l); tgt = j.get("target_entity")
            if tgt in mem:
                sent.append({"id": tgt, "ts": epoch(j.get("as_of")), "p": j.get("polarity")})

    ts_all = [l["ts"] for l in links if l["ts"]]
    meta = {"tmin": min(ts_all), "tmax": max(ts_all),
            "n_causal": sum(1 for l in links if l["kind"] == "causal"),
            "n_sensitivity": sum(1 for l in links if l["kind"] == "sensitivity")}
    json.dump({"nodes": nodes, "links": links, "sentiments": sent, "meta": meta}, open(gd / "graph.json", "w"))
    print(f"nodes {len(nodes)}  causal {meta['n_causal']}  sensitivity {meta['n_sensitivity']}  sentiments {len(sent)}")
    print(f"time span {dt.date.fromtimestamp(meta['tmin'])} .. {dt.date.fromtimestamp(meta['tmax'])}")

if __name__ == "__main__":
    main()
