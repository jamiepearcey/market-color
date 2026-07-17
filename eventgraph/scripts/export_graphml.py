# /// script
# requires-python = ">=3.10"
# ///
"""
Export the eventgraph causal graph for visualization tools.

Reads a graph-dir (pg_upsert.sql + lake/*.jsonl + entity_symbol.jsonl) and writes:
  graph.graphml   -> Gephi / yEd / Cytoscape (directed, typed nodes, edge attrs)
  nodes.csv       -> Cosmograph / Neo4j / Gephi   (id,label,type,ticker,degree,...)
  edges.csv       -> Cosmograph / Neo4j / Gephi   (source,target,mechanism,direction,weight,quote)

Nodes = entities that appear in causal edges (isolated ones dropped). Parallel
cause->effect edges are aggregated to one weighted edge (weight = # docs asserting
it), keeping the dominant mechanism/direction and one example quote. `--min-degree`
prunes leaves for a cleaner picture.

Usage: uv run eventgraph/scripts/export_graphml.py --graph-dir /tmp/eg_6k [--min-degree 2]
"""
import argparse, json, re, collections
from pathlib import Path

def load_entities(pg):
    pat = re.compile(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','((?:[^']|'')*)','([^']*)'")
    return {m.group(1): (m.group(2).replace("''", "'"), m.group(3))
            for m in (pat.search(l) for l in open(pg)) if m}

def xesc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))

def cesc(s):  # CSV field
    s = str(s or "").replace('"', '""')
    return f'"{s}"'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--min-degree", type=int, default=1)
    ap.add_argument("--sensitivities", action="store_true", help="also include asset~factor edges")
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    ents = load_entities(gd / "pg_upsert.sql")
    sym = {}
    p = gd / "entity_symbol.jsonl"
    if p.exists():
        for l in open(p):
            j = json.loads(l); sym[j["entity_id"]] = j["symbol"]

    # aggregate causal edges by (cause,effect)
    agg = {}
    for l in open(gd / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        c, e = j.get("cause_entity"), j.get("effect_entity")
        if not c or not e or c == e:
            continue
        k = (c, e)
        d = agg.setdefault(k, {"w": 0, "mech": collections.Counter(), "dir": collections.Counter(),
                               "mod": collections.Counter(), "quote": ""})
        d["w"] += 1
        d["mech"][j.get("mechanism") or "other"] += 1
        d["dir"][j.get("effect_dir") or "unchanged"] += 1
        d["mod"][j.get("modality") or "happened"] += 1
        if not d["quote"] and j.get("quote"):
            d["quote"] = j["quote"][:160]

    sens = []
    if a.sensitivities and (gd / "lake" / "sensitivity_edge.jsonl").exists():
        for l in open(gd / "lake" / "sensitivity_edge.jsonl"):
            j = json.loads(l)
            if j.get("asset_entity") and j.get("factor_id"):
                sens.append((j["asset_entity"], "factor:" + j["factor_id"], j.get("sign")))

    # degrees over the edge set (before pruning)
    deg = collections.Counter()
    for (c, e) in agg: deg[c] += 1; deg[e] += 1
    for (aE, f, _) in sens: deg[aE] += 1; deg[f] += 1

    keep = {n for n in deg if deg[n] >= a.min_degree}
    agg = {k: v for k, v in agg.items() if k[0] in keep and k[1] in keep}
    sens = [(aE, f, s) for (aE, f, s) in sens if aE in keep and f in keep]
    nodes = set(keep) & (set(x for k in agg for x in k) | set(x for t in sens for x in t[:2]))

    # ---- GraphML ----
    with open(gd / "graph.graphml", "w") as g:
        g.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for k, t in [("label", "string"), ("etype", "string"), ("ticker", "string"), ("degree", "int")]:
            g.write(f'  <key id="{k}" for="node" attr.name="{k}" attr.type="{t}"/>\n')
        for k, t in [("mechanism", "string"), ("direction", "string"), ("modality", "string"),
                     ("weight", "int"), ("quote", "string"), ("kind", "string")]:
            g.write(f'  <key id="{k}" for="edge" attr.name="{k}" attr.type="{t}"/>\n')
        g.write('  <graph edgedefault="directed">\n')
        for n in nodes:
            if n.startswith("factor:"):
                name, et, tk = n[7:], "factor", n[7:]
            else:
                name, et = ents.get(n, (n, "other")); tk = sym.get(n, "")
            g.write(f'    <node id="{xesc(n)}"><data key="label">{xesc(name)}</data>'
                    f'<data key="etype">{xesc(et)}</data><data key="ticker">{xesc(tk)}</data>'
                    f'<data key="degree">{deg[n]}</data></node>\n')
        for (c, e), d in agg.items():
            g.write(f'    <edge source="{xesc(c)}" target="{xesc(e)}">'
                    f'<data key="mechanism">{xesc(d["mech"].most_common(1)[0][0])}</data>'
                    f'<data key="direction">{xesc(d["dir"].most_common(1)[0][0])}</data>'
                    f'<data key="modality">{xesc(d["mod"].most_common(1)[0][0])}</data>'
                    f'<data key="weight">{d["w"]}</data><data key="kind">causal</data>'
                    f'<data key="quote">{xesc(d["quote"])}</data></edge>\n')
        for (aE, f, s) in sens:
            g.write(f'    <edge source="{xesc(aE)}" target="{xesc(f)}">'
                    f'<data key="direction">{"up" if (s or 0) > 0 else "down"}</data>'
                    f'<data key="kind">sensitivity</data><data key="weight">1</data></edge>\n')
        g.write('  </graph>\n</graphml>\n')

    # ---- CSVs (Cosmograph / Neo4j / Gephi) ----
    with open(gd / "nodes.csv", "w") as f:
        f.write("id,label,type,ticker,degree\n")
        for n in nodes:
            if n.startswith("factor:"):
                name, et, tk = n[7:], "factor", n[7:]
            else:
                name, et = ents.get(n, (n, "other")); tk = sym.get(n, "")
            f.write(f'{cesc(n)},{cesc(name)},{cesc(et)},{cesc(tk)},{deg[n]}\n')
    with open(gd / "edges.csv", "w") as f:
        f.write("source,target,mechanism,direction,modality,weight,quote\n")
        for (c, e), d in agg.items():
            f.write(f'{cesc(c)},{cesc(e)},{cesc(d["mech"].most_common(1)[0][0])},'
                    f'{cesc(d["dir"].most_common(1)[0][0])},{cesc(d["mod"].most_common(1)[0][0])},'
                    f'{d["w"]},{cesc(d["quote"])}\n')

    print(f"nodes: {len(nodes)}  causal edges: {len(agg)}  sensitivity edges: {len(sens)}  (min_degree={a.min_degree})")
    print(f"wrote {gd/'graph.graphml'}, {gd/'nodes.csv'}, {gd/'edges.csv'}")

if __name__ == "__main__":
    main()
