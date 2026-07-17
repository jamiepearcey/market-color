# /// script
# requires-python = ">=3.10"
# dependencies = ["python-igraph", "matplotlib"]
# ///
"""
Headless 'Gephi-style' render of the causal graph: force-directed layout,
nodes colored by entity type + sized by degree, directed edges, top hubs labeled.
Reads nodes.csv/edges.csv from a graph-dir; writes graph.png.

Usage: uv run eventgraph/scripts/render_graph.py --graph-dir /tmp/eg_6k [--min-degree 4]
"""
import argparse, csv, math
from pathlib import Path
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def xesc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

# Okabe-Ito colorblind-safe palette, one per entity type
PAL = {
    "company": "#0072B2", "bank": "#56B4E9", "central_bank": "#E69F00",
    "sovereign": "#009E73", "country": "#009E73", "commodity": "#D55E00",
    "currency": "#CC79A7", "equity_index": "#F0E442", "person": "#8a8f98",
    "regulator": "#b07aa1", "exchange": "#76b7b2", "rate_or_bond": "#ff9da7",
    "sector": "#59a14f", "factor": "#e15759", "other": "#5a6069",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--min-degree", type=int, default=4, help="prune leaves for a legible picture")
    ap.add_argument("--labels", type=int, default=40, help="label the top-N hubs")
    a = ap.parse_args()
    gd = Path(a.graph_dir)

    nrows = {r["id"]: r for r in csv.DictReader(open(gd / "nodes.csv"))}
    edges = [(r["source"], r["target"], int(r["weight"] or 1))
             for r in csv.DictReader(open(gd / "edges.csv"))]

    # prune by degree
    deg = {}
    for s, t, _ in edges:
        deg[s] = deg.get(s, 0) + 1; deg[t] = deg.get(t, 0) + 1
    keep = {n for n, d in deg.items() if d >= a.min_degree and n in nrows}
    edges = [(s, t, w) for s, t, w in edges if s in keep and t in keep]
    ids0 = sorted({x for s, t, _ in edges for x in (s, t)})
    idx0 = {n: i for i, n in enumerate(ids0)}
    g0 = ig.Graph(directed=True)
    g0.add_vertices(len(ids0))
    g0.add_edges([(idx0[s], idx0[t]) for s, t, _ in edges])

    # keep only the largest weakly-connected component (drop scattered singletons
    # that otherwise blow out the scale and crush the real graph)
    comps = g0.connected_components(mode="weak")
    members = [i for i, c in enumerate(comps.membership) if c == comps.sizes().index(max(comps.sizes()))]
    memset = set(members)
    ids = [ids0[i] for i in members]
    idx = {n: i for i, n in enumerate(ids)}
    edges = [(s, t, w) for s, t, w in edges if idx0[s] in memset and idx0[t] in memset]
    g = ig.Graph(directed=True)
    g.add_vertices(len(ids))
    g.add_edges([(idx[s], idx[t]) for s, t, _ in edges])
    print(f"rendering giant component: {len(ids)} nodes / {len(edges)} edges (min_degree={a.min_degree}) ...")

    layout = g.layout_fruchterman_reingold(niter=1500)   # spreads dense hub graphs
    xy = layout.coords
    degv = [deg[n] for n in ids]
    mx = max(degv)
    colors = [PAL.get(nrows[n]["type"], "#5a6069") for n in ids]
    sizes = [12 + 340 * math.sqrt(d / mx) for d in degv]

    fig, ax = plt.subplots(figsize=(26, 26), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    segs = [(xy[idx[s]], xy[idx[t]]) for s, t, _ in edges]
    ax.add_collection(LineCollection(segs, colors="#2a3140", linewidths=0.25, alpha=0.5, zorder=1))
    xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.92, linewidths=0, zorder=2)

    top = sorted(range(len(ids)), key=lambda i: -degv[i])[:a.labels]
    for i in top:
        ax.text(xy[i][0], xy[i][1], nrows[ids[i]]["label"], color="white",
                fontsize=6.5 + 9 * (degv[i] / mx), ha="center", va="center", zorder=3,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="#0e1116", ec="none", alpha=0.55))
    xr = (max(xs) - min(xs)) * 0.04; yr = (max(ys) - min(ys)) * 0.04
    ax.set_xlim(min(xs) - xr, max(xs) + xr); ax.set_ylim(min(ys) - yr, max(ys) + yr)
    # legend
    from matplotlib.patches import Patch
    seen = sorted({nrows[n]["type"] for n in ids})
    ax.legend(handles=[Patch(color=PAL.get(t, "#5a6069"), label=t) for t in seen],
              loc="upper left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=11, title="entity type", title_fontsize=12)
    ax.set_title(f"eventgraph causal graph — Bloomberg 2006-2013  ({len(ids)} entities, {len(edges)} cause→effect edges)",
                 color="white", fontsize=20, pad=16)
    ax.axis("off")
    out = gd / "graph.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0e1116")
    print(f"wrote {out}")

    # ---- GEXF with layout + color + size baked in (Gephi shows it immediately) ----
    def h2rgb(h):
        h = h.lstrip("#"); return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    gx = gd / "graph.gexf"
    with open(gx, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">\n')
        f.write('<graph defaultedgetype="directed" mode="static">\n')
        f.write('<attributes class="node"><attribute id="0" title="type" type="string"/>'
                '<attribute id="1" title="ticker" type="string"/>'
                '<attribute id="2" title="degree" type="integer"/></attributes>\n<nodes>\n')
        for i, n in enumerate(ids):
            r, gg, b = h2rgb(colors[i]); tk = nrows[n].get("ticker", "") or ""
            f.write(f'<node id="{xesc(n)}" label="{xesc(nrows[n]["label"])}">'
                    f'<attvalues><attvalue for="0" value="{xesc(nrows[n]["type"])}"/>'
                    f'<attvalue for="1" value="{xesc(tk)}"/><attvalue for="2" value="{deg[n]}"/></attvalues>'
                    f'<viz:position x="{xy[i][0]*10:.2f}" y="{xy[i][1]*10:.2f}" z="0"/>'
                    f'<viz:size value="{sizes[i]/18:.2f}"/><viz:color r="{r}" g="{gg}" b="{b}"/></node>\n')
        f.write('</nodes>\n<edges>\n')
        for j, (s, t, w) in enumerate(edges):
            f.write(f'<edge id="{j}" source="{xesc(s)}" target="{xesc(t)}" weight="{w}"/>\n')
        f.write('</edges>\n</graph>\n</gexf>\n')
    print(f"wrote {gx}")

if __name__ == "__main__":
    main()
