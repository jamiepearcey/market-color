# /// script
# requires-python = ">=3.10"
# dependencies = ["python-igraph"]
# ///
"""
Self-contained interactive HTML graph (vis-network) — opens in any browser.
Zoom/pan/drag, hover nodes+edges for tooltips (with the causal quote), search box,
type-color legend. Layout baked in (igraph FR on the giant component). Local file,
no upload.

Usage: uv run eventgraph/scripts/export_html.py --graph-dir /tmp/eg_6k [--min-degree 3]
"""
import argparse, csv, json, math
from pathlib import Path
import igraph as ig

PAL = {
    "company": "#4c9be8", "bank": "#56B4E9", "central_bank": "#E69F00",
    "sovereign": "#22c99a", "country": "#22c99a", "commodity": "#D55E00",
    "currency": "#CC79A7", "equity_index": "#F0E442", "person": "#9aa0aa",
    "regulator": "#b07aa1", "exchange": "#76b7b2", "rate_or_bond": "#ff9da7",
    "sector": "#59a14f", "factor": "#e15759", "other": "#6b7280",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--min-degree", type=int, default=3)
    ap.add_argument("--label-degree", type=int, default=8, help="only show labels for hubs >= this degree")
    a = ap.parse_args()
    gd = Path(a.graph_dir)

    nrows = {r["id"]: r for r in csv.DictReader(open(gd / "nodes.csv"))}
    erows = list(csv.DictReader(open(gd / "edges.csv")))

    deg = {}
    for r in erows:
        deg[r["source"]] = deg.get(r["source"], 0) + 1
        deg[r["target"]] = deg.get(r["target"], 0) + 1
    keep = {n for n, d in deg.items() if d >= a.min_degree and n in nrows}
    erows = [r for r in erows if r["source"] in keep and r["target"] in keep]
    ids0 = sorted({x for r in erows for x in (r["source"], r["target"])})
    idx0 = {n: i for i, n in enumerate(ids0)}
    g = ig.Graph(directed=True)
    g.add_vertices(len(ids0))
    g.add_edges([(idx0[r["source"]], idx0[r["target"]]) for r in erows])
    comps = g.connected_components(mode="weak")
    biggest = comps.sizes().index(max(comps.sizes()))
    members = set(i for i, c in enumerate(comps.membership) if c == biggest)
    ids = [ids0[i] for i in sorted(members)]
    idset = set(ids)
    erows = [r for r in erows if idx0[r["source"]] in members and idx0[r["target"]] in members]

    g2 = ig.Graph(directed=True)
    g2.add_vertices(len(ids)); idx = {n: i for i, n in enumerate(ids)}
    g2.add_edges([(idx[r["source"]], idx[r["target"]]) for r in erows])
    xy = g2.layout_fruchterman_reingold(niter=1200).coords
    mx = max(deg[n] for n in ids)

    nodes = []
    for i, n in enumerate(ids):
        r = nrows[n]; d = deg[n]
        nodes.append({
            "id": n,
            "label": r["label"] if d >= a.label_degree else "",
            "title": f'{r["label"]}\ntype: {r["type"]}' + (f' | ticker: {r["ticker"]}' if r.get("ticker") else "") + f'\ndegree (edges): {d}',
            "x": xy[i][0] * 14, "y": xy[i][1] * 14,
            "color": PAL.get(r["type"], "#6b7280"),
            "size": 6 + 34 * math.sqrt(d / mx),
            "font": {"size": 10 + 22 * (d / mx), "color": "#e6edf3"},
        })
    DIRC = {"up": "#22c99a", "down": "#f65b5b", "widen": "#f65b5b", "tighten": "#4c9be8"}
    edges = []
    for r in erows:
        q = (r.get("quote") or "").strip()
        edges.append({
            "from": r["source"], "to": r["target"],
            "title": f'{nrows[r["source"]]["label"]} → {nrows[r["target"]]["label"]}\n'
                     f'{r["mechanism"]} / {r["direction"]} ({r["modality"]})' + (f'\n"{q}"' if q else ""),
            "color": {"color": DIRC.get(r["direction"], "#8b98a5"), "opacity": 0.35},
            "arrows": "to",
        })
    legend = "".join(
        f'<span class="lg"><i style="background:{PAL.get(t, "#6b7280")}"></i>{t}</span>'
        for t in sorted({nrows[n]["type"] for n in ids}))

    html = HTML.replace("__NODES__", json.dumps(nodes)).replace("__EDGES__", json.dumps(edges)) \
               .replace("__LEGEND__", legend) \
               .replace("__TITLE__", f"eventgraph — Bloomberg 2006-2013  ·  {len(ids)} entities · {len(erows)} cause→effect edges")
    out = gd / "graph.html"
    out.write_text(html)
    print(f"nodes {len(nodes)}  edges {len(edges)}  -> {out}")

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>eventgraph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#0e1116;color:#e6edf3;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
  #net{width:100%;height:100vh}
  #bar{position:fixed;top:10px;left:10px;z-index:5;background:#161b22cc;border:1px solid #30363d;border-radius:8px;padding:8px 10px;max-width:340px}
  #bar h1{font-size:13px;margin:0 0 6px}
  #q{width:200px;background:#0e1116;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:4px 6px}
  #legend{margin-top:8px;font-size:11px;line-height:1.7}
  .lg{display:inline-block;margin-right:10px;white-space:nowrap}
  .lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
  .hint{font-size:11px;color:#8b98a5;margin-top:6px}
</style></head><body>
<div id="bar">
  <h1>__TITLE__</h1>
  <input id="q" placeholder="find an entity (e.g. greece)…" autocomplete="off">
  <div id="legend">__LEGEND__</div>
  <div class="hint">scroll = zoom · drag = pan · hover a node/edge for details · click a node to isolate it</div>
</div>
<div id="net"></div>
<script>
const nodes=new vis.DataSet(__NODES__), edges=new vis.DataSet(__EDGES__);
const net=new vis.Network(document.getElementById('net'),{nodes,edges},{
  physics:false, interaction:{hover:true,tooltipDelay:80,navigationButtons:true,keyboard:true},
  nodes:{shape:'dot',borderWidth:0},
  edges:{smooth:false,width:0.5,arrows:{to:{scaleFactor:0.35}}}
});
net.once('afterDrawing',()=>net.fit());
// click a node -> highlight its neighborhood
net.on('click',p=>{
  if(!p.nodes.length){edges.forEach(e=>edges.update({id:e.id,color:{opacity:0.35}}));return;}
  const id=p.nodes[0], con=new Set([id]);
  edges.forEach(e=>{if(e.from===id||e.to===id){con.add(e.from);con.add(e.to);}});
  edges.forEach(e=>edges.update({id:e.id,color:{opacity:(e.from===id||e.to===id)?0.9:0.05}}));
});
// search
document.getElementById('q').addEventListener('keydown',ev=>{
  if(ev.key!=='Enter')return;
  const s=ev.target.value.trim().toLowerCase(); if(!s)return;
  let hit=null; nodes.forEach(n=>{if(!hit && (n.id.toLowerCase().includes(s)||(n.label||'').toLowerCase().includes(s)))hit=n.id;});
  if(hit){net.selectNodes([hit]);net.focus(hit,{scale:1.4,animation:true});}
});
</script></body></html>"""

if __name__ == "__main__":
    main()
