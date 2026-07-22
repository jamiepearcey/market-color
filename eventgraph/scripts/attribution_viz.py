# /// script
# requires-python = ">=3.10"
# ///
"""Render attribution.json -> interactive PRICE-MOVE ATTRIBUTION. Pick a firm, see its biggest monthly
moves decomposed into MARKET/macro + SECTOR + IDIOSYNCRATIC, with the named news events (verbatim quote +
source article) that explain the idiosyncratic part. Contemporaneous, first-order, direct — no prediction."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
D = json.loads((G / "attribution.json").read_text())

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Price-move attribution</title>
<style>
:root{--surface:#151514;--panel:#1e1e1c;--panel2:#262624;--ink:#fff;--ink2:#c3c2b7;--ink3:#86857c;--grid:#2b2b29;--line:#323230;--mac:#6b6a63;--sec:#3987e5;--idio:#d95926;--pos:#199e70;--neg:#d95926;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--surface);color:var(--ink);font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:300px 1fr;height:100vh}
.side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
.brand{padding:15px 16px 11px;border-bottom:1px solid var(--line)}.brand h1{font-size:14.5px}.brand p{font-size:11px;color:var(--ink3);margin-top:3px}
.search{margin:11px 12px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-size:12.5px;width:calc(100% - 24px)}
.list{flex:1;overflow-y:auto;padding:2px 8px 20px}
.row{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:7px 10px;border-radius:9px;cursor:pointer}
.row:hover{background:var(--panel2)}.row.sel{background:var(--sec);color:#fff}
.row .nm{font-weight:600;font-size:12.5px;display:flex;gap:6px;align-items:center;min-width:0}.row .nm b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .nm .tk{opacity:.6;font-weight:500}.row .sc{width:8px;height:8px;border-radius:2px;flex:none}
.main{overflow-y:auto;padding:22px 28px 60px}
.hd{font-size:21px;font-weight:650;display:flex;gap:10px;align-items:baseline}.hd .tk{color:var(--ink3);font-weight:500;font-size:15px}
.sub{color:var(--ink2);font-size:13px;margin:5px 0 16px;max-width:820px}
.moves{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
.mv{padding:7px 12px;border:1px solid var(--line);border-radius:9px;cursor:pointer;background:var(--panel);font-variant-numeric:tabular-nums}
.mv:hover{border-color:var(--ink3)}.mv.on{border-color:var(--sec);background:var(--panel2)}
.mv .mo{font-size:11px;color:var(--ink3)}.mv .rt{font-size:15px;font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.card h3{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);margin-bottom:12px;font-weight:600}
.decomp{display:flex;flex-direction:column;gap:9px}
.drow{display:grid;grid-template-columns:120px 1fr 74px;gap:12px;align-items:center}
.dlab{font-size:13px;color:var(--ink2);display:flex;gap:7px;align-items:center}.dlab .sw{width:11px;height:11px;border-radius:3px}
.dbarwrap{position:relative;height:20px;background:var(--panel2);border-radius:4px}
.dbar{position:absolute;top:0;height:100%;border-radius:4px;opacity:.9}
.dmid{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink3);left:50%}
.dval{font-size:13.5px;font-weight:700;text-align:right;font-variant-numeric:tabular-nums}
.expl{color:var(--ink3);font-size:12px;margin-top:11px}
.fact{border-left:3px solid var(--idio);padding:4px 0 6px 12px;margin-bottom:9px}
.fq{font-size:13.5px}.fq .fd{font-weight:700}.fq q{font-style:italic;color:var(--ink)}
.fcat{font-weight:600;color:var(--ink)}.fmech{font-size:9.5px;padding:1px 6px;border-radius:20px;background:var(--panel2);color:var(--ink3);text-transform:uppercase;margin-left:6px}
.fsrc{display:block;font-size:11.5px;color:var(--ink3);text-decoration:none;margin-top:2px}.fsrc:hover{color:var(--sec);text-decoration:underline}.fsrc .src{text-transform:uppercase;font-size:10px}
.note{color:var(--ink3);font-size:12px;margin-top:14px;max-width:820px}
::-webkit-scrollbar{width:9px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px}
</style></head>
<body><div class="app">
  <aside class="side"><div class="brand"><h1>Price-move attribution</h1><p>Why did it move? Market vs sector vs news — with the article. BBG 2010–2012.</p></div>
    <input class="search" id="search" placeholder="Search firms…"><div class="list" id="list"></div></aside>
  <main class="main" id="main"></main>
</div>
<script>
const F=__DATA__.firms, SECTORS=__DATA__.sectors;
const PAL=["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"];
const sc=s=>PAL[Math.max(0,SECTORS.indexOf(s))%PAL.length];
F.forEach((f,i)=>f._i=i);
let sel=F[0], mv=sel.moves[0];
const el=id=>document.getElementById(id);
const pct=v=>(v>=0?'+':'')+(v*100).toFixed(1)+'%';
function renderList(){
  const q=el("search").value.toLowerCase();
  el("list").innerHTML=F.filter(f=>!q||f.n.toLowerCase().includes(q)||f.t.toLowerCase().includes(q)).map(f=>{
    const big=f.moves.reduce((a,m)=>Math.abs(m.tot)>Math.abs(a.tot)?m:a,f.moves[0]);
    return `<div class="row ${f===sel?'sel':''}" data-i="${f._i}"><div class="nm"><span class="sc" style="background:${sc(f.sec)}"></span><b>${f.n}</b> <span class="tk">${f.t}</span></div><div style="font-variant-numeric:tabular-nums;font-weight:600;color:${big.tot>=0?'var(--pos)':'var(--neg)'}">${pct(big.tot)}</div></div>`}).join("");
}
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function render(){
  const f=sel; if(!f.moves.includes(mv)) mv=f.moves[0];
  const mvs=f.moves.map(m=>`<div class="mv ${m===mv?'on':''}" data-m="${m.m}"><div class="mo">${m.m}</div><div class="rt" style="color:${m.tot>=0?'var(--pos)':'var(--neg)'}">${pct(m.tot)}</div></div>`).join("");
  const parts=[["Market / macro","--mac",mv.macro],["Sector ("+f.sec+")","--sec",mv.sector],["Idiosyncratic","--idio",mv.idio]];
  const scale=Math.max(...parts.map(p=>Math.abs(p[2])),Math.abs(mv.tot),0.01);
  const dec=parts.map(([lab,cv,v])=>{
    const w=Math.abs(v)/scale*50, left=v>=0?50:50-w;
    return `<div class="drow"><div class="dlab"><span class="sw" style="background:var(${cv})"></span>${lab}</div>
      <div class="dbarwrap"><div class="dmid"></div><div class="dbar" style="left:${left}%;width:${w}%;background:var(${cv})"></div></div>
      <div class="dval" style="color:${v>=0?'var(--pos)':'var(--neg)'}">${pct(v)}</div></div>`}).join("");
  const facts=mv.events.map(e=>`<div class="fact"><div class="fq"><span class="fd" style="color:${e.dir>=0?'var(--pos)':'var(--neg)'}">${e.dir>=0?'▲':'▼'}</span> <span class="fcat">${esc(e.cat)}</span> — <q>${esc(e.q)}</q>${e.mech?`<span class="fmech">${esc(e.mech)}</span>`:''}</div>`+
    (e.u?`<a class="fsrc" href="${esc(e.u)}" target="_blank">${esc(e.h)||'(article)'} · <span class="src">${esc(e.s)}</span> · ${e.d||''} ↗</a>`:`<div class="fsrc">${esc(e.h)||''} · ${e.d||''}</div>`)+`</div>`).join("");
  el("main").innerHTML=`<div class="hd">${f.n} <span class="tk">${f.t}</span></div>
    <p class="sub">Pick a month to attribute the move. Each is split additively into <b>market/macro</b>, <b>sector</b>, and <b>idiosyncratic</b> — and the idiosyncratic part is explained by the named news below. Contemporaneous &amp; first-order — no forecasting.</p>
    <div class="moves">${mvs}</div>
    <div class="card"><h3>${mv.m} · total move ${pct(mv.tot)} — attribution</h3><div class="decomp">${dec}</div>
      <p class="expl">Of the ${pct(mv.tot)} move, <b>${pct(mv.macro)}</b> was the market/macro, <b>${pct(mv.sector)}</b> the sector, and <b style="color:var(--idio)">${pct(mv.idio)}</b> firm-specific — the part the news below explains.</p></div>
    <div class="card"><h3>news driving the idiosyncratic move — ${mv.m}</h3>${facts||'<span class="note">no quoted event captured this month</span>'}</div>
    <p class="note">Honest note: for most moves the market + sector dominate — the news names the <i>idiosyncratic</i> slice (often small, sometimes large for M&amp;A/litigation). This view is <b>explanation</b>, grounded in the source article — not prediction. The idiosyncratic % is the abnormal (macro+sector-removed) return; the news is the contemporaneous catalyst on record.</p>`;
}
function all(){renderList();render()}
el("search").oninput=renderList;
el("list").addEventListener("click",ev=>{const r=ev.target.closest(".row");if(!r)return;sel=F[+r.dataset.i];mv=sel.moves[0];all()});
el("main").addEventListener("click",ev=>{const m=ev.target.closest(".mv");if(!m)return;mv=sel.moves.find(x=>x.m===m.dataset.m);render()});
all();
</script></body></html>"""
out = G / "attribution.html"
out.write_text(HTML.replace("__DATA__", json.dumps(D)))
print(f"-> {out}  ({len(D['firms'])} firms)")
