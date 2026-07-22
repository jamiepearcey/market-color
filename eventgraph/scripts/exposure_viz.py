# /// script
# requires-python = ">=3.10"
# ///
"""Render exposure_landscape_emb.json -> interactive EVENT-EXPOSURE LANDSCAPE. Pick an event; see the
full-universe exposure ranking (semantic embedding), named firms vs the unnamed firms the model
SURFACES, each with realized co-movement. The event-anchored view: exposure beyond the journalist's list."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
D = json.loads((G / "exposure_landscape_emb.json").read_text())

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Event-exposure landscape</title>
<style>
:root{--surface:#151514;--panel:#1e1e1c;--panel2:#262624;--ink:#fff;--ink2:#c3c2b7;--ink3:#86857c;--grid:#2b2b29;--line:#323230;--pos:#3987e5;--neg:#d95926;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--surface);color:var(--ink);font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:300px 1fr;height:100vh}
.side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
.brand{padding:15px 16px 11px;border-bottom:1px solid var(--line)}
.brand h1{font-size:14px;letter-spacing:-.01em}.brand p{font-size:11px;color:var(--ink3);margin-top:3px}
.search{margin:10px 12px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-size:12.5px;width:calc(100% - 24px)}
.list{flex:1;overflow-y:auto;padding:2px 8px 20px}
.ev{padding:7px 10px;border-radius:9px;cursor:pointer}.ev:hover{background:var(--panel2)}.ev.sel{background:var(--pos);color:#fff}
.ev .t{font-weight:600;font-size:12.5px;display:flex;justify-content:space-between;gap:6px}
.ev .s{font-size:10.5px;color:var(--ink3)}.ev.sel .s{color:rgba(255,255,255,.8)}
.main{overflow-y:auto;padding:22px 28px 60px}
.hd{font-size:20px;font-weight:650;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.badge{font-size:10px;padding:1px 8px;border-radius:20px;background:var(--panel2);color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--ink2);font-size:13px;margin:6px 0 4px;max-width:820px}
.sub b{color:var(--ink)}
.leg{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:12px 0 8px}.leg span{display:flex;gap:6px;align-items:center}
.dot{width:10px;height:10px;border-radius:3px}.tagn{background:var(--pos)}.tags{border:1px dashed var(--ink3)}
.rows{margin-top:4px}
.row{display:grid;grid-template-columns:26px 120px 1fr 96px;gap:10px;align-items:center;padding:3px 0}
.rk{color:var(--ink3);font-size:11px;text-align:right;font-variant-numeric:tabular-nums}
.nm{display:flex;gap:7px;align-items:center;min-width:0}
.nm .tk{font-weight:700;font-size:13px}.nm .sc{width:9px;height:9px;border-radius:2px;flex:none}
.tag{font-size:9px;padding:1px 6px;border-radius:20px;font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.tag.n{background:var(--pos);color:#fff}.tag.s{border:1px dashed var(--ink3);color:var(--ink3)}
.bar{height:15px;border-radius:4px;position:relative;background:var(--panel2)}
.bar .fill{height:100%;border-radius:4px;opacity:.85}
.rl{font-size:11.5px;text-align:right;font-variant-numeric:tabular-nums}
.rl b{font-weight:700}
.note{color:var(--ink3);font-size:12px;margin-top:20px;max-width:820px}
::-webkit-scrollbar{width:9px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px}
</style></head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand"><h1>Event-exposure landscape</h1><p>Who's exposed to an event — named <i>or not</i>. Semantic embeddings, BBG 2010–2012.</p></div>
    <input class="search" id="search" placeholder="Search events…">
    <div class="list" id="list"></div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script>
const EV=__DATA__.events, SECTORS=__DATA__.sectors;
const PAL=["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"];
const sc=s=>PAL[Math.max(0,SECTORS.indexOf(s))%PAL.length];
EV.forEach((e,i)=>e._i=i);
EV.sort((a,b)=>a.month<b.month?-1:1);
let sel=EV.find(e=>e.label&&e.label.includes("Almunia"))||EV[0];
const el=id=>document.getElementById(id);
function renderList(){
  const q=el("search").value.toLowerCase();
  el("list").innerHTML=EV.filter(e=>!q||e.label.toLowerCase().includes(q)||e.type.includes(q)).map(e=>
    `<div class="ev ${e===sel?'sel':''}" data-i="${e._i}"><div class="t"><span>${e.label}</span></div><div class="s">${e.month} · ${e.type} · named ${e.n_named}</div></div>`).join("");
}
function render(){
  const e=sel; const rows=e.rows.filter(r=>r.exp>0);
  const mx=Math.max(...rows.map(r=>r.exp),0.1);
  const surfaced=rows.filter(r=>!r.m).slice(0,6).map(r=>r.t);
  el("main").innerHTML=`<div class="hd">${e.label} <span class="badge">${e.type}</span> <span style="color:var(--ink3);font-size:14px">${e.month}</span></div>
    <p class="sub">The article <b>named ${e.n_named}</b> firm${e.n_named>1?'s':''}. The exposure map below ranks the <b>whole universe</b> by
    semantic similarity to the event — surfacing the unnamed firms most exposed (top: <b>${surfaced.join(', ')}</b>). <b>Realized</b> = each firm's
    actual co-movement with the event during its window, after macro+sector — the check that a surfaced exposure is real.</p>
    <div class="leg"><span><span class="dot tagn"></span>named by the article</span><span><span class="dot tags"></span>surfaced (unnamed)</span><span style="color:var(--ink3)">bar = exposure · right = realized co-move</span></div>
    <div class="rows">`+rows.slice(0,28).map((r,i)=>{
      const rl=r.real==null?'—':(r.real>=0?'+':'')+r.real.toFixed(2);
      const rlc=r.real==null?'var(--ink3)':(r.real>=0.05?'var(--pos)':(r.real<=-0.05?'var(--neg)':'var(--ink3)'));
      return `<div class="row"><div class="rk">${i+1}</div>
        <div class="nm"><span class="sc" style="background:${sc(r.sec)}"></span><span class="tk">${r.t}</span><span class="tag ${r.m?'n':'s'}">${r.m?'named':'surf'}</span></div>
        <div class="bar"><div class="fill" style="width:${(r.exp/mx*100).toFixed(0)}%;background:${r.m?'var(--pos)':sc(r.sec)}"></div></div>
        <div class="rl">real <b style="color:${rlc}">${rl}</b></div></div>`}).join("")+`</div>
    <p class="note">Read: the top of the list is usually the named firms (their profile contains the event), but the <b>surfaced</b> names just below are the payoff —
    firms the journalist never mentioned that the event still touches. A high exposure with a high <b>realized</b> co-move is a real exposure the mention graph missed
    (e.g. HSBC in the LIBOR probe). Descriptive, small, and embedding-based — a different, fuller view than the named list.</p>`;
}
function all(){renderList();render()}
el("search").oninput=renderList;
el("list").addEventListener("click",ev=>{const r=ev.target.closest(".ev");if(!r)return;sel=EV[+r.dataset.i];all()});
all();
</script></body></html>"""
out = G / "exposure_landscape.html"
out.write_text(HTML.replace("__DATA__", json.dumps(D)))
print(f"-> {out}  ({len(D['events'])} events)")
