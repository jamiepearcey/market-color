# /// script
# requires-python = ">=3.10"
# ///
"""Unified DEMO — one discoverable app for the whole event-intelligence system: thesis + headline
numbers, a browsable/searchable event catalogue, a per-event DOSSIER (source article -> named firms ->
who is ACTUALLY exposed across the whole market -> realized co-movement validation), the economic
mechanisms, and an honest 'what it can & can't do' panel. Built on exposure_landscape_emb.json."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
D = json.loads((G / "exposure_landscape_emb.json").read_text())
MECH = [
 ("US financial legislation", "Dodd-Frank, Barney Frank, Carl Levin — regulatory reform of banks"),
 ("UK banking reform", "Vickers / Independent Commission on Banking / Project Merlin — ring-fencing"),
 ("Antitrust & enforcement", "Almunia / EU / SEC — the LIBOR-Euribor collusion probe"),
 ("Credit downgrades", "Moody's / CDS / Deutsche Bank — rating actions and contagion"),
 ("Bank capital rules", "Basel Committee / Volcker Rule — capital & prop-trading"),
 ("Industrial metals", "copper / nickel / tin — China demand"),
 ("German industrials", "BMW / Audi / Evonik — autos & chemicals"),
 ("Labour / China industry", "mineworkers / steel association — strikes & output"),
 ("Natural disasters", "Japan earthquake-tsunami / Hurricane Irene"),
 ("US housing / GSEs", "FHFA / DeMarco / Freddie-Fannie — mortgage complex"),
]
STATS = [
 ("116", "named events, 2010–2012"),
 ("~10", "economic mechanisms"),
 ("+0.10", "exposure IC (who's exposed)"),
 ("chance", "direction (up/down) — honest limit"),
]

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Market Event Intelligence — demo</title>
<style>
:root{--surface:#151514;--panel:#1e1e1c;--panel2:#262624;--ink:#fff;--ink2:#c3c2b7;--ink3:#86857c;--grid:#2b2b29;--line:#323230;--pos:#3987e5;--neg:#d95926;--gd:#199e70;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--surface);color:var(--ink);font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
a{color:inherit}
.hero{padding:22px 30px 18px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#1b1b19,#151514)}
.hero h1{font-size:20px;letter-spacing:-.01em}
.hero p{color:var(--ink2);font-size:13.5px;max-width:900px;margin-top:6px}
.hero p b{color:var(--ink)}
.chips{display:flex;gap:22px;margin-top:14px;flex-wrap:wrap}
.chip{display:flex;flex-direction:column}.chip .v{font-size:22px;font-weight:700;letter-spacing:-.02em}.chip .k{font-size:11px;color:var(--ink3)}
.tabs{display:flex;gap:6px;padding:10px 30px 0}
.tab{padding:7px 14px;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;cursor:pointer;color:var(--ink2);font-size:13px;background:var(--panel)}
.tab.on{background:var(--surface);color:var(--ink);border-color:var(--line)}
.view{display:none}.view.on{display:block}
.app{display:grid;grid-template-columns:320px 1fr;min-height:70vh}
.side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;max-height:calc(100vh - 200px);position:sticky;top:0}
.search{margin:12px;padding:8px 11px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px;width:calc(100% - 24px)}
.list{overflow-y:auto;padding:2px 8px 24px}
.grp{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);padding:10px 10px 4px}
.ev{padding:7px 10px;border-radius:9px;cursor:pointer}.ev:hover{background:var(--panel2)}.ev.sel{background:var(--pos);color:#fff}
.ev .t{font-weight:600;font-size:12.5px}.ev .s{font-size:10.5px;color:var(--ink3)}.ev.sel .s{color:rgba(255,255,255,.85)}
.main{padding:22px 30px 60px;overflow-y:auto}
.hd{font-size:21px;font-weight:650;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.badge{font-size:10px;padding:1px 8px;border-radius:20px;background:var(--panel2);color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px 17px;margin:14px 0}
.card h3{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);margin-bottom:9px;font-weight:600}
.fq q{font-style:italic;color:var(--ink);font-size:13.5px}
.fmech{font-size:9.5px;padding:1px 6px;border-radius:20px;background:var(--panel2);color:var(--ink3);text-transform:uppercase;margin-left:6px}
.fsrc{display:block;font-size:11.5px;color:var(--ink3);text-decoration:none;margin-top:3px}.fsrc:hover{color:var(--pos);text-decoration:underline}.fsrc .src{text-transform:uppercase;font-size:10px}
.named{display:flex;gap:6px;flex-wrap:wrap}.pill{font-size:12px;font-weight:600;padding:3px 9px;border-radius:7px;background:var(--pos);color:#fff}
.leg{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:2px 0 10px}.leg span{display:flex;gap:6px;align-items:center}
.dot{width:10px;height:10px;border-radius:3px}.tn{background:var(--pos)}.ts{border:1px dashed var(--ink3)}
.row{display:grid;grid-template-columns:24px 116px 1fr 92px;gap:10px;align-items:center;padding:3px 0}
.rk{color:var(--ink3);font-size:11px;text-align:right;font-variant-numeric:tabular-nums}
.nm{display:flex;gap:7px;align-items:center;min-width:0}.nm .tk{font-weight:700;font-size:13px}.nm .sc{width:9px;height:9px;border-radius:2px;flex:none}
.tag{font-size:9px;padding:1px 6px;border-radius:20px;font-weight:700;text-transform:uppercase}.tag.n{background:var(--pos);color:#fff}.tag.s{border:1px dashed var(--ink3);color:var(--ink3)}
.bar{height:14px;border-radius:4px;background:var(--panel2)}.bar .fill{height:100%;border-radius:4px;opacity:.85}
.rl{font-size:11.5px;text-align:right;font-variant-numeric:tabular-nums}
.note{color:var(--ink3);font-size:12px;margin-top:8px}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:20px 30px}
.mcard{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:13px 15px}
.mcard .mt{font-weight:650;font-size:14px}.mcard .md{color:var(--ink2);font-size:12.5px;margin-top:3px}
.cando{padding:22px 30px;max-width:960px}
.cando table{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}
.cando td,.cando th{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
.cando .y{color:var(--gd);font-weight:700}.cando .n{color:var(--neg);font-weight:700}
</style></head>
<body>
<div class="hero">
  <h1>Market Event Intelligence <span style="color:var(--ink3);font-weight:400;font-size:14px">— demo</span></h1>
  <p>For any news event, this maps <b>which companies are exposed — named in the article or not</b> — through interpretable
  economic mechanisms, grounded in the source text and validated against realized market co-movement. It is a
  <b>descriptive exposure map</b> (who's connected, through what, how strongly), cross-market validated. It is deliberately
  <b>not</b> a return predictor — direction (up vs down) tests at chance, and we show that honestly.</p>
  <div class="chips">__CHIPS__</div>
</div>
<div class="tabs">
  <div class="tab on" data-v="events">Events &amp; exposure</div>
  <div class="tab" data-v="mech">Economic mechanisms</div>
  <div class="tab" data-v="cando">What it can &amp; can't do</div>
</div>
<div class="view on" id="v-events"><div class="app">
  <aside class="side"><input class="search" id="search" placeholder="Search events…"><div class="list" id="list"></div></aside>
  <main class="main" id="main"></main>
</div></div>
<div class="view" id="v-mech"><div class="mgrid">__MECH__</div></div>
<div class="view" id="v-cando"><div class="cando">
  <h3 style="font-size:15px">The honest boundary — established across 25 tests</h3>
  <p style="color:var(--ink2);margin-top:6px">Every capability was stress-tested. Here is exactly what holds and what doesn't:</p>
  <table><tr><th>Question</th><th>Result</th><th>Evidence</th></tr>
  <tr><td>Who is exposed to an event? (beyond the named firms)</td><td class="y">YES</td><td>descriptive exposure IC +0.10; surfaces unnamed banks in the LIBOR probe (HSBC realized +0.50)</td></tr>
  <tr><td>Through what mechanism, how strongly?</td><td class="y">YES</td><td>return-anchored mechanism factors, magnitude +0.101 (t13); interpretable economic factors</td></tr>
  <tr><td>Does this structure generalise across markets?</td><td class="y">YES</td><td>replicated out-of-sample in India (signed attribution t4.9, persistence +19–25pp)</td></tr>
  <tr><td>Is it more than a sector re-label?</td><td class="y">YES</td><td>survives a 12-ETF sub-industry model (t6.0); event-specific within-industry selection</td></tr>
  <tr><td>Which way will a firm move — up or down?</td><td class="n">NO</td><td>direction ~chance (51%); sentiment best contemporaneous (+0.125) but lead null</td></tr>
  <tr><td>Does the news lead the correlation / forecast returns?</td><td class="n">NO</td><td>lead-lag null; news is coincident, describes not predicts</td></tr>
  </table>
  <p class="note" style="margin-top:14px">Bottom line: a genuine, cross-validated, evidence-grounded <b>exposure and explanation</b> layer — <b>not</b> alpha. The value is knowing <i>who is connected to what, and why</i>, on the day it happens.</p>
</div></div>
<script>
const EV=__DATA__.events, SECTORS=__DATA__.sectors;
const PAL=["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"];
const sc=s=>PAL[Math.max(0,SECTORS.indexOf(s))%PAL.length];
EV.forEach((e,i)=>e._i=i); EV.sort((a,b)=>a.month<b.month?-1:1);
let sel=EV.find(e=>e.label&&e.label.includes("Almunia"))||EV[0];
const el=id=>document.getElementById(id);
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===t));
  document.querySelectorAll(".view").forEach(v=>v.classList.toggle("on",v.id==="v-"+t.dataset.v));});
function renderList(){
  const q=el("search").value.toLowerCase();
  const items=EV.filter(e=>!q||e.label.toLowerCase().includes(q)||e.type.includes(q));
  const byT={}; items.forEach(e=>(byT[e.type]=byT[e.type]||[]).push(e));
  el("list").innerHTML=Object.keys(byT).sort().map(t=>`<div class="grp">${t}</div>`+byT[t].map(e=>
    `<div class="ev ${e===sel?'sel':''}" data-i="${e._i}"><div class="t">${e.label}</div><div class="s">${e.month} · named ${e.n_named}</div></div>`).join("")).join("");
}
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function render(){
  const e=sel, rows=e.rows.filter(r=>r.exp>0), mx=Math.max(...rows.map(r=>r.exp),.1);
  const surf=rows.filter(r=>!r.m).slice(0,5).map(r=>r.t);
  const evi=(e.facts&&e.facts.length)?e.facts.slice(0,2).map(f=>`<div style="margin-bottom:8px"><div class="fq"><q>${esc(f.q)}</q>${f.mech?`<span class="fmech">${esc(f.mech)}</span>`:''}</div>`+
    (f.u?`<a class="fsrc" href="${esc(f.u)}" target="_blank">${esc(f.h)||'(article)'} · <span class="src">${esc(f.s)}</span> · ${f.d||''} ↗</a>`:'')+`</div>`).join(""):'<span class="note">no source quote captured</span>';
  el("main").innerHTML=`<div class="hd">${e.label} <span class="badge">${e.type}</span> <span style="color:var(--ink3);font-size:14px">${e.month}</span></div>
    <div class="card"><h3>1 · what the event is — the source article</h3>${evi}</div>
    <div class="card"><h3>2 · who the article named</h3><div class="named">${e.named.map(t=>`<span class="pill">${t}</span>`).join("")}</div></div>
    <div class="card"><h3>3 · who is actually exposed — the whole market, ranked</h3>
      <div class="leg"><span><span class="dot tn"></span>named</span><span><span class="dot ts"></span>surfaced (unnamed): <b style="color:var(--ink)">${surf.join(', ')}</b></span><span style="color:var(--ink3)">bar = exposure · right = realized co-move</span></div>`+
      rows.slice(0,22).map((r,i)=>{const rl=r.real==null?'—':(r.real>=0?'+':'')+r.real.toFixed(2);const c=r.real==null?'var(--ink3)':(r.real>=0.05?'var(--pos)':(r.real<=-0.05?'var(--neg)':'var(--ink3)'));
        return `<div class="row"><div class="rk">${i+1}</div><div class="nm"><span class="sc" style="background:${sc(r.sec)}"></span><span class="tk">${r.t}</span><span class="tag ${r.m?'n':'s'}">${r.m?'named':'surf'}</span></div>
        <div class="bar"><div class="fill" style="width:${(r.exp/mx*100).toFixed(0)}%;background:${r.m?'var(--pos)':sc(r.sec)}"></div></div><div class="rl">real <b style="color:${c}">${rl}</b></div></div>`}).join("")+
      `<p class="note">The <b>surfaced</b> firms are the point — exposed but never named. A high <b>realized</b> co-move confirms it's real (e.g. HSBC in the LIBOR probe). Magnitude is modelled well; <b>direction</b> (would it rise or fall) is not — see the last tab.</p></div>`;
}
function all(){renderList();render()}
el("search").oninput=renderList;
el("list").addEventListener("click",ev=>{const r=ev.target.closest(".ev");if(!r)return;sel=EV[+r.dataset.i];all();});
all();
</script></body></html>"""

chips = "".join(f'<div class="chip"><span class="v">{v}</span><span class="k">{k}</span></div>' for v, k in STATS)
mech = "".join(f'<div class="mcard"><div class="mt">{t}</div><div class="md">{d}</div></div>' for t, d in MECH)
html = (HTML.replace("__DATA__", json.dumps(D)).replace("__CHIPS__", chips).replace("__MECH__", mech))
out = G / "demo.html"
out.write_text(html)
print(f"-> {out}  ({len(D['events'])} events)")
