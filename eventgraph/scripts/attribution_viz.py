# /// script
# requires-python = ">=3.10"
# ///
"""Render attribution_daily.json -> a self-contained interactive HTML monitor.
Multi-line chart (driver attribution over time), validated colorblind-safe palette,
crosshair+tooltip, legend + direct end-labels, light/dark toggle, data-table view."""
import json
from pathlib import Path

G = Path("../data/eg_runs/eg100k_graph")
data = json.loads((G / "attribution_daily.json").read_text())
# fixed categorical slot order (dataviz validated palette): light / dark hex per slot
PAL = [("#2a78d6", "#3987e5"), ("#008300", "#008300"), ("#e87ba4", "#d55181"),
       ("#eda100", "#c98500"), ("#1baf7a", "#199e70"), ("#eb6834", "#d95926")]
for i, d in enumerate(data["drivers"]):
    d["cLight"], d["cDark"] = PAL[i % len(PAL)]

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver co-movement attribution — daily monitor</title>
<style>
  :root{--surface:#fcfcfb;--panel:#f4f3f0;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--zero:#b9b8b3;color-scheme:light}
  [data-theme=dark]{--surface:#1a1a19;--panel:#232321;--ink:#fff;--ink2:#c3c2b7;--grid:#333331;--zero:#4a4a47;color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:28px 24px 60px}
  h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--ink2);font-size:13px;margin:0 0 20px;max-width:760px}
  .bar{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
  button{font:inherit;color:var(--ink2);background:var(--panel);border:1px solid var(--grid);border-radius:7px;padding:5px 11px;cursor:pointer}
  button:hover{color:var(--ink)}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:2px 0 10px}
  .lg{display:flex;gap:7px;align-items:center;font-size:12.5px;color:var(--ink2);cursor:pointer;user-select:none}
  .lg.off{opacity:.32}
  .sw{width:11px;height:11px;border-radius:3px;flex:none}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .gridline{stroke:var(--grid);stroke-width:1}
  .zeroline{stroke:var(--zero);stroke-width:1}
  .axlab{fill:var(--ink2);font-size:11px}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .endlab{font-size:11.5px;font-weight:600}
  .cross{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:0}
  .tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:8px;
       padding:9px 11px;font-size:12px;box-shadow:0 6px 22px rgba(0,0,0,.22);opacity:0;transition:opacity .08s;z-index:9;min-width:190px}
  .tip .dt{color:var(--ink2);margin-bottom:5px;font-variant-numeric:tabular-nums}
  .tip .row{display:flex;justify-content:space-between;gap:14px;align-items:center}
  .tip .row b{font-variant-numeric:tabular-nums}
  .tip .nm{display:flex;gap:6px;align-items:center;color:var(--ink2)}
  table{border-collapse:collapse;font-size:12px;margin-top:16px;width:100%;display:none;font-variant-numeric:tabular-nums}
  table.show{display:table}
  th,td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--ink2);font-weight:600}
  .note{color:var(--ink2);font-size:12px;margin-top:18px;max-width:760px}
</style></head>
<body data-theme="dark"><div class="wrap">
<h1>Driver co-movement attribution &mdash; daily monitor</h1>
<p class="sub">Each line: how much a news-graph <b>driver</b> explains its connected pairs' co-movement,
measured every trading day in a trailing 63-day window (leave-two-out signed residualization).
High = a <b>specific</b> channel tightly bundling its names; near-zero = a broad market-wide driver
whose effect is already in the baseline. Historical BBG corpus, 2010&ndash;2012.</p>
<div class="bar">
  <button id="theme">Toggle light/dark</button>
  <button id="tbl">Show data table</button>
</div>
<div class="legend" id="legend"></div>
<svg id="chart" viewBox="0 0 1000 520" role="img" aria-label="Driver attribution over time"></svg>
<div class="tip" id="tip"></div>
<table id="table"></table>
<p class="note">Attribution is <b>descriptive</b> (it decomposes correlation that exists), not a return
forecast. The measured drop is trusted for its out-of-sample stability (r&nbsp;=&nbsp;+0.75 quarter-to-quarter),
not the driver's headline importance &mdash; a broad macro hub can be economically huge yet score low here
because it moves everything.</p>
</div>
<script>
const DATA = __DATA__;
const D = DATA.drivers, DATES = DATES_JSON;
const NS="http://www.w3.org/2000/svg", svg=document.getElementById("chart");
const W=1000,H=520,mL=48,mR=150,mT=14,mB=34, iw=W-mL-mR, ih=H-mT-mB;
const off=new Set();
let ymax=0; D.forEach(d=>d.series.forEach(p=>{if(p.attr!=null&&p.attr>ymax)ymax=p.attr}));
ymax=Math.ceil(ymax/0.05)*0.05;
const x=i=>mL+iw*i/(DATES.length-1), y=v=>mT+ih*(1-v/ymax);
const cvar=d=>document.body.dataset.theme==="dark"?d.cDark:d.cLight;
function draw(){
  svg.innerHTML="";
  const g=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  // y grid + labels
  for(let v=0;v<=ymax+1e-9;v+=0.05){const yy=y(v);
    g(v===0?"line":"line",{class:v===0?"zeroline":"gridline",x1:mL,x2:mL+iw,y1:yy,y2:yy});
    const t=g("text",{class:"axlab",x:mL-8,y:yy+3,"text-anchor":"end"});t.textContent=v.toFixed(2)}
  // x ticks: first trading day of each quarter
  let last="";DATES.forEach((d,i)=>{const q=d.slice(0,4)+"Q"+(Math.floor((+d.slice(5,7)-1)/3)+1);
    if(q!==last){last=q;const xx=x(i);g("line",{class:"gridline",x1:xx,x2:xx,y1:mT,y2:mT+ih});
      const t=g("text",{class:"axlab",x:xx,y:H-12,"text-anchor":"middle"});t.textContent=q}});
  // lines (skip null gaps)
  const ends=[];
  D.forEach(d=>{if(off.has(d.id))return;const col=cvar(d);let path="",pen=false;
    d.series.forEach((p,i)=>{if(p.attr==null){pen=false;return}path+=(pen?"L":"M")+x(i).toFixed(1)+" "+y(p.attr).toFixed(1)+" ";pen=true});
    g("path",{class:"ln",d:path,stroke:col});
    for(let i=d.series.length-1;i>=0;i--){if(d.series[i].attr!=null){ends.push({y:y(d.series[i].attr),col,label:d.label});break}}});
  // end labels: de-collide by pushing apart to a 14px min gap, then leader-align
  ends.sort((a,b)=>a.y-b.y);
  for(let i=1;i<ends.length;i++){if(ends[i].y-ends[i-1].y<14)ends[i].y=ends[i-1].y+14}
  ends.forEach(e=>{const t=g("text",{class:"endlab",x:mL+iw+8,y:e.y+4,fill:e.col});
    t.textContent=e.label.length>18?e.label.slice(0,17)+"…":e.label});
  g("line",{class:"cross",id:"cross",x1:0,x2:0,y1:mT,y2:mT+ih});
}
// legend
const leg=document.getElementById("legend");
D.forEach(d=>{const el=document.createElement("div");el.className="lg";el.dataset.id=d.id;
  el.innerHTML=`<span class="sw" style="background:${cvar(d)}"></span>${d.label} <span style="opacity:.6">(${d.n_names})</span>`;
  el.onclick=()=>{off.has(d.id)?off.delete(d.id):off.add(d.id);el.classList.toggle("off");draw()};leg.appendChild(el)});
function paintLegend(){[...leg.children].forEach(el=>{const d=D.find(x=>x.id===el.dataset.id);el.querySelector(".sw").style.background=cvar(d)})}
// hover
const tip=document.getElementById("tip");
svg.addEventListener("mousemove",e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
  let i=Math.round((px-mL)/iw*(DATES.length-1));i=Math.max(0,Math.min(DATES.length-1,i));
  const cr=document.getElementById("cross");if(cr){cr.setAttribute("x1",x(i));cr.setAttribute("x2",x(i));cr.style.opacity=1}
  let rows="";D.forEach(d=>{if(off.has(d.id))return;const p=d.series[i];
    rows+=`<div class="row"><span class="nm"><span class="sw" style="background:${cvar(d)}"></span>${d.label}</span><b>${p.attr==null?"—":p.attr>=0?"+"+p.attr.toFixed(3):p.attr.toFixed(3)}</b></div>`});
  tip.innerHTML=`<div class="dt">${DATES[i]}</div>${rows}`;tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+16,innerWidth-tip.offsetWidth-10)+"px";tip.style.top=(e.clientY+14)+"px"});
svg.addEventListener("mouseleave",()=>{tip.style.opacity=0;const cr=document.getElementById("cross");if(cr)cr.style.opacity=0});
// controls
document.getElementById("theme").onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==="dark"?"light":"dark";draw();paintLegend()};
const table=document.getElementById("table");
document.getElementById("tbl").onclick=()=>{if(!table.classList.contains("show")){
    let h="<tr><th>Date</th>"+D.map(d=>`<th>${d.label}</th>`).join("")+"</tr>";
    DATES.forEach((dt,i)=>{h+=`<tr><td>${dt}</td>`+D.map(d=>{const a=d.series[i].attr;return `<td>${a==null?"":a.toFixed(3)}</td>`}).join("")+"</tr>"});
    table.innerHTML=h}
  table.classList.toggle("show");
  document.getElementById("tbl").textContent=table.classList.contains("show")?"Hide data table":"Show data table"};
draw();
</script></body></html>"""

html = (TEMPLATE
        .replace("__DATA__", json.dumps({"drivers": data["drivers"]}))
        .replace("DATES_JSON", json.dumps(data["dates"])))
out = G / "attribution_daily.html"
out.write_text(html)
print(f"-> {out}  ({len(data['dates'])} days, {len(data['drivers'])} drivers)")
