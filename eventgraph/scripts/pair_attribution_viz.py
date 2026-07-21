# /// script
# requires-python = ">=3.10"
# ///
"""Render pair_attribution.json -> small-multiples HTML.
One panel per pair: raw rolling correlation vs the counterfactual with the driver's
channel removed; the shaded gap = attribution. Crosshair+tooltip, light/dark, table."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
data = json.loads((G / "pair_attribution.json").read_text())

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pair correlation attribution — __DRIVER__</title>
<style>
  :root{--surface:#fcfcfb;--panel:#f4f3f0;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--zero:#b9b8b3;
        --raw:#2a78d6;--cf:#eb6834;color-scheme:light}
  [data-theme=dark]{--surface:#1a1a19;--panel:#232321;--ink:#fff;--ink2:#c3c2b7;--grid:#333331;--zero:#4a4a47;
        --raw:#3987e5;--cf:#d95926;color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:28px 24px 60px}
  h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--ink2);font-size:13px;margin:0 0 16px;max-width:760px}
  .bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
  button{font:inherit;color:var(--ink2);background:var(--panel);border:1px solid var(--grid);border-radius:7px;padding:5px 11px;cursor:pointer}
  button:hover{color:var(--ink)}
  .legend{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 6px;font-size:12.5px;color:var(--ink2)}
  .lg{display:flex;gap:7px;align-items:center}
  .sw{width:16px;height:0;border-top:2px solid;flex:none}
  .sw.dash{border-top-style:dashed}
  .sw.fill{height:11px;border:none;border-radius:3px;opacity:.9}
  .panel{margin:10px 0 4px}
  .ttl{font-size:14px;font-weight:600;margin:14px 0 2px}
  .ttl small{font-weight:400;color:var(--ink2)}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .gridline{stroke:var(--grid);stroke-width:1}.zeroline{stroke:var(--zero);stroke-width:1}
  .axlab{fill:var(--ink2);font-size:11px}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .cross{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:0}
  .tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:8px;
       padding:9px 11px;font-size:12px;box-shadow:0 6px 22px rgba(0,0,0,.22);opacity:0;transition:opacity .08s;z-index:9;min-width:180px}
  .tip .dt{color:var(--ink2);margin-bottom:5px;font-variant-numeric:tabular-nums}
  .tip .row{display:flex;justify-content:space-between;gap:14px}.tip b{font-variant-numeric:tabular-nums}
  .note{color:var(--ink2);font-size:12px;margin-top:16px;max-width:760px}
</style></head>
<body data-theme="dark"><div class="wrap">
<h1>Pair correlation &amp; the &ldquo;__DRIVER__ removed&rdquo; counterfactual</h1>
<p class="sub">For each pair: the <b>raw</b> rolling 63-day correlation, and what the correlation
would be with the <b>__DRIVER__</b> news channel residualized out (leave-two-out signed factor).
The shaded gap is the attribution &mdash; how much of these two names' co-movement runs through
that specific channel, day by day. BBG corpus, 2010&ndash;2012.</p>
<div class="bar"><button id="theme">Toggle light/dark</button><button id="tbl">Show data table</button></div>
<div class="legend">
  <span class="lg"><span class="sw" style="border-color:var(--raw)"></span>raw correlation</span>
  <span class="lg"><span class="sw dash" style="border-color:var(--cf)"></span>__DRIVER__ removed</span>
  <span class="lg"><span class="sw fill" style="background:var(--raw)"></span>attribution (gap)</span>
</div>
<div id="panels"></div>
<div class="tip" id="tip"></div>
<table id="table" style="border-collapse:collapse;font-size:12px;margin-top:14px;width:100%;display:none;font-variant-numeric:tabular-nums"></table>
<p class="note">Read: for GS&ndash;MS a persistent ~0.13 slice of a ~0.70 correlation runs through the
Moody's channel &mdash; it widens through the 2011 downgrade wave and narrows after. This is
<b>descriptive</b> (it decomposes existing correlation), trusted for its out-of-sample stability
(r&nbsp;=&nbsp;+0.75), not as a return forecast. The level carries a sector component &mdash; but
<b>~70% of it survives</b> residualizing against a 9 GICS-sector factor model (t&nbsp;=&nbsp;23),
so most of this gap is genuine news-specific covariance beyond macro and sector. The <i>timing</i>
is driven by the news flow.</p>
</div>
<script>
const DATA=__DATA__, DATES=__DATES__, DRIVER="__DRIVER__";
const NS="http://www.w3.org/2000/svg";
const W=1000,H=210,mL=42,mR=16,mT=10,mB=26,iw=W-mL-mR,ih=H-mT-mB;
const ymin=-0.1,ymax=1.0;
const x=i=>mL+iw*i/(DATES.length-1), y=v=>mT+ih*(1-(v-ymin)/(ymax-ymin));
const cv=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
const panels=document.getElementById("panels"), svgs=[];
DATA.forEach(p=>{const d=document.createElement("div");d.className="panel";
  const share=(()=>{const g=p.series.filter(s=>s.c0!=null&&s.c1!=null);const raw=g.reduce((a,s)=>a+s.c0,0)/g.length;
    const at=g.reduce((a,s)=>a+(s.c0-s.c1),0)/g.length;return {raw,at,pct:Math.round(at/raw*100)}})();
  d.innerHTML=`<div class="ttl">${p.label} <small>&mdash; raw corr ${share.raw.toFixed(2)}, ${DRIVER} channel +${share.at.toFixed(2)} (${share.pct}%)</small></div>`;
  const svg=document.createElementNS(NS,"svg");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.dataset.pi=svgs.length;
  d.appendChild(svg);panels.appendChild(d);svgs.push(svg)});
function drawAll(){svgs.forEach(svg=>{const p=DATA[+svg.dataset.pi];svg.innerHTML="";
  const g=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
  for(let v=0;v<=1.001;v+=0.25){const yy=y(v);g("line",{class:v===0?"zeroline":"gridline",x1:mL,x2:mL+iw,y1:yy,y2:yy});
    const t=g("text",{class:"axlab",x:mL-7,y:yy+3,"text-anchor":"end"});t.textContent=v.toFixed(2)}
  let last="";DATES.forEach((dt,i)=>{const q=dt.slice(0,4)+"Q"+(Math.floor((+dt.slice(5,7)-1)/3)+1);
    if(q!==last){last=q;const xx=x(i);g("line",{class:"gridline",x1:xx,x2:xx,y1:mT,y2:mT+ih});
      const t=g("text",{class:"axlab",x:xx,y:H-10,"text-anchor":"middle"});t.textContent=q}});
  // shaded gap (raw above cf)
  let up="",dn="";const pts=[];p.series.forEach((s,i)=>{if(s.c0!=null&&s.c1!=null)pts.push([i,s.c0,s.c1])});
  if(pts.length){up=pts.map(q=>x(q[0]).toFixed(1)+" "+y(q[1]).toFixed(1)).join(" L ");
    dn=pts.slice().reverse().map(q=>x(q[0]).toFixed(1)+" "+y(q[2]).toFixed(1)).join(" L ");
    g("path",{d:`M ${up} L ${dn} Z`,fill:cv("--raw"),opacity:.14,stroke:"none"})}
  const line=(key,col,dash)=>{let path="",pen=false;p.series.forEach((s,i)=>{const v=s[key];
    if(v==null){pen=false;return}path+=(pen?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1)+" ";pen=true});
    g("path",{class:"ln",d:path,stroke:col,...(dash?{"stroke-dasharray":"5 4"}:{})})};
  line("c1",cv("--cf"),true);line("c0",cv("--raw"),false);
  g("line",{class:"cross",x1:0,x2:0,y1:mT,y2:mT+ih});})}
const tip=document.getElementById("tip");
function hover(e,svg){const p=DATA[+svg.dataset.pi];const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
  let i=Math.round((px-mL)/iw*(DATES.length-1));i=Math.max(0,Math.min(DATES.length-1,i));
  svgs.forEach(s=>{const c=s.querySelector(".cross");if(c){c.setAttribute("x1",x(i));c.setAttribute("x2",x(i));c.style.opacity=1}});
  const s=p.series[i];const at=(s.c0!=null&&s.c1!=null)?(s.c0-s.c1):null;
  tip.innerHTML=`<div class="dt">${p.label} &nbsp;·&nbsp; ${DATES[i]}</div>`+
    `<div class="row"><span>raw corr</span><b>${s.c0==null?"—":s.c0.toFixed(3)}</b></div>`+
    `<div class="row"><span>${DRIVER} removed</span><b>${s.c1==null?"—":s.c1.toFixed(3)}</b></div>`+
    `<div class="row"><span>attribution</span><b>${at==null?"—":"+"+at.toFixed(3)}</b></div>`;
  tip.style.opacity=1;tip.style.left=Math.min(e.clientX+16,innerWidth-tip.offsetWidth-10)+"px";tip.style.top=(e.clientY+14)+"px"}
svgs.forEach(svg=>{svg.addEventListener("mousemove",e=>hover(e,svg));
  svg.addEventListener("mouseleave",()=>{tip.style.opacity=0;svgs.forEach(s=>{const c=s.querySelector(".cross");if(c)c.style.opacity=0})})});
document.getElementById("theme").onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==="dark"?"light":"dark";drawAll()};
const table=document.getElementById("table");
document.getElementById("tbl").onclick=()=>{if(!table.classList.contains("show")){
  let h="<tr><th style='text-align:left;border-bottom:1px solid var(--grid);padding:4px 8px'>Date</th>"+
    DATA.map(p=>`<th colspan=3 style='border-bottom:1px solid var(--grid);padding:4px 8px'>${p.label}</th>`).join("")+"</tr>";
  DATES.forEach((dt,i)=>{h+=`<tr><td style='padding:3px 8px'>${dt}</td>`+DATA.map(p=>{const s=p.series[i];
    const at=(s.c0!=null&&s.c1!=null)?(s.c0-s.c1).toFixed(3):"";
    return `<td style='text-align:right;padding:3px 8px'>${s.c0==null?"":s.c0.toFixed(3)}</td><td style='text-align:right;padding:3px 8px'>${s.c1==null?"":s.c1.toFixed(3)}</td><td style='text-align:right;padding:3px 8px;color:var(--ink2)'>${at}</td>`}).join("")+"</tr>"});
  table.innerHTML=h}
  table.classList.toggle("show");table.style.display=table.classList.contains("show")?"table":"none";
  document.getElementById("tbl").textContent=table.classList.contains("show")?"Hide data table":"Show data table"};
drawAll();
</script></body></html>"""

driver_label = data["driver"]
html = (TEMPLATE
        .replace("__DATA__", json.dumps(data["pairs"]))
        .replace("__DATES__", json.dumps(data["dates"]))
        .replace("__DRIVER__", driver_label))
out = G / "pair_attribution.html"
out.write_text(html)
print(f"-> {out}  ({len(data['pairs'])} pairs, {len(data['dates'])} days)")
