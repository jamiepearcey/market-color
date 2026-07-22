# /// script
# requires-python = ">=3.10"
# ///
"""Render events.json -> a self-contained interactive EVENT-LINKAGE timeline.
Each bubble = a named catalyst that binds >=2 firms in the residual (after macro+sector). Height =
sector-orthogonal residual co-movement, size = #firms, colour = catalyst type. Click -> the bound
firms (sector + direction) and a residual-linkage arc diagram. Shows the bursty, event-driven signal."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
D = json.loads((G / "events.json").read_text())

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>News event-linkage timeline</title>
<style>
:root{--surface:#fcfcfb;--panel:#f4f3f0;--panel2:#eceae5;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8983;--grid:#e4e3df;--line:#dcdbd6;--pos:#2a78d6;--neg:#eb6834;color-scheme:light}
[data-theme=dark]{--surface:#151514;--panel:#1e1e1c;--panel2:#262624;--ink:#fff;--ink2:#c3c2b7;--ink3:#86857c;--grid:#2b2b29;--line:#323230;--pos:#3987e5;--neg:#d95926;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--surface);color:var(--ink);font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 26px 70px}
h1{font-size:19px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:13px;margin:3px 0 16px;max-width:820px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
button{font:inherit;color:var(--ink2);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:5px 11px;cursor:pointer}
button:hover{color:var(--ink)}
.legend{display:flex;gap:12px;flex-wrap:wrap}
.lg{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--ink2);cursor:pointer;user-select:none}
.lg.off{opacity:.3}.lg .sw{width:11px;height:11px;border-radius:50%}
.ctrl{display:flex;gap:7px;align-items:center;font-size:12px;color:var(--ink3)}
input[type=range]{accent-color:var(--pos)}
svg{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--grid);stroke-width:1}.axlab{fill:var(--ink3);font-size:11px}
.bub{cursor:pointer;stroke:var(--surface);stroke-width:1}.bub:hover{stroke:var(--ink);stroke-width:1.5}
.bub.sel{stroke:var(--ink);stroke-width:2}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);opacity:0;transition:.08s;z-index:20;max-width:250px}
.tip b{font-weight:600}.tip .m{color:var(--ink3)}
.detail{margin-top:18px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;min-height:80px}
.detail .empty{color:var(--ink3);text-align:center;padding:30px}
.dh{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:3px}
.dh .t{font-size:17px;font-weight:650}.badge{font-size:10.5px;padding:1px 8px;border-radius:20px;background:var(--panel2);color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.dh .mo{color:var(--ink3);font-size:13px}
.dstat{color:var(--ink2);font-size:12.5px;margin-bottom:12px}
.dstat b{color:var(--ink);font-variant-numeric:tabular-nums}
.firms{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.firm{display:flex;align-items:center;gap:7px;background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:6px 10px}
.firm .tk{font-weight:700}.firm .nm{color:var(--ink3);font-size:11.5px;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.firm .sec{font-size:10px;padding:1px 6px;border-radius:20px;color:#fff}.firm .d{font-weight:700}
.spec{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-top:12px}
.spec b{color:var(--ink);font-variant-numeric:tabular-nums}
.cohort-lbl{font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.04em;margin:10px 0 6px}
.cohort{display:flex;gap:5px;flex-wrap:wrap;align-items:center}
.chip2{font-size:11.5px;padding:3px 9px;border-radius:8px;font-weight:600}
.chip2.bound{color:#fff}.chip2.peer{background:transparent;border:1px dashed var(--line);color:var(--ink3);font-weight:500}
.arcwrap{margin-top:8px}
.note{color:var(--ink3);font-size:12px;margin-top:16px;max-width:820px}
</style></head>
<body data-theme="dark"><div class="wrap">
<h1>News event-linkage timeline</h1>
<p class="sub">Each bubble is a <b>named catalyst</b> — a regulation, ruling, strike, or corporate event — that binds
two or more firms together <b>beyond what macro &amp; sector explain</b>. Height = the sector-orthogonal
<b>residual co-movement</b> of the bound firms; size = how many firms; colour = catalyst type. <b>Click a bubble</b>
to open the cluster — the panel shows the specificity: <i>which</i> names the catalyst named out of their whole
sector, and how much more they co-move than the peers it skipped (the gap a sector model can't produce). BBG 2010–2012.</p>
<div class="bar">
  <button id="theme">◐ theme</button>
  <div class="ctrl">min residual <input type="range" id="minres" min="0" max="0.7" step="0.05" value="0.15"><span id="minlbl">0.15</span></div>
  <div class="legend" id="legend"></div>
</div>
<svg id="tl" viewBox="0 0 1140 400"></svg>
<div class="detail" id="detail"><div class="empty">Click an event bubble to see the firms it binds and their residual-linkage.</div></div>
<p class="note">Read: most days are empty — this signal fires <b>only when a specific catalyst binds specific names</b>
(~1/day, bursty). The bound firms are usually a fine sub-sector or national cluster (UK banks, German autos, US
mortgage agencies) that broad sector factors can't group. It's descriptive and small, but it names <i>why</i>
these particular names moved together.</p>
</div>
<div class="tip" id="tip"></div>
<script>
const EV=__DATA__.events, SECTORS=__DATA__.sectors;
const PAL=["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"];
const secColor=s=>PAL[Math.max(0,SECTORS.indexOf(s))%PAL.length];
const TYPES=[...new Set(EV.map(e=>e.type))].sort();
const TPAL=["#3987e5","#d55181","#199e70","#c98500","#9085e9","#d95926","#e66767","#008300"];
const tColor=t=>TPAL[TYPES.indexOf(t)%TPAL.length];
const off=new Set(); let minres=0.15, sel=null;
const MO=[];for(let y=2010;y<=2012;y++)for(let m=1;m<=12;m++)MO.push(`${y}-${String(m).padStart(2,'0')}`);
const NS="http://www.w3.org/2000/svg";
const W=1140,H=400,mL=44,mR=14,mT=14,mB=28,iw=W-mL-mR,ih=H-mT-mB;
const x=i=>mL+iw*(i+0.5)/MO.length, y=v=>mT+ih*(1-v/0.9);
const el=id=>document.getElementById(id);
function jitter(i){let h=0;const s=String(i);for(const c of s)h=(h*31+c.charCodeAt(0))%97;return (h/97-0.5)*14}
function draw(){
  const tl=el("tl");tl.innerHTML="";
  const g=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);tl.appendChild(e);return e};
  for(let v=0;v<=0.9001;v+=0.3){const yy=y(v);g("line",{class:"gridline",x1:mL,x2:mL+iw,y1:yy,y2:yy});
    const t=g("text",{class:"axlab",x:mL-7,y:yy+3,"text-anchor":"end"});t.textContent=v.toFixed(1)}
  let last="";MO.forEach((m,i)=>{const q=m.slice(0,4)+"Q"+(Math.floor((+m.slice(5,7)-1)/3)+1);
    if(q!==last){last=q;const xx=x(i)-iw/MO.length/2;g("line",{class:"gridline",x1:xx,x2:xx,y1:mT,y2:mT+ih});
      const t=g("text",{class:"axlab",x:xx+2,y:H-10});t.textContent=q}});
  const g0=g("text",{class:"axlab",x:mL-30,y:mT+8,"transform":`rotate(-90 ${mL-30} ${mT+ih/2})`});g0.textContent="residual co-movement";
  EV.forEach((e,i)=>{if(off.has(e.type)||e.res<minres)return;
    const c=g("circle",{class:"bub"+(sel===e?" sel":""),cx:x(MO.indexOf(e.month))+jitter(i),cy:y(e.res),
      r:3+Math.sqrt(e.size)*2.4,fill:tColor(e.type),"fill-opacity":.8,"data-i":i});
  });
}
const tip=el("tip");
el("tl").addEventListener("mousemove",ev=>{const t=ev.target;if(!t.dataset||t.dataset.i===undefined){tip.style.opacity=0;return}
  const e=EV[+t.dataset.i];tip.innerHTML=`<b>${e.label}</b> <span class="m">${e.type}</span><br><span class="m">${e.month} · ${e.size} firms · residual ${e.res>=0?'+':''}${e.res.toFixed(2)}</span>`;
  tip.style.opacity=1;tip.style.left=Math.min(ev.clientX+14,innerWidth-260)+"px";tip.style.top=(ev.clientY+14)+"px"});
el("tl").addEventListener("mouseleave",()=>tip.style.opacity=0);
el("tl").addEventListener("click",ev=>{const t=ev.target;if(!t.dataset||t.dataset.i===undefined)return;sel=EV[+t.dataset.i];draw();detail(sel)});
function detail(e){
  const arcW=Math.min(820,120+e.names.length*64),arcH=150;
  const px=i=>34+(arcW-68)*(e.names.length<2?0.5:i/(e.names.length-1));
  let arcs="";const base=arcH-22;
  // strongest links first (declutter big clusters), height-capped bezier arcs
  e.pairs.slice().sort((p,q)=>Math.abs(q[2])-Math.abs(p[2])).forEach(([a,b,rc])=>{
    if(Math.abs(rc)<0.08)return;const x1=px(a),x2=px(b),mx=(x1+x2)/2;
    const peak=Math.min(base-6,Math.abs(x2-x1)*0.5+8);
    arcs+=`<path d="M ${x1} ${base} Q ${mx} ${base-peak} ${x2} ${base}" fill="none" stroke="${rc>=0?'var(--pos)':'var(--neg)'}" stroke-width="${0.6+Math.abs(rc)*5}" stroke-opacity="${0.22+Math.abs(rc)*0.6}"/>`});
  let nodes="";e.names.forEach((n,i)=>{nodes+=`<circle cx="${px(i)}" cy="${base}" r="4.5" fill="${secColor(n.sec)}"/><text x="${px(i)}" y="${base+16}" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--ink)">${n.t}</text>`});
  const firms=e.names.map(n=>`<div class="firm"><span class="d" style="color:${n.dir>=0?'var(--pos)':'var(--neg)'}">${n.dir>=0?'▲':'▼'}</span><span class="tk">${n.t}</span><span class="nm">${n.n}</span><span class="sec" style="background:${secColor(n.sec)}">${n.sec}</span></div>`).join("");
  const total=e.size+(e.cohort?e.cohort.length:0);
  const rsub=e.res_sub==null?null:e.res_sub, cb=e.cohort_base==null?null:e.cohort_base;
  const specificity=(rsub!=null&&cb!=null)?
    `<div class="spec">This catalyst named <b>${e.size}</b> of the <b>${total}</b> ${e.dom} names in view — and those <b>${e.size}</b> co-move <b>${rsub>=0?'+':''}${rsub.toFixed(2)}</b> together <i>beyond even sub-industry</i>, versus <b>${cb>=0?'+':''}${cb.toFixed(2)}</b> for the ${e.cohort.length} peers it did <b>not</b> name. <span style="color:var(--ink3)">That gap is the specificity — a sector model can't pick these ${e.size} out of the ${total}.</span></div>`:"";
  const bound=e.names.map(n=>`<span class="chip2 bound" style="background:${secColor(n.sec)}">${n.t}</span>`).join("");
  const peers=(e.cohort||[]).map(n=>`<span class="chip2 peer">${n.t}</span>`).join("");
  const cohort=(e.cohort&&e.cohort.length)?`<div class="cohort-lbl">the ${e.dom} cohort — <span style="color:var(--pos)">named by this catalyst</span> vs <span style="color:var(--ink3)">skipped</span>:</div><div class="cohort">${bound}${peers}</div>`:"";
  el("detail").innerHTML=`<div class="dh"><span class="t">${e.label}</span><span class="badge">${e.type}</span><span class="mo">${e.month}</span></div>
    <div class="dstat">binds <b>${e.size}</b> firms · residual co-movement <b>${e.res>=0?'+':''}${e.res.toFixed(2)}</b> (peak <b>${e.resmax>=0?'+':''}${e.resmax.toFixed(2)}</b>) after macro &amp; broad sector</div>
    ${specificity}
    <div class="arcwrap"><svg viewBox="0 0 ${arcW} ${arcH}" style="max-width:${arcW}px">${arcs}${nodes}</svg></div>
    <div class="firms">${firms}</div>
    ${cohort}`;
}
const leg=el("legend");
TYPES.forEach(t=>{const d=document.createElement("div");d.className="lg";d.innerHTML=`<span class="sw" style="background:${tColor(t)}"></span>${t}`;
  d.onclick=()=>{off.has(t)?off.delete(t):off.add(t);d.classList.toggle("off");draw()};leg.appendChild(d)});
el("theme").onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==="dark"?"light":"dark";draw();if(sel)detail(sel)};
el("minres").oninput=e=>{minres=+e.target.value;el("minlbl").textContent=minres.toFixed(2);draw()};
draw();
</script></body></html>"""
out = G / "events.html"
out.write_text(HTML.replace("__DATA__", json.dumps(D)))
print(f"-> {out}  ({len(D['events'])} events)")
