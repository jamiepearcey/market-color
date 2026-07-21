# /// script
# requires-python = ">=3.10"
# ///
"""Render explorer.json -> a self-contained interactive correlation-attribution EXPLORER.
Pick an instrument, see the others ranked by correlation (with sparklines), drill into a
pair, and see the news drivers that explain their co-movement over time — with an auto
narrative. Vanilla JS + SVG, validated colorblind-safe palette, light/dark."""
import json
from pathlib import Path
G = Path("../data/eg_runs/eg100k_graph")
DATA = json.loads((G / "explorer.json").read_text())

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>News-driver correlation explorer</title>
<style>
:root{--surface:#fcfcfb;--panel:#f4f3f0;--panel2:#eceae5;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8983;
      --grid:#e4e3df;--line:#dcdbd6;--zero:#b9b8b3;--raw:#2a78d6;--cf:#eb6834;--accent:#2a78d6;color-scheme:light}
[data-theme=dark]{--surface:#151514;--panel:#1e1e1c;--panel2:#262624;--ink:#fff;--ink2:#c3c2b7;--ink3:#86857c;
      --grid:#2e2e2b;--line:#323230;--zero:#4a4a47;--raw:#3987e5;--cf:#d95926;--accent:#3987e5;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--surface);color:var(--ink);font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:340px 1fr;height:100vh}
.side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
.brand{padding:16px 18px 12px;border-bottom:1px solid var(--line)}
.brand h1{font-size:15px;letter-spacing:-.01em}
.brand p{font-size:11.5px;color:var(--ink3);margin-top:2px}
.pick{padding:12px 14px;border-bottom:1px solid var(--line)}
.pickhdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.pickhdr .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3)}
.chg{font-size:11.5px;color:var(--accent);cursor:pointer;background:none;border:none}
.aChip{display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px}
.search{width:100%;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;font-size:13px}
.modes{display:flex;gap:4px;padding:8px 14px 4px}
.modes button{flex:1;font:inherit;font-size:11px;padding:4px 2px;border:1px solid var(--line);background:var(--surface);color:var(--ink2);border-radius:6px;cursor:pointer}
.modes button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.list{flex:1;overflow-y:auto;padding:4px 8px 20px}
.row{display:grid;grid-template-columns:1fr auto;gap:6px 10px;align-items:center;padding:7px 9px;border-radius:9px;cursor:pointer}
.row:hover{background:var(--panel2)}
.row.sel{background:var(--accent);color:#fff}
.row.sel .sub,.row.sel .val{color:rgba(255,255,255,.85)}
.row .nm{font-weight:600;font-size:13px;display:flex;align-items:center;gap:6px;min-width:0}
.row .nm .tk{opacity:.6;font-weight:500}
.row .nm b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .sub{font-size:11px;color:var(--ink3);grid-column:1;display:flex;align-items:center;gap:6px}
.row .val{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.row .spark{grid-column:2;grid-row:2}
.dot{width:8px;height:8px;border-radius:2px;flex:none}
.chip{font-size:10px;padding:1px 6px;border-radius:20px;background:var(--panel2);color:var(--ink2);white-space:nowrap}
.main{overflow-y:auto;padding:22px 26px 60px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.title{font-size:20px;font-weight:650;letter-spacing:-.01em;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.title .x{color:var(--ink3);font-weight:400}
.tools{display:flex;gap:7px}
.tools button{font:inherit;font-size:12px;color:var(--ink2);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:5px 11px;cursor:pointer}
.tools button:hover{color:var(--ink)}
.narr{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:16px 0;font-size:14px;line-height:1.6}
.narr b{color:var(--ink)} .narr .hi{color:var(--accent);font-weight:600}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.card h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3);margin-bottom:4px;font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:6px 0 4px}
.legend .lg{display:flex;gap:7px;align-items:center}
.sw{width:16px;height:0;border-top:2px solid;flex:none}.sw.dash{border-top-style:dashed}.sw.fill{height:11px;border:none;border-radius:3px;opacity:.9}
svg{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--grid);stroke-width:1}.zeroline{stroke:var(--zero);stroke-width:1}
.axlab{fill:var(--ink3);font-size:11px}
.ln{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.cross{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:0}
.drv{display:grid;grid-template-columns:auto 1fr auto auto;gap:10px;align-items:center;padding:9px 11px;border-radius:9px;cursor:pointer;border:1px solid transparent}
.drv:hover{background:var(--panel2)}.drv.on{border-color:var(--cf);background:var(--panel2)}
.drv .dl{font-weight:600;font-size:13px}
.drv .tag{font-size:10.5px;padding:1px 7px;border-radius:20px;font-weight:600}
.drv .co{background:rgba(41,120,214,.16);color:var(--raw)}.drv .dv{background:rgba(235,104,52,.18);color:var(--cf)}
.drv .mn{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px}
.empty{color:var(--ink3);text-align:center;padding:70px 20px;font-size:14px}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:9px;
     padding:9px 11px;font-size:12px;box-shadow:0 8px 26px rgba(0,0,0,.25);opacity:0;transition:opacity .08s;z-index:20;min-width:190px}
.tip .dt{color:var(--ink3);margin-bottom:5px;font-variant-numeric:tabular-nums}
.tip .r{display:flex;justify-content:space-between;gap:14px}.tip b{font-variant-numeric:tabular-nums}
.hint{font-size:12px;color:var(--ink3);margin-top:3px}
::-webkit-scrollbar{width:9px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px}
</style></head>
<body data-theme="dark">
<div class="app">
  <aside class="side">
    <div class="brand"><h1>News-driver correlation explorer</h1>
      <p>Which assets move together — and which news channel is holding them there. BBG 2010–2012.</p></div>
    <div class="pick">
      <div class="pickhdr"><span class="lbl" id="pickLbl">Pick an instrument</span>
        <button class="chg" id="chg" style="display:none">← change</button></div>
      <div id="aChip"></div>
      <input class="search" id="search" placeholder="Search 64 instruments…">
    </div>
    <div class="modes" id="modes" style="display:none">
      <button data-m="mean" class="on">avg corr</button>
      <button data-m="peak">peak</button>
      <button data-m="now">latest</button>
      <button data-m="drivers">shared drivers</button>
    </div>
    <div class="list" id="list"></div>
  </aside>
  <main class="main" id="mainpane"></main>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA=__DATA__;
const INSTR=DATA.instruments, DATES=DATA.dates, CORR=DATA.corr, ATTR=DATA.attr, WIN=DATA.window;
const IM={};INSTR.forEach(x=>IM[x.sym]=x);
const SECPAL=["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"];
const SECS=[...new Set(INSTR.map(x=>x.sector))].sort();
const secColor=s=>{const i=SECS.indexOf(s);return i<0?"#888":SECPAL[i%SECPAL.length]};
const pk=(a,b)=>[a,b].sort().join("|");
const getCorr=(a,b)=>CORR[pk(a,b)];
const getAttr=(a,b)=>ATTR[pk(a,b)];
let A=null,B=null,DRV=null,MODE="mean";
const el=id=>document.getElementById(id);
const fmt=v=>v==null?"—":(v>=0?"+":"")+v.toFixed(2);
const mean=a=>{const v=a.filter(x=>x!=null);return v.length?v.reduce((x,y)=>x+y,0)/v.length:null};
const peak=a=>{let m=null;a.forEach(x=>{if(x!=null&&(m==null||x>m))m=x});return m};
const nowv=a=>{for(let i=a.length-1;i>=0;i--)if(a[i]!=null)return a[i];return null};

function sparkline(ser,w,h,col,fillZero){
  const v=ser.map(x=>x==null?null:x);const lo=-0.2,hi=1;
  const X=i=>i/(ser.length-1)*w, Y=t=>h-(t-lo)/(hi-lo)*h;
  let d="",pen=false;v.forEach((t,i)=>{if(t==null){pen=false;return}d+=(pen?"L":"M")+X(i).toFixed(1)+" "+Y(t).toFixed(1)+" ";pen=true});
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><line x1="0" x2="${w}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--zero)" stroke-width="1" opacity=".5"/><path d="${d}" fill="none" stroke="${col}" stroke-width="1.5"/></svg>`;
}
function secDot(s){return `<span class="dot" style="background:${secColor(s)}"></span>`}

function renderList(){
  const q=el("search").value.trim().toLowerCase();
  const list=el("list");
  if(!A){ // pick A
    el("pickLbl").textContent="Pick an instrument";el("modes").style.display="none";el("chg").style.display="none";el("aChip").innerHTML="";
    const items=INSTR.filter(x=>!q||x.name.toLowerCase().includes(q)||x.sym.toLowerCase().includes(q)||x.sector.toLowerCase().includes(q))
      .sort((a,b)=>b.freq-a.freq);
    list.innerHTML=items.map(x=>`<div class="row" data-sym="${x.sym}">
      <div class="nm"><b>${x.name}</b> <span class="tk">${x.sym}</span></div>
      <div class="val" style="font-weight:500;color:var(--ink3);font-size:11px">${x.freq}</div>
      <div class="sub">${secDot(x.sector)}${x.sector} · ${x.drivers.length} drivers</div></div>`).join("");
    return;
  }
  // A selected -> rank others
  el("pickLbl").textContent="Correlated with";el("modes").style.display="flex";el("chg").style.display="inline";
  el("aChip").innerHTML=`<div class="aChip">${secDot(A.sector)}${A.name} <span class="tk" style="opacity:.6">${A.sym}</span></div>`;
  let others=INSTR.filter(x=>x.sym!==A.sym).map(x=>{
    const c=getCorr(A.sym,x.sym);const sh=A.drivers.filter(d=>x.drivers.some(e=>e.id===d.id)).length;
    return {x,ser:c,mean:c?mean(c):null,peak:c?peak(c):null,now:c?nowv(c):null,shared:sh};
  }).filter(o=>o.ser);
  const key=MODE==="drivers"?"shared":MODE;
  others.sort((a,b)=>(b[key]??-9)-(a[key]??-9));
  if(q)others=others.filter(o=>o.x.name.toLowerCase().includes(q)||o.x.sym.toLowerCase().includes(q)||o.x.sector.toLowerCase().includes(q));
  list.innerHTML=others.map(o=>{const val=MODE==="drivers"?o.shared:o[key];
    return `<div class="row ${B&&o.x.sym===B.sym?'sel':''}" data-sym="${o.x.sym}">
      <div class="nm"><b>${o.x.name}</b> <span class="tk">${o.x.sym}</span></div>
      <div class="val">${MODE==="drivers"?val:fmt(val)}</div>
      <div class="sub">${secDot(o.x.sector)}${o.x.sector}${o.shared?` · ${o.shared} shared`:''}</div>
      <div class="spark">${sparkline(o.ser,84,20,secColor(o.x.sector))}</div></div>`}).join("");
}

function narrative(){
  const c=getCorr(A.sym,B.sym);const mc=mean(c),pc=peak(c);
  let pi=0;c.forEach((v,i)=>{if(v!=null&&v===pc)pi=i});
  const at=getAttr(A.sym,B.sym);
  let s=`<b>${A.name}</b> and <b>${B.name}</b> co-move at an average correlation of <span class="hi">${fmt(mc)}</span>`;
  s+=`, peaking <span class="hi">${fmt(pc)}</span> around <b>${DATES[pi].slice(0,7)}</b>.`;
  if(at){const ds=Object.values(at);const top=ds[0];
    let ai=0;top.series.forEach((v,i)=>{if(v!=null&&v===Math.max(...top.series.filter(x=>x!=null)))ai=i});
    s+=` The dominant news channel is <span class="hi">${top.label}</span> `;
    s+=top.sign>=0?`(a shared <b>co-movement</b> driver)`:`(a <b>divergence</b> driver)`;
    s+=`, explaining on average <span class="hi">${fmt(top.mean)}</span> of the correlation — strongest around <b>${DATES[ai].slice(0,7)}</b>.`;
    if(ds.length>1)s+=` ${ds.length-1} other channel${ds.length>2?'s':''} also contribute${ds.length===2?'s':''}.`;
  } else { s+=` No shared news driver in the corpus explains this pair — the co-movement is sector/market, not a specific news channel.`; }
  return s;
}

function chart(){
  const c=getCorr(A.sym,B.sym);const at=getAttr(A.sym,B.sym);
  const dObj=DRV&&at&&at[DRV]?at[DRV]:null;
  const W=1000,H=300,mL=44,mR=18,mT=14,mB=28,iw=W-mL-mR,ih=H-mT-mB;
  const lo=-0.2,hi=1;
  const X=i=>mL+iw*i/(DATES.length-1), Y=v=>mT+ih*(1-(v-lo)/(hi-lo));
  const g=(t,a,p)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",t);for(const k in a)e.setAttribute(k,a[k]);if(p!=null)e.textContent=p;return e};
  const svg=g("svg",{viewBox:`0 0 ${W} ${H}`,id:"cc"});
  for(let v=0;v<=1.001;v+=0.25){const yy=Y(v);svg.appendChild(g("line",{class:v===0?"zeroline":"gridline",x1:mL,x2:mL+iw,y1:yy,y2:yy}));
    svg.appendChild(g("text",{class:"axlab",x:mL-7,y:yy+3,"text-anchor":"end"},v.toFixed(2)))}
  let last="";DATES.forEach((d,i)=>{const q=d.slice(0,4)+"Q"+(Math.floor((+d.slice(5,7)-1)/3)+1);
    if(q!==last){last=q;const xx=X(i);svg.appendChild(g("line",{class:"gridline",x1:xx,x2:xx,y1:mT,y2:mT+ih}));
      svg.appendChild(g("text",{class:"axlab",x:xx,y:H-10,"text-anchor":"middle"},q))}});
  // counterfactual + shaded gap
  if(dObj){const cf=c.map((v,i)=>v==null||dObj.series[i]==null?null:v-dObj.series[i]);
    const pts=[];c.forEach((v,i)=>{if(v!=null&&cf[i]!=null)pts.push([i,v,cf[i]])});
    if(pts.length){const up=pts.map(p=>X(p[0]).toFixed(1)+" "+Y(p[1]).toFixed(1)).join(" L ");
      const dn=pts.slice().reverse().map(p=>X(p[0]).toFixed(1)+" "+Y(p[2]).toFixed(1)).join(" L ");
      svg.appendChild(g("path",{d:`M ${up} L ${dn} Z`,fill:"var(--raw)",opacity:.13,stroke:"none"}))}
    let d1="",pen=false;cf.forEach((v,i)=>{if(v==null){pen=false;return}d1+=(pen?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" ";pen=true});
    svg.appendChild(g("path",{class:"ln",d:d1,stroke:"var(--cf)","stroke-dasharray":"5 4"}))}
  // raw corr
  let d0="",pen=false;c.forEach((v,i)=>{if(v==null){pen=false;return}d0+=(pen?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" ";pen=true});
  svg.appendChild(g("path",{class:"ln",d:d0,stroke:"var(--raw)"}));
  svg.appendChild(g("line",{class:"cross",id:"cx",x1:0,x2:0,y1:mT,y2:mT+ih}));
  // hover
  svg.addEventListener("mousemove",e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
    let i=Math.round((px-mL)/iw*(DATES.length-1));i=Math.max(0,Math.min(DATES.length-1,i));
    const cx=el("cx");cx.setAttribute("x1",X(i));cx.setAttribute("x2",X(i));cx.style.opacity=1;
    const cf=dObj&&c[i]!=null&&dObj.series[i]!=null?c[i]-dObj.series[i]:null;
    let h=`<div class="dt">${DATES[i]}</div><div class="r"><span>correlation</span><b>${fmt(c[i])}</b></div>`;
    if(dObj){h+=`<div class="r"><span>${dObj.label} removed</span><b>${fmt(cf)}</b></div>`;
      h+=`<div class="r"><span>attribution</span><b>${dObj.series[i]==null?"—":"+"+dObj.series[i].toFixed(2)}</b></div>`}
    const tip=el("tip");tip.innerHTML=h;tip.style.opacity=1;
    tip.style.left=Math.min(e.clientX+16,innerWidth-tip.offsetWidth-10)+"px";tip.style.top=(e.clientY+14)+"px"});
  svg.addEventListener("mouseleave",()=>{el("tip").style.opacity=0;const cx=el("cx");if(cx)cx.style.opacity=0});
  return svg;
}

function renderMain(){
  const m=el("mainpane");
  if(!A){m.innerHTML=`<div class="empty">← Pick an instrument to begin.<br><br>You'll see every other name ranked by how tightly it co-moves,<br>then drill into any pair to see the news channels holding them together.</div>`;return}
  if(!B){
    const dv=A.drivers.slice(0,10).map(d=>`<span class="chip">${d.label} ${d.sign>0?'↑':'↓'}</span>`).join(" ");
    m.innerHTML=`<div class="top"><div class="title">${secDot(A.sector)}${A.name} <span class="x">— select a correlated instrument →</span></div>
      <div class="tools"><button id="theme">◐ theme</button></div></div>
      <div class="card"><h3>${A.name} · ${A.sector} · ${A.freq} news mentions</h3>
      <div class="hint">News drivers most often cited for this name:</div><div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">${dv}</div></div>
      <div class="empty">Pick a name from the ranked list on the left to see their correlation and its news drivers over time.</div>`;
    el("theme").onclick=toggleTheme;return}
  const at=getAttr(A.sym,B.sym);
  const drvHtml=at?Object.entries(at).map(([id,d])=>`<div class="drv ${DRV===id?'on':''}" data-drv="${id}">
      <span class="dot" style="background:${d.sign>=0?'var(--raw)':'var(--cf)'}"></span>
      <span class="dl">${d.label}</span>
      <span class="tag ${d.sign>=0?'co':'dv'}">${d.sign>=0?'co-move':'divergence'}</span>
      <span class="mn">${fmt(d.mean)}</span></div>`).join(""):
      `<div class="hint">No shared news driver explains this pair — its co-movement is sector/market, not a specific news channel.</div>`;
  m.innerHTML=`<div class="top">
      <div class="title">${secDot(A.sector)}${A.name} <span class="x">↔</span> ${secDot(B.sector)}${B.name}</div>
      <div class="tools"><button id="swap">⇄ swap</button><button id="theme">◐ theme</button></div></div>
    <div class="narr" id="narr">${narrative()}</div>
    <div class="card"><h3>Rolling ${WIN}-day correlation${DRV?` &amp; the “${at[DRV].label} removed” counterfactual`:''}</h3>
      <div class="legend"><span class="lg"><span class="sw" style="border-color:var(--raw)"></span>correlation</span>
        ${DRV?`<span class="lg"><span class="sw dash" style="border-color:var(--cf)"></span>driver removed</span><span class="lg"><span class="sw fill" style="background:var(--raw)"></span>attribution</span>`:'<span class="hint">click a driver below to overlay its counterfactual</span>'}</div>
      <div id="chartholder"></div></div>
    <div class="card"><h3>News channels explaining this co-movement — ranked by attribution</h3>
      <div style="margin-top:6px">${drvHtml}</div></div>`;
  el("chartholder").appendChild(chart());
  el("theme").onclick=toggleTheme;
  const sw=el("swap");if(sw)sw.onclick=()=>{const t=A;A=B;B=t;DRV=null;renderAll()};
  m.querySelectorAll(".drv").forEach(r=>r.onclick=()=>{DRV=DRV===r.dataset.drv?null:r.dataset.drv;renderMain()});
}
function renderAll(){renderList();renderMain()}
function toggleTheme(){document.body.dataset.theme=document.body.dataset.theme==="dark"?"light":"dark";renderMain()}
el("search").oninput=renderList;
el("chg").onclick=()=>{A=null;B=null;DRV=null;el("search").value="";renderAll()};
el("modes").querySelectorAll("button").forEach(b=>b.onclick=()=>{MODE=b.dataset.m;
  el("modes").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));renderList()});
el("list").addEventListener("click",e=>{const r=e.target.closest(".row");if(!r)return;const s=r.dataset.sym;
  if(!A){A=IM[s];el("search").value=""}else{B=IM[s];DRV=null}renderAll()});
renderAll();
</script></body></html>"""

out = G / "explorer.html"
out.write_text(HTML.replace("__DATA__", json.dumps(DATA)))
print(f"-> {out}  ({len(DATA['instruments'])} instruments, {len(DATA['corr'])} pairs, {round(len(HTML)/1e6,1)}MB template)")
