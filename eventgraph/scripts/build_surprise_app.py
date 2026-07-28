# /// script
# requires-python = ">=3.10"
# ///
"""Render news_surprise_top.json -> a SINGLE self-contained HTML surprise ranker.
Vanilla JS + inline CSS, no fetch/build/CDN — must open straight from file://.
Follows the repo's self-contained-HTML convention (see build_explorer_app.py)."""
import json
from pathlib import Path

G = Path("../data/eg_runs/eg100k_graph")
DATA = json.loads((G / "news_surprise_top.json").read_text())

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Class-adjusted news surprise ranker</title>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--ink3:#898781;
      --grid:#e1e0d9;--base:#c3c2b7;--line:rgba(11,11,11,.10);--neutral:#f0efec;
      --pos:#2a78d6;--neg:#e34948;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;color-scheme:light}
[data-theme=dark]{--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#898781;
      --grid:#2c2c2a;--base:#383835;--line:rgba(255,255,255,.10);--neutral:#383835;
      --pos:#3987e5;--neg:#e66767;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--page);color:var(--ink);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:22px 26px 60px}
.wrap{max-width:1280px;margin:0 auto}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.hdr h1{font-size:19px;font-weight:650;letter-spacing:-.01em}
.hdr p{font-size:12.5px;color:var(--ink3);margin-top:3px;max-width:640px}
.tools button{font:inherit;font-size:12px;color:var(--ink2);background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:6px 12px;cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .n{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.tile .l{font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.filters{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--surface);border:1px solid var(--line);
         border-radius:12px;padding:10px 14px;margin-bottom:14px;font-size:12px}
.filters label{display:flex;align-items:center;gap:6px;color:var(--ink2);cursor:pointer;white-space:nowrap}
.filters select,.filters input[type=text]{font:inherit;font-size:12px;background:var(--page);color:var(--ink);
         border:1px solid var(--line);border-radius:7px;padding:5px 9px}
.filters input[type=range]{width:110px}
.filters .grow{flex:1;min-width:140px}
.filters .val{font-variant-numeric:tabular-nums;color:var(--ink3);min-width:34px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-bottom:10px;align-items:center}
.legend .lg{display:flex;gap:6px;align-items:center}
.badge{font-size:10px;padding:2px 7px;border-radius:20px;font-weight:600;white-space:nowrap}
.tick{display:inline-block;width:2px;height:11px;background:var(--ink2);vertical-align:-2px}
.list{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.colhdr{display:grid;grid-template-columns:78px 64px 74px 1fr 320px 1fr 90px;gap:10px;padding:8px 14px;
        font-size:10.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--line)}
.row{display:grid;grid-template-columns:78px 64px 74px 1fr 320px 1fr 90px;gap:10px;padding:9px 14px;align-items:center;
     border-bottom:1px solid var(--line);cursor:default}
.row:last-child{border-bottom:none}
.row:hover{background:var(--page)}
.dt{color:var(--ink3);font-variant-numeric:tabular-nums;font-size:12px}
.tk{font-weight:650;font-size:12.5px}
.sd{font-variant-numeric:tabular-nums;font-size:12px;color:var(--ink2);text-align:right}
.hl{font-size:12.5px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.flags{display:flex;gap:5px}
.flag{font-size:9.5px;padding:1px 6px;border-radius:20px;background:var(--page);border:1px solid var(--line);color:var(--ink3)}
.flag.on{color:var(--ink);border-color:var(--ink2)}
svg.bullet{display:block;overflow:visible}
.empty{padding:50px;text-align:center;color:var(--ink3)}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:9px;
     padding:10px 12px;font-size:12px;box-shadow:0 8px 26px rgba(0,0,0,.3);opacity:0;transition:opacity .08s;z-index:20;max-width:340px}
.tip .r{display:flex;justify-content:space-between;gap:14px;margin-bottom:2px}
.tip b{font-variant-numeric:tabular-nums}
.tip .q{margin-top:6px;color:var(--ink2);font-style:italic;font-size:11.5px;line-height:1.4}
.tip .h{font-weight:600;margin-bottom:4px}
.count{font-size:11.5px;color:var(--ink3);margin:8px 2px 0}
</style></head>
<body data-theme="dark">
<div class="wrap">
  <div class="hdr">
    <div>
      <h1>Class-adjusted news surprise ranker</h1>
      <p>Ranks news by <b>surprise = observed abnormal volume − expected for its event class</b>
      (out-of-sample priors from train window ${TRAIN}), not raw abnormality — so a quiet-class story
      that still moves the tape outranks a routine earnings print. Scored on the news day T
      (window ${SCORE}); the volume spike is &gt;half decayed by T+1.</p>
    </div>
    <div class="tools"><button id="theme">◐ theme</button></div>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="filters">
    <label><input type="checkbox" id="fUnsched"> unscheduled only</label>
    <label><input type="checkbox" id="fTape"> tape-confirmed only</label>
    <select id="fSector"><option value="">All sectors</option></select>
    <select id="fGroup"><option value="">All event groups</option></select>
    <label>min surprise <input type="range" id="fMin" min="-3" max="9" step="0.25" value="-3"><span class="val" id="fMinVal">-3.0</span></label>
    <input class="grow" type="text" id="fSearch" placeholder="Search ticker / headline…">
    <select id="fSort">
      <option value="surprise">Sort: surprise</option>
      <option value="date">Sort: date</option>
      <option value="observed">Sort: observed</option>
    </select>
  </div>
  <div class="legend">
    <span class="lg"><span class="badge" style="background:var(--s2);color:#fff">corporate</span></span>
    <span class="lg"><span class="badge" style="background:var(--s1);color:#fff">macro</span></span>
    <span class="lg"><span class="badge" style="background:var(--s3);color:#fff">commodity</span></span>
    <span class="lg"><span class="badge" style="background:var(--s4);color:#fff">market</span></span>
    <span class="lg"><span class="badge" style="background:var(--base);color:var(--ink)">other</span></span>
    <span class="lg"><span class="tick"></span> class-expected vol-z (the gap to the bar end is the surprise)</span>
    <span class="lg"><span style="color:var(--pos)">■</span> positive surprise</span>
    <span class="lg"><span style="color:var(--neg)">■</span> negative surprise</span>
  </div>
  <div class="colhdr"><div>Date</div><div>Ticker</div><div>Group</div><div>Observed vs expected</div><div>Headline</div><div>Sector</div><div style="text-align:right">Surprise</div></div>
  <div class="list" id="list"></div>
  <div class="count" id="count"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const ROWS=D.rows;
const GROUP_COLOR={corporate:"var(--s2)",macro:"var(--s1)",commodity:"var(--s3)",market:"var(--s4)",other:"var(--base)"};
const el=id=>document.getElementById(id);
const fmt=v=>v==null?"—":(v>=0?"+":"")+v.toFixed(2);

// ---- axis: shared x-scale across all bullet rows ----
const allVals=ROWS.flatMap(r=>[r.obs_vol_z,r.expected_vol_z]).filter(v=>v!=null);
const XMIN=Math.min(-1,Math.floor(Math.min(...allVals))), XMAX=Math.max(1,Math.ceil(Math.max(...allVals)));
const BW=290,BH=22,zero=BW*(0-XMIN)/(XMAX-XMIN);
const xs=v=>BW*(v-XMIN)/(XMAX-XMIN);

function bullet(r){
  const x0=xs(0),x1=xs(r.obs_vol_z),xt=xs(r.expected_vol_z);
  const pos=r.surprise>=0;
  const barCol=pos?"var(--pos)":"var(--neg)";
  const left=Math.min(x0,x1),w=Math.max(2,Math.abs(x1-x0));
  return `<svg class="bullet" width="${BW}" height="${BH}" viewBox="0 0 ${BW} ${BH}">
    <line x1="${x0}" x2="${BW}" y1="${BH/2}" y2="${BH/2}" stroke="var(--grid)" stroke-width="1"/>
    <line x1="${x0}" x2="${x0}" y1="4" y2="${BH-4}" stroke="var(--base)" stroke-width="1"/>
    <rect x="${left}" y="${BH/2-5}" width="${w}" height="10" rx="4" fill="${barCol}"/>
    <line x1="${xt}" x2="${xt}" y1="2" y2="${BH-2}" stroke="var(--ink)" stroke-width="2"/>
  </svg>`;
}

function populateSelects(){
  const secs=[...new Set(ROWS.map(r=>r.std_sector))].sort();
  const groups=[...new Set(ROWS.map(r=>r.event_group))].sort();
  el("fSector").innerHTML+=secs.map(s=>`<option value="${s}">${s}</option>`).join("");
  el("fGroup").innerHTML+=groups.map(g=>`<option value="${g}">${g}</option>`).join("");
}

function tiles(){
  const n=ROWS.length;
  const med=(()=>{const v=ROWS.map(r=>r.surprise).sort((a,b)=>a-b);return v.length?v[Math.floor(v.length/2)]:0})();
  const pctU=n?100*ROWS.filter(r=>r.unscheduled).length/n:0;
  const pctT=n?100*ROWS.filter(r=>r.tape_confirmed).length/n:0;
  el("tiles").innerHTML=`
    <div class="tile"><div class="n">${n.toLocaleString()}</div><div class="l">scored items</div></div>
    <div class="tile"><div class="n">${fmt(med)}</div><div class="l">median surprise</div></div>
    <div class="tile"><div class="n">${pctU.toFixed(1)}%</div><div class="l">unscheduled</div></div>
    <div class="tile"><div class="n">${pctT.toFixed(1)}%</div><div class="l">tape-confirmed (|z|≥2)</div></div>`;
}

function filtered(){
  const unsched=el("fUnsched").checked, tape=el("fTape").checked;
  const sec=el("fSector").value, grp=el("fGroup").value;
  const min=parseFloat(el("fMin").value);
  const q=el("fSearch").value.trim().toLowerCase();
  const sort=el("fSort").value;
  let r=ROWS.filter(r=>
    (!unsched||r.unscheduled) && (!tape||r.tape_confirmed) &&
    (!sec||r.std_sector===sec) && (!grp||r.event_group===grp) &&
    (r.surprise>=min) &&
    (!q||r.ticker.toLowerCase().includes(q)||(r.headline||"").toLowerCase().includes(q)||(r.name||"").toLowerCase().includes(q)));
  if(sort==="surprise")r=r.sort((a,b)=>b.surprise-a.surprise);
  else if(sort==="date")r=r.sort((a,b)=>b.date.localeCompare(a.date));
  else if(sort==="observed")r=r.sort((a,b)=>b.obs_vol_z-a.obs_vol_z);
  return r;
}

function renderList(){
  const rows=filtered();
  const list=el("list");
  if(!rows.length){list.innerHTML=`<div class="empty">No rows match the current filters.</div>`;el("count").textContent="";return}
  list.innerHTML=rows.slice(0,500).map(r=>{
    const col=GROUP_COLOR[r.event_group]||"var(--base)";
    const flags=`<span class="flags">
        <span class="flag ${r.unscheduled?'on':''}">unsched</span>
        <span class="flag ${r.tape_confirmed?'on':''}">tape</span></span>`;
    return `<div class="row" data-i="${ROWS.indexOf(r)}">
      <div class="dt">${r.date}</div>
      <div class="tk">${r.ticker}</div>
      <div><span class="badge" style="background:${col};color:${r.event_group==='other'?'var(--ink)':'#fff'}">${r.event_group}</span></div>
      <div>${bullet(r)}</div>
      <div class="hl" title="${(r.headline||'').replace(/"/g,'&quot;')}">${r.headline||'—'}</div>
      <div>${r.std_sector}${flags}</div>
      <div class="sd">${fmt(r.surprise)}</div>
    </div>`;
  }).join("");
  el("count").textContent=`Showing ${Math.min(500,rows.length)} of ${rows.length} matching rows (of ${ROWS.length} total).`;
  list.querySelectorAll(".row").forEach(rowEl=>{
    const r=ROWS[+rowEl.dataset.i];
    rowEl.addEventListener("mousemove",e=>{
      const tip=el("tip");
      tip.innerHTML=`<div class="h">${r.name||r.ticker} · ${r.std_event_type}</div>
        <div class="r"><span>observed vol-z</span><b>${fmt(r.obs_vol_z)}</b></div>
        <div class="r"><span>expected (class prior)</span><b>${fmt(r.expected_vol_z)}</b></div>
        <div class="r"><span>surprise</span><b>${fmt(r.surprise)}</b></div>
        ${r.headline?`<div class="q">"${r.headline}"</div>`:''}
        ${r.quote?`<div class="q">${r.quote}</div>`:''}`;
      tip.style.opacity=1;
      tip.style.left=Math.min(e.clientX+16,innerWidth-360)+"px";
      tip.style.top=Math.min(e.clientY+14,innerHeight-160)+"px";
    });
    rowEl.addEventListener("mouseleave",()=>{el("tip").style.opacity=0});
  });
}

function renderAll(){tiles();renderList()}
["fUnsched","fTape","fSector","fGroup","fSort"].forEach(id=>el(id).addEventListener("change",renderList));
el("fSearch").addEventListener("input",renderList);
el("fMin").addEventListener("input",()=>{el("fMinVal").textContent=parseFloat(el("fMin").value).toFixed(2);renderList()});
el("theme").onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==="dark"?"light":"dark"};
populateSelects();
renderAll();
</script></body></html>"""

train = f"{DATA['train_window'][0]} .. {DATA['train_window'][1]}"
score = f"{DATA['score_window'][0]} .. {DATA['score_window'][1]}"
HTML = HTML.replace("${TRAIN}", train).replace("${SCORE}", score)
HTML = HTML.replace("__DATA__", json.dumps(DATA))

out = G / "news_surprise.html"
out.write_text(HTML)
size_kb = round(len(HTML) / 1024, 1)
has_ext = "http://" in HTML or "https://" in HTML
print(f"-> {out}  ({size_kb} KB, rows={len(DATA['rows'])}, external refs present={has_ext})")
