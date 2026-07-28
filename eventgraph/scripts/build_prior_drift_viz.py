# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
RELATIVE RISK DRIFT — news classes do not hold their risk ranking over time.

WHY THIS MATTERS. Every gate decision in this system compares an observed
idiosyncratic move against a MEASURED prior for its news class ("a 1.6-sigma
reaction is normal for earnings"). Those priors were estimated once, pooled over
2010-2012. A period split showed that is not safe:

    class             2010-H2  2011-H1  2011-H2   2012   spread
    CONTROL              0.69     0.76     0.88    0.82     0.19
    earnings             1.48     1.40     1.84    1.78     0.44
    rating_action        1.06     0.97     1.45    0.80     0.64   <- 81% swing
    legal_regulatory     1.38     0.89     1.22    0.99     0.49

Classes drift TWICE as much as the baseline (mean spread 0.38 vs 0.19 sigma), and
the drift is not a common factor — the RANKING reorders. legal_regulatory is the
second-riskiest class in 2010-H2 and mid-pack by 2012. A static prior therefore
mis-gates in both directions: too strict when the true prior is low (real
surprises marked insufficient), too loose when it is high (ordinary days waved
through).

WHAT THIS BUILDS. Rolling estimates on a trailing window, stepped monthly:
  1. RELATIVE RISK over time — each class's prior divided by the CONTROL measured
     on the same window. Dividing by the contemporaneous control is the point: it
     separates "this class got riskier" from "everything got riskier".
  2. RANK (bump chart) — the ordering itself, which is what a static prior most
     badly misrepresents.
  3. DISPERSION — how far apart the classes are at each date, i.e. how much
     discriminating power the class taxonomy actually has in that regime.

Usage:
    uv run scripts/build_prior_drift_viz.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate import Gate, G  # noqa: E402

OUT = G / "prior_drift.html"
N_CLASSES = 7       # categorical palette limit — never cycle hues


def main() -> None:
    """Read the SHARED rolling table. This view deliberately does not re-derive the
    pipeline: an earlier copy-paste of it drifted (different control, stale names
    left in) and that is exactly the failure mode gate.py exists to prevent."""
    g = Gate()
    freq = {c: len(v) for c, v in g.priors.items()}
    classes = sorted(freq, key=lambda c: -freq[c])[:N_CLASSES]
    points = []
    for k, d in enumerate(g.dates):
        cls = {}
        for c in classes:
            v = g.priors[c].get(d)
            if v:
                cls[c] = {"mult": round(v[0] / g.control[k], 3), "sig": round(v[0], 3),
                          "sd": round(v[1], 3)}
        if len(cls) >= 4:
            points.append({"d": d, "ctrl": round(g.control[k], 3), "cls": cls})
    print(f"{len(points)} rolling points {points[0]['d']}..{points[-1]['d']}, "
          f"classes {classes}")
    for p_ in points:
        for r, c in enumerate(sorted(p_["cls"], key=lambda c: -p_["cls"][c]["mult"]), 1):
            p_["cls"][c]["rank"] = r
    ranks = {c: [p_["cls"][c]["rank"] for p_ in points if c in p_["cls"]] for c in classes}
    churn = {c: (min(v), max(v)) for c, v in ranks.items() if len(v) > 5}
    print("rank range:", {c: f"{a}-{b}" for c, (a, b) in churn.items()})
    print(f"control {min(g.control):.2f}..{max(g.control):.2f}σ")
    payload = {"points": points, "classes": classes,
               "rank_range": {c: list(v) for c, v in churn.items()}}
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relative risk drift — news classes reorder over time</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--warn:#fab219;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1120px;margin:0 auto;padding:22px 18px 60px}
 h1{font-size:19px;margin:0 0 3px}h2{font-size:14px;margin:0 0 3px}
 .sub{color:var(--ink2);font-size:12.5px;margin:0 0 16px;max-width:82ch}
 .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;margin-bottom:15px}
 .note{color:var(--muted);font-size:11.5px;margin:0 0 12px;max-width:86ch}
 .lg{display:flex;gap:15px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-top:11px}
 .key{display:inline-flex;align-items:center;gap:6px;cursor:pointer;opacity:.95}
 .key.off{opacity:.3}
 .sw{width:13px;height:3px;border-radius:2px}
 .tog{position:fixed;top:12px;right:14px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
 .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface);
  border:1px solid var(--border);border-radius:7px;padding:8px 11px;font-size:12px;
  box-shadow:0 8px 26px rgba(0,0,0,.45);z-index:9}
 .tip b{color:var(--ink)}.tip div{color:var(--ink2);margin-top:2px}
 table{border-collapse:collapse;font-size:12px;margin-top:4px}
 td,th{padding:4px 10px;text-align:left}th{color:var(--muted);font-size:10.5px;
  text-transform:uppercase;letter-spacing:.05em;font-weight:500}
 footer{color:var(--muted);font-size:11px;margin-top:20px;line-height:1.7}
</style></head><body>
<button class="tog" onclick="tg()">◐</button><div class="wrap">
<h1>Relative risk drift — news classes reorder over time</h1>
<p class="sub">Every gate decision compares an observed move against a measured prior for its news
class. Those priors are <b>not stable</b>. Each series below is a class's risk
<b>relative to the control measured on the same window</b> — dividing by the contemporaneous control
separates "this class got riskier" from "everything got riskier". Rolling 250-day windows, stepped
monthly. Click the legend to isolate a class.</p>

<div class="card"><h2>1 · Relative risk over time (class prior ÷ contemporaneous control)</h2>
<p class="note">1.0 = indistinguishable from an ordinary non-event day. A class crossing below 1.0
has, in that regime, stopped carrying measurable idiosyncratic risk.</p>
<svg id="c1" viewBox="0 0 1020 330" style="width:100%"></svg><div class="lg" id="lg1"></div></div>

<div class="card"><h2>2 · Rank — the ordering itself reorders</h2>
<p class="note">Rank 1 = riskiest class in that window. This is what a static prior most badly
misrepresents: it assumes these lines are flat and parallel.</p>
<svg id="c2" viewBox="0 0 1020 300" style="width:100%"></svg>
<table id="rr"><tr><th>class</th><th>rank range across the sample</th></tr></table></div>

<div class="card"><h2>3 · Discriminating power</h2>
<p class="note">Spread between the riskiest and least-risky class at each date. When this narrows,
the class taxonomy carries less information — knowing the news type tells you less about the risk.</p>
<svg id="c3" viewBox="0 0 1020 190" style="width:100%"></svg></div>

<footer><b>Method.</b> Idiosyncratic move = market-model residual, standardised by each name's own
trailing 60-day residual vol. Control = mean idiosyncratic move on random non-event days within the
same trailing window. Classes shown are the 7 most frequent; a class appears only where the window
holds ≥18 events of that type.<br>
<b>Consequence.</b> Gate priors should be trailing, not pooled — and the UI should show "prior as of
this date", not a constant.</footer>
</div><div class="tip" id="tip"></div>
<script>
const D=__DATA__,tip=document.getElementById('tip');
const PAL=['--s1','--s2','--s3','--s4','--s5','--s6','--s7'];
const CLS=D.classes,off=new Set();
const esc=s=>String(s).replace(/_/g,' ');
function show(e,h){tip.innerHTML=h;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-240)+'px';tip.style.top=(e.clientY+14)+'px'}
function hide(){tip.style.opacity=0}
const P=D.points,W=1020,L=54,R=118;
const X=i=>L+(i/(P.length-1))*(W-L-R);
function axes(svg,H,lo,hi,fmt,ticks){
 let s='';const Y=v=>H-28-((v-lo)/(hi-lo))*(H-52);
 for(let k=0;k<=ticks;k++){const v=lo+(hi-lo)*k/ticks,y=Y(v);
  s+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>
   <text x="${L-8}" y="${y+3.5}" font-size="10" fill="var(--muted)" text-anchor="end">${fmt(v)}</text>`}
 P.forEach((p,i)=>{if(i%6)return;s+=`<text x="${X(i)}" y="${H-10}" font-size="9.5"
  fill="var(--muted)" text-anchor="middle">${p.d.slice(0,7)}</text>`});
 return{s,Y}}
function draw(){
 // 1. relative risk
 let vals=[];P.forEach(p=>CLS.forEach(c=>{if(p.cls[c]&&!off.has(c))vals.push(p.cls[c].mult)}));
 const lo=Math.min(0.9,...vals)*0.98,hi=Math.max(...vals)*1.02;
 let{s,Y}=axes(null,330,lo,hi,v=>v.toFixed(1)+'×',5);
 s+=`<line x1="${L}" y1="${Y(1)}" x2="${W-R}" y2="${Y(1)}" stroke="var(--warn)" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="${W-R+6}" y="${Y(1)+3.5}" font-size="10" fill="var(--warn)">1.0× control</text>`;
 CLS.forEach((c,ci)=>{if(off.has(c))return;
  const pts=P.map((p,i)=>p.cls[c]?[X(i),Y(p.cls[c].mult)]:null).filter(Boolean);
  if(pts.length<2)return;
  s+=`<path d="M${pts.map(q=>q.join(',')).join(' L')}" fill="none"
   stroke="var(${PAL[ci%7]})" stroke-width="2"/>`;
  const last=pts[pts.length-1];
  s+=`<text x="${last[0]+7}" y="${last[1]+3.5}" font-size="10" fill="var(${PAL[ci%7]})">${esc(c)}</text>`;
  P.forEach((p,i)=>{if(!p.cls[c])return;
   s+=`<circle cx="${X(i)}" cy="${Y(p.cls[c].mult)}" r="7" fill="transparent"
    onmousemove='show(event,${JSON.stringify(`<b>${esc(c)}</b><div>${p.d}</div>`)}+"<div>"+${p.cls[c].mult}+"× control</div><div>"+${p.cls[c].sig}+"σ vs "+${p.ctrl}+"σ</div><div>n = "+${p.cls[c].n}+"</div>")' onmouseleave="hide()"/>`})});
 document.getElementById('c1').innerHTML=s;
 // 2. rank bump
 const mr=Math.max(...P.flatMap(p=>Object.values(p.cls).map(v=>v.rank)));
 let a2=axes(null,300,mr+.5,.5,v=>Math.round(v),mr-1);s=a2.s;const Y2=a2.Y;
 CLS.forEach((c,ci)=>{if(off.has(c))return;
  const pts=P.map((p,i)=>p.cls[c]?[X(i),Y2(p.cls[c].rank)]:null).filter(Boolean);
  if(pts.length<2)return;
  s+=`<path d="M${pts.map(q=>q.join(',')).join(' L')}" fill="none" stroke="var(${PAL[ci%7]})"
   stroke-width="2.4" stroke-linejoin="round" opacity=".92"/>`;
  pts.forEach(q=>{s+=`<circle cx="${q[0]}" cy="${q[1]}" r="3.1" fill="var(${PAL[ci%7]})"/>`});
  const last=pts[pts.length-1];
  s+=`<text x="${last[0]+7}" y="${last[1]+3.5}" font-size="10" fill="var(${PAL[ci%7]})">${esc(c)}</text>`});
 document.getElementById('c2').innerHTML=s;
 // 3. dispersion
 const disp=P.map(p=>{const v=Object.values(p.cls).map(x=>x.mult);return Math.max(...v)-Math.min(...v)});
 const a3=axes(null,190,Math.min(...disp)*.9,Math.max(...disp)*1.05,v=>v.toFixed(1),4);
 s=a3.s;const Y3=a3.Y;
 s+=`<path d="M${P.map((p,i)=>[X(i),Y3(disp[i])].join(',')).join(' L')}" fill="none"
  stroke="var(--s1)" stroke-width="2"/>`;
 P.forEach((p,i)=>{s+=`<circle cx="${X(i)}" cy="${Y3(disp[i])}" r="7" fill="transparent"
  onmousemove='show(event,${JSON.stringify(`<b>${'spread'}</b>`)}+"<div>"+${JSON.stringify(p.d)}+"</div><div>"+${disp[i].toFixed(2)}+"× between riskiest and least</div>")' onmouseleave="hide()"/>`});
 document.getElementById('c3').innerHTML=s;
 // legend + rank table
 document.getElementById('lg1').innerHTML=CLS.map((c,i)=>
  `<span class="key${off.has(c)?' off':''}" onclick="tk('${c}')">
   <span class="sw" style="background:var(${PAL[i%7]})"></span>${esc(c)}</span>`).join('');
 document.getElementById('rr').innerHTML='<tr><th>class</th><th>rank range across the sample</th></tr>'+
  CLS.filter(c=>D.rank_range[c]).map((c,i)=>{const[a,b]=D.rank_range[c];
   return `<tr><td style="color:var(${PAL[CLS.indexOf(c)%7]})">${esc(c)}</td>
   <td>${a===b?`stable at ${a}`:`<b>${a} → ${b}</b> (moves ${b-a} places)`}</td></tr>`}).join('');
}
function tk(c){off.has(c)?off.delete(c):off.add(c);draw()}
draw();
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light'}
</script></body></html>"""


if __name__ == "__main__":
    main()
