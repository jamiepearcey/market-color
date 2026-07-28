# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
TICKER DESK — one name at a time: who it moves with, what kind of news it gets,
and which of its events actually survive the gate.

Built in the `news_desk.html` visual language (observed-vs-expected with the
class-prior reference tick, verbatim quotes, explicit refusals) and carrying the
line-drawing from `sector_risk_viz.html`. Three panels per ticker:

  A. NEWS WEIGHT BREAKDOWN — the mix of event classes this name's news falls
     into, each bar positioned against the MEASURED class prior. This answers
     "what kind of risk does this company's news actually carry" rather than
     "how much news does it get". A name whose coverage is mostly m_and_a
     (prior 0.91 sigma) is in a different risk regime from one dominated by
     earnings (1.62 sigma), even at identical article counts.

  B. CORRELATION LINE-DRAWING — the ticker at the centre, its strongest
     IDIOSYNCRATIC-correlation neighbours radiating out; line thickness =
     strength, colour = the neighbour's GICS sector, cross-sector links drawn
     solid because those are the exposures sector classification does not
     encode (28% of the strongest links, measured). Residual correlation, so a
     line means shared company-specific risk, not two names tracking the index.

  C. GATED EVENT HISTORY — every event-day for this name, scored
     observed-vs-class-prior, with failures rendered as "no explanation" AND
     THE REASON. Same three gates as the desk: move below the random-day
     control (0.72 sigma); class that does not itself clear the control
     (election, credit_event); surprise <= 0 (within class expectation).

Usage:
    uv run scripts/build_ticker_desk.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# SIC (SEC EDGAR), not GICS. GICS is proprietary so gics.py was ~140 tickers
# assigned BY HAND — 12% of the universe, unverifiable, and flat. SIC is public
# domain, assigned per filer by the SEC itself (74% coverage here), and genuinely
# hierarchical: division -> major group -> 4-digit industry, which is what the
# drill-down needs. SIC is an older and coarser taxonomy than GICS; it is used
# because it is real and checkable, which beats a better-shaped invented one.
from sic import sector as gics_sector, subsector, industry  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
OUT = G / "ticker_desk.html"
BETA_WIN, TRAIL = 250, 60
MAX_TICKERS = 140          # names with enough news+price to be worth a page
MAX_NEIGHBOURS = 9
MAX_EVENTS = 26

PRIOR = {"earnings": 1.621, "employment": 1.386, "debt_issuance": 1.318,
         "econ_indicator": 1.257, "guidance": 1.242, "other": 1.161,
         "legal_regulatory": 1.153, "monetary_policy": 1.118,
         "rating_action": 1.063, "growth": 1.053, "m_and_a": 0.912,
         "credit_event": 0.797, "election": 0.779}
FAILS_CONTROL = {"credit_event", "election"}


# STALE-PRICE FILTER. 17% of the resolved universe are illiquid OTC foreign ADRs
# (GECFF, QUCCF, KAKKF ...) that barely trade: >20% of their days print an exactly
# zero return, some 99%. Their trailing residual vol collapses toward zero, so the
# one day they DO move divides by ~0 — a 267-sigma "control" was traced to this.
# A `std() > 0` test does not catch it; a mostly-zero window has a tiny-but-positive
# std. Reject the name outright, and reject any window that is mostly stale.
MAX_ZERO_FRAC = 0.10
MIN_RESID_VOL = 0.002   # 0.2%/day floor on trailing residual vol
def _tradeable(r):
    v = np.abs(np.array(list(r.values())))
    return len(v) > 600 and (v < 1e-12).mean() <= MAX_ZERO_FRAC


def load_prices(cache: Path, sym: str):
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or ind["quote"][0]["close"]
    except Exception:
        return {}
    dd, px = [], []
    for t, c in zip(ts, cl):
        if c and c > 0:
            dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
    return {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}


def main() -> None:
    ent, name_of = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = j.get("resolved_ticker")
        if j.get("resolution_status") == "resolved_security" and tk:
            ent[j["entity_id"]] = tk.upper()
            name_of.setdefault(tk.upper(), j.get("name") or tk)
    docmeta = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docmeta[j["doc_id"]] = (d, j.get("headline") or "")
    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])

    cells = collections.defaultdict(lambda: {"types": set(), "q": None, "head": None})
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        meta = docmeta.get(doc)
        if e not in ent or not meta:
            continue
        d, head = meta
        c = cells[(ent[e], d)]
        c["types"] |= (doctypes.get(doc) or {"other"})
        q = (j.get("quote") or "").strip()
        if q and not c["q"]:
            c["q"] = q[:180]
        if head and not c["head"]:
            c["head"] = head[:120]

    cache = G / "prices"
    rets = {}
    for tk in sorted({t for t, _ in cells}):
        r = load_prices(cache, tk.replace("/", "-"))
        if r and not _tradeable(r):
            continue
        if len(r) > 600:
            rets[tk] = r
    dc = collections.Counter()
    for r in rets.values():
        dc.update(r.keys())
    thresh = max(30, int(0.30 * len(rets)))
    alldays = sorted([d for d, c in dc.items() if c >= thresh])
    dayidx = {d: i for i, d in enumerate(alldays)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])

    resid = {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < 600:
            continue
        e = np.full(len(alldays), np.nan)
        for i in range(BETA_WIN, len(alldays)):
            sl = slice(i - BETA_WIN, i); m = ok[sl]
            if m.sum() < 150:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean(); den = float(fc @ fc)
            if den > 0 and ok[i]:
                e[i] = y[i] - float((yy - yy.mean()) @ fc / den) * F[i]
        resid[tk] = e
    print(f"tickers with residuals: {len(resid)}")

    def msig(tk, i):
        e = resid[tk]
        if not np.isfinite(e[i]):
            return None
        w = e[i - TRAIL:i]; w = w[np.isfinite(w)]
        if len(w) < 40 or (np.abs(w) < 1e-12).mean() > MAX_ZERO_FRAC \
            or w.std() < MIN_RESID_VOL:
            return None
        return abs(e[i]) / w.std(), float(e[i] / w.std())

    rng = np.random.default_rng(11)
    evset = set(cells); tl = list(resid); ctrl = []
    for _ in range(9000):
        tk = tl[rng.integers(len(tl))]
        i = int(rng.integers(BETA_WIN + TRAIL + 5, len(alldays) - 2))
        if (tk, alldays[i]) in evset:
            continue
        m = msig(tk, i)
        if m:
            ctrl.append(m[0])
    CTRL = float(np.mean(ctrl))
    print(f"control: {CTRL:.3f} sigma")

    # rank tickers by how much gated evidence they actually have
    ev_by_tk = collections.defaultdict(list)
    for (tk, d), c in cells.items():
        if tk in resid and d in dayidx:
            ev_by_tk[tk].append((d, c))
    ranked = sorted(ev_by_tk, key=lambda t: -len(ev_by_tk[t]))[:MAX_TICKERS]

    # neighbours on the ranked set
    M = np.vstack([resid[t] for t in ranked]); ok = np.isfinite(M)
    neigh = {}
    for a, ta in enumerate(ranked):
        out = []
        for b, tb in enumerate(ranked):
            if a == b:
                continue
            m = ok[a] & ok[b]
            if m.sum() < 300:
                continue
            x, y = M[a][m], M[b][m]
            if x.std() > 0 and y.std() > 0:
                c = float(np.corrcoef(x, y)[0, 1])
                if abs(c) > 0.08:
                    out.append({"t": tb, "c": round(c, 3), "s": gics_sector(tb),
                                "x": gics_sector(tb) != gics_sector(ta)})
        out.sort(key=lambda o: -abs(o["c"]))
        neigh[ta] = out[:MAX_NEIGHBOURS]

    payload = {"ctrl": round(CTRL, 3), "prior": PRIOR,
               "fails": sorted(FAILS_CONTROL), "tickers": {}}
    for tk in ranked:
        mix = collections.Counter()
        evs = []
        for d, c in sorted(ev_by_tk[tk], key=lambda x: x[0]):
            i = dayidx[d]
            if i < BETA_WIN + TRAIL + 5 or i + 2 >= len(alldays):
                continue
            m = msig(tk, i)
            if m is None:
                continue
            obs, signed = m
            cls = max(c["types"], key=lambda t: PRIOR.get(t, 1.0))
            mix[cls] += 1
            exp = PRIOR.get(cls, 1.0)
            sur = obs - exp
            why = []
            if obs < CTRL:
                why.append(f"move {obs:.2f}σ is below the {CTRL:.2f}σ random-day control")
            if cls in FAILS_CONTROL:
                why.append(f"'{cls}' does not itself clear the control")
            if sur <= 0:
                why.append(f"within expectation for {cls} ({exp:.2f}σ)")
            evs.append({"d": d, "cls": cls, "obs": round(obs, 2), "exp": round(exp, 2),
                        "sur": round(sur, 2), "dir": "up" if signed > 0 else "down",
                        "head": c["head"], "q": c["q"], "ok": not why, "why": why})
        evs.sort(key=lambda e: -e["sur"])
        resid_vol = float(np.nanstd(resid[tk])) * math.sqrt(252)
        payload["tickers"][tk] = {
            "name": name_of.get(tk, tk), "sec": gics_sector(tk),
            "n_ev": len(evs), "pass": sum(1 for e in evs if e["ok"]),
            "idio_vol": round(resid_vol, 3),
            "mix": [[c, n, PRIOR.get(c, 1.0)] for c, n in mix.most_common()],
            "nb": neigh.get(tk, []),
            "ev": evs[:MAX_EVENTS],
        }
    print(f"tickers on the desk: {len(payload['tickers'])}")
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ticker desk — correlations, news weight, gated events</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --border:rgba(255,255,255,.10);--up:#0ca30c;--down:#e66767;--warn:#fab219;--acc:#3987e5;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;
  --s7:#9085e9;--s8:#e66767;--s9:#898781;--s10:#5598e7}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --border:rgba(11,11,11,.10);--up:#006300;--down:#e34948;--acc:#2a78d6;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;
  --s7:#4a3aa7;--s8:#e34948;--s9:#898781;--s10:#86b6ef}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1280px;margin:0 auto;padding:20px 18px 60px}
 h1{font-size:18px;margin:0 0 3px}
 .sub{color:var(--ink2);font-size:12.5px;margin:0 0 14px;max-width:84ch}
 .bar{display:flex;gap:10px;align-items:center;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;padding:10px 12px;margin-bottom:14px}
 select,input{background:var(--page);color:var(--ink);border:1px solid var(--border);
  border-radius:6px;padding:6px 9px;font-size:13px}
 .hd{display:flex;gap:26px;align-items:baseline;margin-bottom:12px;flex-wrap:wrap}
 .hd .t{font-size:26px;font-weight:600;letter-spacing:-.01em}
 .hd .n{color:var(--ink2);font-size:13px}
 .kpi{font-size:11px;color:var(--muted)}.kpi b{color:var(--ink);font-size:15px;
  font-variant-numeric:tabular-nums;display:block}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
 .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:15px 17px}
 h2{font-size:13.5px;margin:0 0 2px}
 .note{color:var(--muted);font-size:11.5px;margin:0 0 12px}
 .mixrow{display:grid;grid-template-columns:112px 1fr 40px;gap:9px;align-items:center;padding:3px 0}
 .lab{font-size:11.5px;color:var(--ink2);text-align:right;white-space:nowrap}
 .track{position:relative;height:15px}
 .fill{position:absolute;top:2px;height:11px;border-radius:0 3px 3px 0;background:var(--acc)}
 .ref{position:absolute;top:0;width:2px;height:15px;background:var(--warn)}
 .val{font-size:11px;color:var(--ink2);font-variant-numeric:tabular-nums}
 .ev{border-top:1px solid var(--border);padding:9px 0;display:grid;
  grid-template-columns:74px 1fr 76px;gap:11px;align-items:start}
 .ev.no{opacity:.5}
 .d{font-size:11px;color:var(--muted)}
 .chip{display:inline-block;font-size:9px;text-transform:uppercase;letter-spacing:.05em;
  padding:1px 5px;border-radius:3px;background:rgba(57,135,229,.16);color:var(--acc);margin-top:2px}
 .chip.x{background:rgba(250,178,25,.15);color:var(--warn)}
 .head{font-size:12px;line-height:1.4}
 .q{font-size:11px;color:var(--ink2);font-style:italic;margin-top:2px;
  border-left:2px solid var(--border);padding-left:7px}
 .no-exp{font-size:11px;color:var(--warn);margin-top:3px}
 .sc{text-align:right;font-variant-numeric:tabular-nums}
 .big{font-size:15px;font-weight:600}.sm{font-size:10px;color:var(--muted)}
 .tog{position:fixed;top:12px;right:14px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
 .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface);
  border:1px solid var(--border);border-radius:7px;padding:8px 11px;font-size:12px;max-width:280px;
  box-shadow:0 8px 26px rgba(0,0,0,.45);z-index:9}
 .tip b{color:var(--ink)}.tip div{color:var(--ink2);margin-top:3px}
 footer{color:var(--muted);font-size:11px;margin-top:20px;line-height:1.7}
</style></head><body>
<button class="tog" onclick="tg()">◐</button><div class="wrap">
<h1>Ticker desk</h1>
<p class="sub">One name at a time: who it shares <em>company-specific</em> risk with, what kind of
news it actually gets, and which of its events survive the gate. Everything is measured against a
random-non-event-day control of <span id="cf"></span>σ.</p>
<div class="bar"><label style="font-size:12px;color:var(--ink2)">ticker
 <select id="sel"></select></label>
 <span style="font-size:11.5px;color:var(--muted)" id="hint"></span></div>
<div class="hd" id="hd"></div>
<div class="grid">
 <div class="card"><h2>News weight breakdown</h2>
  <p class="note">How this name's coverage splits by event class. Bar = share of its events;
  <span style="color:var(--warn)">amber tick</span> = that class's measured risk prior (σ).
  A name covered mostly by low-prior classes is in a different risk regime than one dominated
  by earnings, at identical article counts.</p>
  <div id="mix"></div></div>
 <div class="card"><h2>Idiosyncratic correlation — who moves with it</h2>
  <p class="note">Residual correlation (market factor removed), so a line means shared
  <em>company-specific</em> risk, not two names tracking the index. Thickness = strength.
  <b>Solid lines are cross-sector</b> — the exposures GICS does not encode.</p>
  <div style="text-align:center"><svg id="net" viewBox="0 0 470 330" style="max-width:100%"></svg></div>
 </div></div>
<div class="card"><h2>Event history — gated</h2>
 <p class="note">Ranked by surprise = observed − the class prior. Failures show
 <span style="color:var(--warn)">no explanation</span> and the reason, never a story.</p>
 <div id="ev"></div></div>
<footer><b>Scope.</b> Contemporaneous explanation, not forecasting — every predictive test in this
programme fell below a ~0.002 incremental-R² noise floor. Idiosyncratic = market-model residual;
moves standardised by each name's own trailing 60-day residual vol.</footer>
</div><div class="tip" id="tip"></div>
<script>
const D=__DATA__,tip=document.getElementById('tip');
const PAL=['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8','--s9','--s10'];
const SEC=[...new Set(Object.values(D.tickers).flatMap(t=>[t.sec,...t.nb.map(n=>n.s)]))].sort();
const ci={};SEC.forEach((s,i)=>ci[s]=i);
document.getElementById('cf').textContent=D.ctrl;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function show(e,h){tip.innerHTML=h;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-300)+'px';tip.style.top=(e.clientY+14)+'px'}
function hide(){tip.style.opacity=0}
const keys=Object.keys(D.tickers).sort();
sel.innerHTML=keys.map(k=>`<option>${k}</option>`).join('');
function draw(){
 const k=sel.value,t=D.tickers[k];
 hint.textContent=`${t.n_ev} event-days · ${t.pass} pass the gate`;
 hd.innerHTML=`<span class="t">${k}</span><span class="n">${esc(t.name)} · ${t.sec}</span>
  <span class="kpi"><b>${t.n_ev}</b>event-days</span>
  <span class="kpi"><b>${t.pass}</b>explained</span>
  <span class="kpi"><b>${t.n_ev-t.pass}</b>no explanation</span>
  <span class="kpi"><b>${(t.idio_vol*100).toFixed(1)}%</b>idio vol (ann.)</span>`;
 // A. news weight
 const tot=t.mix.reduce((a,m)=>a+m[1],0)||1,mxp=Math.max(...t.mix.map(m=>m[2]),1.7);
 mix.innerHTML=t.mix.map(([c,n,p])=>`<div class="mixrow"
   onmousemove='show(event,${JSON.stringify(`<b>${c}</b><div>${n} of ${tot} events (${(n/tot*100).toFixed(0)}%)</div><div>class risk prior ${p}σ vs ${D.ctrl}σ normal day</div>`)})'
   onmouseleave="hide()">
  <span class="lab">${esc(c).replace(/_/g,' ')}</span>
  <span class="track"><span class="fill" style="width:${n/tot*100}%;${D.fails.includes(c)?'opacity:.45':''}"></span>
   <span class="ref" style="left:${Math.min(p/mxp*100,99)}%"></span></span>
  <span class="val">${n}</span></div>`).join('')||'<div class="note">no classified events</div>';
 // B. line drawing
 const nb=t.nb,CX=235,CY=163,R=118;let sv='';
 nb.forEach((n,i)=>{const a=(i/Math.max(nb.length,1))*2*Math.PI-Math.PI/2;
  const x=CX+R*Math.cos(a),y=CY+R*Math.sin(a),w=Math.abs(n.c);
  sv+=`<line x1="${CX}" y1="${CY}" x2="${x}" y2="${y}" stroke="var(${PAL[ci[n.s]%10]})"
   stroke-width="${(.6+w*7).toFixed(2)}" opacity="${n.x?0.95:0.4}"
   ${n.x?'':'stroke-dasharray="3,3"'}/>`;
  sv+=`<circle cx="${x}" cy="${y}" r="5.5" fill="var(${PAL[ci[n.s]%10]})"
   onmousemove='show(event,${JSON.stringify(`<b>${n.t}</b><div>${n.s}</div><div>residual corr ${n.c}</div><div>${n.x?'CROSS-sector — not encoded by GICS':'same sector'}</div>`)})' onmouseleave="hide()"/>`;
  const lx=CX+(R+21)*Math.cos(a),ly=CY+(R+21)*Math.sin(a);
  sv+=`<text x="${lx}" y="${ly}" font-size="9.5" fill="var(--ink2)" text-anchor="middle"
   dominant-baseline="middle">${n.t}</text>`});
 sv+=`<circle cx="${CX}" cy="${CY}" r="9" fill="var(--ink)"/>
  <text x="${CX}" y="${CY+22}" font-size="11" fill="var(--ink)" text-anchor="middle" font-weight="600">${k}</text>`;
 net.innerHTML=sv||'';
 // C. events
 const mx=Math.max(1,...t.ev.map(e=>e.obs));
 ev.innerHTML=t.ev.map(e=>`<div class="ev${e.ok?'':' no'}">
  <div><div class="d">${e.d}</div><div class="chip${e.ok?'':' x'}">${esc(e.cls).replace(/_/g,' ')}</div></div>
  <div><div class="head">${esc(e.head||'(no headline captured)')}</div>
   ${e.q?`<div class="q">“${esc(e.q)}”</div>`:''}
   ${e.ok?'':`<div class="no-exp">⚠ no explanation — ${esc(e.why.join('; '))}</div>`}
   <div class="track" style="margin-top:5px;height:6px">
    <div class="fill" style="height:6px;top:0;width:${e.obs/mx*100}%;background:${e.dir==='up'?'var(--up)':'var(--down)'}"></div>
    <div class="ref" style="height:6px;left:${e.exp/mx*100}%"></div></div>
   <div class="sm" style="margin-top:2px">observed ${e.obs}σ · expected ${e.exp}σ</div></div>
  <div class="sc"><div class="big" style="color:${e.sur>0?'var(--ink)':'var(--muted)'}">${e.sur>0?'+':''}${e.sur}σ</div>
   <div class="sm">surprise</div></div></div>`).join('')||'<div class="note">no events</div>';
}
sel.addEventListener('change',draw);draw();
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light'}
</script></body></html>"""


if __name__ == "__main__":
    main()
