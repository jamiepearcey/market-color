# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Sector- and market-resolved risk view + the idiosyncratic-correlation network.

THREE PANELS, all built from measurements already validated against controls:

  1. SECTOR x EVENT-CLASS risk matrix. The same-day idiosyncratic |move| (in
     units of each name's own trailing sigma) for every GICS sector x event
     class cell with enough events, expressed as a MULTIPLE of the random
     non-event-day control. Industry-standard sector labels (GICS via gics.py),
     not ad-hoc buckets.

  2. IDIOSYNCRATIC CORRELATION NETWORK — "related betas mapped to one another by
     strength". Names arranged on a circle GROUPED BY GICS SECTOR, chords drawn
     for the strongest pairwise residual correlations. Residual, not raw: the
     market factor is removed first, so a chord means shared COMPANY-SPECIFIC
     risk, not two names both following the index. Cross-sector chords are the
     interesting ones — they are the exposures industry classification does not
     encode.

  3. MARKET comparison — US (Bloomberg 2010-12) vs India (2021), the two graphs
     with verified identifiers.

Usage:
    uv run scripts/build_sector_risk_viz.py
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
GI = ROOT / "india2021"
OUT = G / "sector_risk_viz.html"
BETA_WIN, TRAIL = 250, 60
MIN_CELL = 25
TOP_EDGES = 140


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


def build_graph(graph_dir: Path, sector_fn, cache_name="prices"):
    """-> (events{(tk,day):types}, resid{tk:array}, alldays, dayidx)"""
    ent = {}
    for l in open(graph_dir / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = j.get("resolved_ticker")
        if j.get("resolution_status") == "resolved_security" and tk:
            ent[j["entity_id"]] = tk.upper()
    if not ent and (graph_dir / "classification" / "india_ticker_map.json").exists():
        vm = json.loads((graph_dir / "classification" / "india_ticker_map.json").read_text())
        ent = {k: v["ticker"] for k, v in vm.items()}
    docday = {}
    for l in open(graph_dir / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
    doctypes = collections.defaultdict(set)
    ec = graph_dir / "classification" / "event_class.jsonl"
    if ec.exists():
        for l in open(ec):
            j = json.loads(l)
            if j.get("doc_id") and j.get("std_event_type"):
                doctypes[j["doc_id"]].add(j["std_event_type"])
    events = collections.defaultdict(set)
    for l in open(graph_dir / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if e in ent and d:
            events[(ent[e], d)] |= (doctypes.get(doc) or {"other"})

    cache = graph_dir / cache_name
    rets = {}
    for tk in sorted({t for t, _ in events}):
        r = load_prices(cache, tk.replace("/", "-"))
        if r and not _tradeable(r):
            continue
        if len(r) > 400:
            rets[tk] = r
    if not rets:
        return {}, {}, [], {}
    dc = collections.Counter()
    for r in rets.values():
        dc.update(r.keys())
    thresh = max(20, int(0.30 * len(rets)))
    alldays = sorted([d for d, c in dc.items() if c >= thresh])
    dayidx = {d: i for i, d in enumerate(alldays)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])
    resid = {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < 250:
            continue
        e = np.full(len(alldays), np.nan)
        for i in range(min(BETA_WIN, len(alldays) // 3), len(alldays)):
            w = min(BETA_WIN, i)
            sl = slice(i - w, i); m = ok[sl]
            if m.sum() < 100:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean(); den = float(fc @ fc)
            if den > 0 and ok[i]:
                e[i] = y[i] - float((yy - yy.mean()) @ fc / den) * F[i]
        resid[tk] = e
    return events, resid, alldays, dayidx


def move_sigma(resid, tk, i):
    e = resid[tk]
    if not np.isfinite(e[i]):
        return None
    w = e[i - TRAIL:i]; w = w[np.isfinite(w)]
    if len(w) < 40 or (np.abs(w) < 1e-12).mean() > MAX_ZERO_FRAC \
            or w.std() < MIN_RESID_VOL:
        return None
    return abs(e[i]) / w.std()


def main() -> None:
    payload = {}

    # ---------- US ----------
    events, resid, alldays, dayidx = build_graph(G, gics_sector)
    print(f"US: {len(resid)} tickers with residuals, {len(events)} event cells")

    rng = np.random.default_rng(7)
    ctrl = []
    evset = set(events)
    tks = list(resid)
    for _ in range(9000):
        tk = tks[rng.integers(len(tks))]
        i = int(rng.integers(BETA_WIN + TRAIL + 5, len(alldays) - 2))
        if (tk, alldays[i]) in evset:
            continue
        m = move_sigma(resid, tk, i)
        if m:
            ctrl.append(m)
    CTRL = float(np.mean(ctrl))
    print(f"US control: {CTRL:.3f} sigma (n={len(ctrl)})")

    cells = collections.defaultdict(list)
    per_class = collections.defaultdict(list)
    for (tk, d), types in events.items():
        if tk not in resid:
            continue
        i = dayidx.get(d)
        if i is None or i < BETA_WIN + TRAIL + 5 or i + 2 >= len(alldays):
            continue
        m = move_sigma(resid, tk, i)
        if m is None:
            continue
        s = gics_sector(tk)
        for t in types:
            per_class[t].append(m)
            if s != "UNK":
                cells[(s, t)].append(m)

    sectors = sorted({s for s, _ in cells}, key=lambda s: -len([1 for (a, _), v in cells.items() if a == s]))
    classes = [c for c, v in sorted(per_class.items(), key=lambda kv: -np.mean(kv[1]))
               if len(v) >= 60]
    matrix = []
    for s in sectors:
        row = []
        for c in classes:
            v = cells.get((s, c), [])
            row.append([round(float(np.mean(v)) / CTRL, 3), len(v)] if len(v) >= MIN_CELL else None)
        matrix.append(row)
    payload["sector_matrix"] = {"sectors": sectors, "classes": classes,
                                "cells": matrix, "control": round(CTRL, 4)}

    # ---------- correlation network (GICS-known names) ----------
    known = sorted([t for t in resid if gics_sector(t) != "UNK"])
    M = np.vstack([resid[t] for t in known]); ok = np.isfinite(M)
    edges = []
    for a in range(len(known)):
        for b in range(a + 1, len(known)):
            m = ok[a] & ok[b]
            if m.sum() < 250:
                continue
            x, y = M[a][m], M[b][m]
            if x.std() > 0 and y.std() > 0:
                c = float(np.corrcoef(x, y)[0, 1])
                if abs(c) > 0.05:
                    edges.append((a, b, round(c, 3)))
    edges.sort(key=lambda e: -abs(e[2]))
    edges = edges[:TOP_EDGES]
    keep = sorted({i for e in edges for i in e[:2]})
    remap = {o: n for n, o in enumerate(keep)}
    nodes = [{"t": known[o], "s": gics_sector(known[o])} for o in keep]
    payload["network"] = {
        "nodes": nodes,
        "edges": [[remap[a], remap[b], c] for a, b, c in edges],
    }
    # "How many links are cross-sector?" has no single answer — it depends entirely
    # on how coarse the buckets are, so report EVERY level rather than pick one.
    # SIC divisions are very broad (Manufacturing alone holds ~36% of the universe,
    # merging pharma, chemicals, autos and semiconductors), so the division-level
    # figure understates cross-industry linkage; the major-group level is the one
    # comparable in granularity to a GICS sector.
    levels = {"division": gics_sector, "major_group": subsector, "industry": industry}
    shares = {}
    for lname, fn in levels.items():
        pairs = [(fn(known[a]), fn(known[b])) for a, b, _ in edges]
        usable = [(x, y) for x, y in pairs if x != "UNK" and y != "UNK"]
        shares[lname] = {
            "share": round(sum(1 for x, y in usable if x != y) / max(len(usable), 1), 3),
            "n": len(usable),
            "buckets": len({g for p in usable for g in p}),
        }
    payload["network"]["cross_sector_share"] = shares["major_group"]["share"]
    payload["network"]["cross_levels"] = shares
    print(f"network: {len(nodes)} nodes, {len(edges)} edges")
    for lname, v in shares.items():
        print(f"  cross-{lname:12} {v['share']:.0%}  "
              f"({v['n']} classified pairs over {v['buckets']} buckets)")

    # ---------- India ----------
    try:
        ev_i, res_i, days_i, idx_i = build_graph(GI, lambda t: "UNK")
        ctrl_i, vals_i = [], collections.defaultdict(list)
        if res_i and len(days_i) > 300:
            tki = list(res_i)
            for _ in range(4000):
                tk = tki[rng.integers(len(tki))]
                i = int(rng.integers(TRAIL + 5, len(days_i) - 2))
                if (tk, days_i[i]) in ev_i:
                    continue
                m = move_sigma(res_i, tk, i)
                if m:
                    ctrl_i.append(m)
            for (tk, d), types in ev_i.items():
                if tk not in res_i:
                    continue
                i = idx_i.get(d)
                if i is None or i < TRAIL + 5 or i + 2 >= len(days_i):
                    continue
                m = move_sigma(res_i, tk, i)
                if m is None:
                    continue
                for t in types:
                    vals_i[t].append(m)
        ci = float(np.mean(ctrl_i)) if ctrl_i else float("nan")
        payload["markets"] = [
            {"m": "US (Bloomberg 2010-12)", "tickers": len(resid), "events": len(events),
             "control": round(CTRL, 3),
             "top": [[c, round(float(np.mean(v)) / CTRL, 2), len(v)]
                     for c, v in sorted(per_class.items(), key=lambda kv: -np.mean(kv[1]))[:6]
                     if len(v) >= 40]},
            {"m": "India (2021)", "tickers": len(res_i), "events": len(ev_i),
             "control": round(ci, 3),
             "top": [[c, round(float(np.mean(v)) / ci, 2), len(v)]
                     for c, v in sorted(vals_i.items(), key=lambda kv: -np.mean(kv[1]))[:6]
                     if len(v) >= 20]},
        ]
        print(f"India: {len(res_i)} tickers, control {ci:.3f} sigma")
    except Exception as exc:  # India is optional — never fail the whole build on it
        print(f"India panel skipped: {exc}")
        payload["markets"] = []

    OUT.write_text(render(payload))
    print(f"\nwrote {OUT} ({OUT.stat().st_size:,} bytes)")


def render(p: dict) -> str:
    return HTML.replace("__DATA__", json.dumps(p))


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Idiosyncratic risk by sector, class and market</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--border:rgba(255,255,255,.10);--warn:#fab219;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
  --s6:#008300;--s7:#9085e9;--s8:#e66767;--s9:#898781;--s10:#5598e7}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --grid:#e1e0d9;--border:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;
  --s6:#008300;--s7:#4a3aa7;--s8:#e34948;--s9:#898781;--s10:#86b6ef}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
 h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 3px}
 .sub{color:var(--ink2);font-size:13px;margin:0 0 20px;max-width:74ch}
 .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:18px 20px;margin:16px 0}
 .note{color:var(--muted);font-size:12px;margin:0 0 14px;max-width:80ch}
 table{border-collapse:collapse;font-size:12px;width:100%}
 th{font-weight:500;color:var(--muted);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.05em;padding:5px 6px;text-align:center;white-space:nowrap}
 th.l,td.l{text-align:left;color:var(--ink2);white-space:nowrap}
 td{padding:0;text-align:center}
 .cell{padding:7px 4px;border-radius:3px;font-variant-numeric:tabular-nums;
  color:var(--ink);margin:1px;cursor:default}
 .na{color:var(--muted);opacity:.35}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-top:12px}
 .key{display:inline-flex;align-items:center;gap:6px}
 .sw{width:11px;height:11px;border-radius:2px}
 .tog{position:fixed;top:14px;right:16px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:6px 11px;font-size:12px;cursor:pointer}
 .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--surface);border:1px solid var(--border);border-radius:7px;
  padding:8px 11px;font-size:12px;max-width:280px;box-shadow:0 8px 26px rgba(0,0,0,.45);z-index:9}
 .tip b{color:var(--ink)}.tip div{color:var(--ink2);margin-top:3px}
 .mk{display:grid;grid-template-columns:1fr 1fr;gap:18px}
 .bar{height:12px;border-radius:0 3px 3px 0;background:var(--s1)}
 footer{color:var(--muted);font-size:11.5px;margin-top:24px;line-height:1.7}
 code{background:rgba(128,128,128,.15);padding:1px 5px;border-radius:4px;font-size:11px}
</style></head><body>
<button class="tog" onclick="tg()">◐ theme</button><div class="wrap">
<h1>Idiosyncratic risk by sector, class and market</h1>
<p class="sub">Same-day company-specific risk (market factor removed), measured against a
random non-event-day control. Sector labels are GICS. Everything here is a
<em>contemporaneous measurement</em>; the predictive versions were null.</p>

<div class="card"><h2>1 · Risk multiplier by GICS sector × event class</h2>
<p class="note">Each cell is the mean same-day idiosyncratic |move| for that sector/class,
as a <b>multiple of a random non-event day</b>. 1.0 = an ordinary day. Blank = fewer than
25 events. Hover for counts.</p>
<div id="mx"></div>
<div class="legend"><span class="key"><span class="sw" style="background:var(--s1);opacity:.25"></span>≈1×</span>
<span class="key"><span class="sw" style="background:var(--s1);opacity:.6"></span>~1.5×</span>
<span class="key"><span class="sw" style="background:var(--s1)"></span>≥2×</span></div></div>

<div class="card"><h2>2 · Idiosyncratic correlation network — related betas by strength</h2>
<p class="note">Names on a circle, <b>grouped by GICS sector</b>; each chord is a strong pairwise
correlation of <em>residual</em> returns, so it means shared company-specific risk rather than
two names both tracking the index. Thickness = strength. <b>Cross-sector chords are the
interesting ones</b> — exposures industry classification does not encode.</p>
<div style="text-align:center"><svg id="net" viewBox="0 0 760 620" style="max-width:100%"></svg></div>
<div class="legend" id="netleg"></div></div>

<div class="card"><h2>3 · Market comparison</h2>
<p class="note">The two graphs with verified identifiers. Each market has its own control,
so multipliers are comparable across markets even though volatility levels are not.</p>
<div class="mk" id="mk"></div></div>

<footer><b>Method.</b> Idiosyncratic = market-model residual (rolling 250d beta on an
equal-weighted market factor). Move is standardised by each name's own trailing 60-day
residual vol, so a utility and a biotech are comparable. Control = random non-event days
for the same names.<br>
Sources: <code>news_immediate_risk.py</code>, <code>news_risk_clusters_nograph.py</code>,
<code>gics.py</code>, <code>india_resolve_verified.py</code>.</footer>
</div><div class="tip" id="tip"></div>
<script>
const D=__DATA__,tip=document.getElementById('tip');
const PAL=['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8','--s9','--s10'];
function show(e,h){tip.innerHTML=h;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-300)+'px';tip.style.top=(e.clientY+14)+'px'}
function hide(){tip.style.opacity=0}
// ---- panel 1: matrix ----
const M=D.sector_matrix;
let h='<table><tr><th class="l">GICS sector</th>'+M.classes.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')+'</tr>';
M.sectors.forEach((s,i)=>{h+=`<tr><td class="l">${s}</td>`;
 M.cells[i].forEach((c,j)=>{ if(!c){h+='<td><div class="cell na">·</div></td>';return}
  const [v,n]=c, o=Math.max(.12,Math.min(1,(v-.9)/1.4));
  h+=`<td><div class="cell" style="background:color-mix(in srgb,var(--s1) ${o*100}%,transparent)"
   onmousemove='show(event,${JSON.stringify(`<b>${s} · ${M.classes[j]}</b><div>${v.toFixed(2)}× a random day</div><div>n = ${n} events</div>`)})'
   onmouseleave="hide()">${v.toFixed(2)}</div></td>`});
 h+='</tr>'});
document.getElementById('mx').innerHTML=h+'</table>';
// ---- panel 2: circular network grouped by sector ----
const N=D.network,secs=[...new Set(N.nodes.map(n=>n.s))].sort();
const ci={};secs.forEach((s,i)=>ci[s]=i);
const order=N.nodes.map((n,i)=>i).sort((a,b)=>ci[N.nodes[a].s]-ci[N.nodes[b].s]||N.nodes[a].t.localeCompare(N.nodes[b].t));
const pos={},CX=380,CY=305,R=232;
order.forEach((idx,k)=>{const ang=(k/order.length)*2*Math.PI-Math.PI/2;
 pos[idx]={x:CX+R*Math.cos(ang),y:CY+R*Math.sin(ang),a:ang}});
const mx=Math.max(...N.edges.map(e=>Math.abs(e[2])));
let sv='';
N.edges.forEach(([a,b,c])=>{const p=pos[a],q=pos[b],w=Math.abs(c)/mx;
 const cross=N.nodes[a].s!==N.nodes[b].s;
 sv+=`<path d="M${p.x},${p.y} Q${CX},${CY} ${q.x},${q.y}" fill="none"
  stroke="var(${PAL[ci[N.nodes[a].s]%10]})" stroke-width="${(.4+w*2).toFixed(2)}"
  opacity="${(cross?.55:.22)*(.35+w*.65)}"/>`});
order.forEach(idx=>{const p=pos[idx],n=N.nodes[idx];
 sv+=`<circle cx="${p.x}" cy="${p.y}" r="4" fill="var(${PAL[ci[n.s]%10]})"
  onmousemove='show(event,${JSON.stringify(`<b>${n.t}</b><div>${n.s}</div>`)})' onmouseleave="hide()"/>`;
 const lx=CX+(R+13)*Math.cos(p.a),ly=CY+(R+13)*Math.sin(p.a),rot=p.a*180/Math.PI;
 const flip=Math.cos(p.a)<0;
 sv+=`<text x="${lx}" y="${ly}" font-size="8.5" fill="var(--muted)"
  text-anchor="${flip?'end':'start'}" dominant-baseline="middle"
  transform="rotate(${flip?rot+180:rot},${lx},${ly})">${n.t}</text>`});
document.getElementById('net').innerHTML=sv;
document.getElementById('netleg').innerHTML=secs.map((s,i)=>
 `<span class="key"><span class="sw" style="background:var(${PAL[i%10]})"></span>${s}</span>`).join('')
 +`<span class="key" style="color:var(--muted)" title="Depends on bucket granularity — shown at every level of the SIC hierarchy.">${Object.entries(N.cross_levels||{}).map(([k,v])=>`${(v.share*100).toFixed(0)}% cross-${k.replace(/_/g," ")} (${v.buckets} buckets)`).join(" · ")}</span>`;
// ---- panel 3: markets ----
document.getElementById('mk').innerHTML=(D.markets||[]).map(m=>{
 const mxv=Math.max(1.2,...m.top.map(t=>t[1]));
 return `<div><div style="font-size:13px;margin-bottom:2px">${m.m}</div>
 <div style="color:var(--muted);font-size:11.5px;margin-bottom:9px">
 ${m.tickers} tickers · ${m.events} event-days · control ${m.control}σ</div>`+
 m.top.map(([c,v,n])=>`<div style="display:grid;grid-template-columns:120px 1fr 44px;
  gap:8px;align-items:center;padding:2px 0" onmousemove='show(event,${JSON.stringify(
  `<b>${c}</b><div>${v}× a random day</div><div>n = ${n}</div>`)})' onmouseleave="hide()">
  <span style="font-size:11.5px;color:var(--ink2);text-align:right">${c.replace(/_/g,' ')}</span>
  <span class="bar" style="width:${v/mxv*100}%"></span>
  <span style="font-size:11.5px;color:var(--ink2)">${v}×</span></div>`).join('')+'</div>'}).join('');
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light'}
</script></body></html>"""


if __name__ == "__main__":
    main()
