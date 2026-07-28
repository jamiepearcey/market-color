# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
THE NEWS DESK — the working surfacing tool, assembling every validated piece.

This is the deliverable the research was for. Everything else measured or
characterised; this SURFACES: real events, real headlines, real verbatim
quotes, scored and filtered by everything we established.

WHAT EACH COLUMN OWES TO WHICH RESULT
  observed        same-day idiosyncratic |move| in the name's own sigma
                  (market factor removed — company news moves the residual)
  expected        the MEASURED prior for that event class, out-of-sample
                  (earnings 1.62 sigma, m_and_a 0.91, election 0.78 ...)
  surprise        observed - expected. A 1.6-sigma move on an earnings day is
                  EXPECTED, not explained; this is the single biggest source of
                  false attribution and the reason raw abnormality ranking is
                  misleading.
  verdict         the ABSTAIN gate. Three ways to fail:
                    - move below the random-day control (0.72 sigma): nothing to
                      explain, however much news exists
                    - event class does not clear the control (election 1.09x
                      move, 0.65x tail): that class is not evidence
                    - surprise <= 0: the move is within class expectation
                  Anything that fails is labelled NO EXPLANATION rather than
                  given a story. This is the anti-hallucination step.
  spillover       other names whose IDIOSYNCRATIC returns correlate with this
                  one — who else is likely carrying the same shock. Residual
                  correlation, so it is not just "same sector, same index".

Filters: sector (GICS), market, event class, unscheduled-only, tape-confirmed,
and "hide abstained". Everything hoverable, every row carries its provenance.

HONEST SCOPE, stated in the UI as well as here: this EXPLAINS, it does not
FORECAST. Every predictive test in the programme came back below an
empirically-calibrated noise floor of ~0.002 incremental R2.

Usage:
    uv run scripts/build_news_desk.py
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
OUT = G / "news_desk.html"
BETA_WIN, TRAIL = 250, 60
TOP_N = 400

# Measured same-day |move| priors, in sigma (news_immediate_risk.py).
PRIOR = {"earnings": 1.621, "employment": 1.386, "debt_issuance": 1.318,
         "econ_indicator": 1.257, "guidance": 1.242, "other": 1.161,
         "legal_regulatory": 1.153, "monetary_policy": 1.118,
         "rating_action": 1.063, "growth": 1.053, "m_and_a": 0.912,
         "credit_event": 0.797, "election": 0.779}
# Classes whose bootstrap CI does NOT clear the control -> not evidence.
FAILS_CONTROL = {"credit_event", "election"}
SCHEDULED = {"earnings", "guidance", "monetary_policy", "employment",
             "econ_indicator", "growth"}


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
        return {}, {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; ind = res["indicators"]; q = ind["quote"][0]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or q["close"]
        vol = q.get("volume") or []
    except Exception:
        return {}, {}
    dd, px, vv = [], [], []
    for t, c, v in zip(ts, cl, vol):
        if c and c > 0:
            dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c)); vv.append(float(v) if v else np.nan)
    rets = {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}
    lv = np.log(np.array([v if v and v > 0 else np.nan for v in vv]))
    vz = {}
    for i in range(TRAIL, len(dd)):
        w = lv[i - TRAIL:i]; w = w[np.isfinite(w)]
        if len(w) > 30 and w.std() > 0 and np.isfinite(lv[i]):
            vz[dd[i]] = float((lv[i] - w.mean()) / w.std())
    return rets, vz


def main() -> None:
    # ---- graph: events with headline + verbatim quote ----
    ent, name_of = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = j.get("resolved_ticker")
        if j.get("resolution_status") == "resolved_security" and tk:
            ent[j["entity_id"]] = tk.upper()
            name_of[tk.upper()] = j.get("name") or tk
    docmeta = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docmeta[j["doc_id"]] = (d, j.get("headline") or "", j.get("url") or "")
    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])

    cells = collections.defaultdict(lambda: {"types": set(), "q": [], "head": [], "url": None})
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        meta = docmeta.get(doc)
        if e not in ent or not meta:
            continue
        d, head, url = meta
        c = cells[(ent[e], d)]
        c["types"] |= (doctypes.get(doc) or {"other"})
        q = (j.get("quote") or "").strip()
        if q and len(c["q"]) < 3:
            c["q"].append(q[:190])
        if head and head not in c["head"] and len(c["head"]) < 2:
            c["head"].append(head[:130])
            c["url"] = c["url"] or url
    print(f"event cells: {len(cells)}")

    # ---- prices -> idiosyncratic residuals ----
    cache = G / "prices"
    rets, vzs = {}, {}
    for tk in sorted({t for t, _ in cells}):
        r, v = load_prices(cache, tk.replace("/", "-"))
        if r and not _tradeable(r):
            continue
        if len(r) > 600:
            rets[tk], vzs[tk] = r, v
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

    # ---- control ----
    rng = np.random.default_rng(11)
    evset = set(cells); tks = list(resid); ctrl = []
    def msig(tk, i):
        e = resid[tk]
        if not np.isfinite(e[i]):
            return None
        w = e[i - TRAIL:i]; w = w[np.isfinite(w)]
        if len(w) < 40 or (np.abs(w) < 1e-12).mean() > MAX_ZERO_FRAC \
            or w.std() < MIN_RESID_VOL:
            return None
        return abs(e[i]) / w.std(), float(e[i] / w.std())
    for _ in range(9000):
        tk = tks[rng.integers(len(tks))]
        i = int(rng.integers(BETA_WIN + TRAIL + 5, len(alldays) - 2))
        if (tk, alldays[i]) in evset:
            continue
        m = msig(tk, i)
        if m:
            ctrl.append(m[0])
    CTRL = float(np.mean(ctrl))
    print(f"control: {CTRL:.3f} sigma")

    # ---- spillover: residual correlation neighbours ----
    keep = [t for t in resid if np.isfinite(resid[t]).sum() > 400]
    M = np.vstack([resid[t] for t in keep]); ok = np.isfinite(M)
    neigh = {}
    for a in range(len(keep)):
        best = []
        for b in range(len(keep)):
            if a == b:
                continue
            m = ok[a] & ok[b]
            if m.sum() < 300:
                continue
            x, y = M[a][m], M[b][m]
            if x.std() > 0 and y.std() > 0:
                c = float(np.corrcoef(x, y)[0, 1])
                if c > 0.12:
                    best.append((round(c, 2), keep[b]))
        best.sort(reverse=True)
        neigh[keep[a]] = best[:4]

    # ---- build rows ----
    rows = []
    for (tk, d), c in cells.items():
        if tk not in resid:
            continue
        i = dayidx.get(d)
        if i is None or i < BETA_WIN + TRAIL + 5 or i + 2 >= len(alldays):
            continue
        m = msig(tk, i)
        if m is None:
            continue
        obs, signed = m
        cls = max(c["types"], key=lambda t: PRIOR.get(t, 1.0))
        exp = PRIOR.get(cls, 1.0)
        surprise = obs - exp
        reasons = []
        if obs < CTRL:
            reasons.append("move is within the random-day baseline")
        if cls in FAILS_CONTROL:
            reasons.append(f"'{cls}' does not clear the control")
        if surprise <= 0:
            reasons.append("move is within expectation for this class")
        rows.append({
            "tk": tk, "nm": name_of.get(tk, tk), "d": d, "cls": cls,
            "sec": gics_sector(tk), "obs": round(obs, 2), "exp": round(exp, 2),
            "sur": round(surprise, 2), "dir": "up" if signed > 0 else "down",
            "vz": round(vzs.get(tk, {}).get(d, float("nan")), 2)
                  if np.isfinite(vzs.get(tk, {}).get(d, float("nan"))) else None,
            "head": c["head"], "q": c["q"], "url": c["url"],
            "unsched": cls not in SCHEDULED,
            "ok": not reasons, "why": reasons,
            "nb": neigh.get(tk, []),
        })
    rows.sort(key=lambda r: -r["sur"])
    rows = rows[:TOP_N]
    kept = sum(1 for r in rows if r["ok"])
    print(f"rows: {len(rows)} surfaced, {kept} pass the gate, {len(rows)-kept} abstained")

    payload = {"rows": rows, "ctrl": round(CTRL, 3),
               "prior": {k: round(v, 2) for k, v in PRIOR.items()},
               "fails": sorted(FAILS_CONTROL)}
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News desk — what moved, and whether the news explains it</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --border:rgba(255,255,255,.10);--up:#0ca30c;--down:#e66767;--warn:#fab219;
  --acc:#3987e5;--dim:#1c5cab}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --border:rgba(11,11,11,.10);--up:#006300;--down:#e34948;--acc:#2a78d6;--dim:#9ec5f4}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1240px;margin:0 auto;padding:22px 18px 60px}
 h1{font-size:19px;margin:0 0 3px}
 .sub{color:var(--ink2);font-size:12.5px;margin:0 0 14px;max-width:82ch}
 .bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;padding:10px 12px;margin-bottom:12px}
 select,input{background:var(--page);color:var(--ink);border:1px solid var(--border);
  border-radius:6px;padding:5px 8px;font-size:12px}
 label{font-size:11.5px;color:var(--ink2);display:inline-flex;align-items:center;gap:5px}
 .stat{display:flex;gap:20px;margin:0 0 12px;font-size:12px;color:var(--ink2)}
 .stat b{color:var(--ink);font-size:16px;font-variant-numeric:tabular-nums}
 .ev{background:var(--surface);border:1px solid var(--border);border-radius:9px;
  padding:11px 13px;margin-bottom:7px;display:grid;
  grid-template-columns:86px 1fr 150px;gap:13px;align-items:start}
 .ev.no{opacity:.5}
 .tick{font-weight:600;font-size:13px}
 .meta{font-size:10.5px;color:var(--muted);margin-top:1px}
 .chip{display:inline-block;font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;
  padding:1px 5px;border-radius:3px;background:rgba(57,135,229,.16);color:var(--acc);margin-top:3px}
 .chip.x{background:rgba(250,178,25,.15);color:var(--warn)}
 .head{font-size:12.8px;line-height:1.45}
 .q{font-size:11.5px;color:var(--ink2);font-style:italic;margin-top:3px;
  border-left:2px solid var(--border);padding-left:8px}
 .no-exp{font-size:11.5px;color:var(--warn)}
 .sc{text-align:right;font-variant-numeric:tabular-nums}
 .big{font-size:17px;font-weight:600}
 .sm{font-size:10.5px;color:var(--muted)}
 .nb{font-size:10.5px;color:var(--ink2);margin-top:4px}
 .track{position:relative;height:6px;background:rgba(128,128,128,.2);border-radius:3px;margin-top:5px}
 .fill{position:absolute;height:6px;border-radius:3px;background:var(--acc)}
 .ref{position:absolute;top:-2px;width:2px;height:10px;background:var(--warn)}
 .tog{position:fixed;top:12px;right:14px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
 footer{color:var(--muted);font-size:11.5px;margin-top:22px;line-height:1.7}
 code{background:rgba(128,128,128,.15);padding:1px 5px;border-radius:4px;font-size:11px}
</style></head><body>
<button class="tog" onclick="tg()">◐</button><div class="wrap">
<h1>News desk — what moved, and whether the news explains it</h1>
<p class="sub">Every row is a real company-day with real news. Scored by
<b>surprise = observed move − what that class of news normally does</b>, so an
ordinary earnings reaction ranks below a genuinely strange one. Rows that fail the
gate are shown as <span style="color:var(--warn)">no explanation</span> rather than
given a story.</p>
<div class="bar">
 <label>sector <select id="fs"></select></label>
 <label>class <select id="fc"></select></label>
 <label><input type="checkbox" id="fu"> unscheduled only</label>
 <label><input type="checkbox" id="fg" checked> hide abstained</label>
 <label>search <input id="fq" placeholder="ticker or headline" size="18"></label>
 <span style="margin-left:auto;font-size:11px;color:var(--muted)" id="cnt"></span>
</div>
<div class="stat" id="stat"></div>
<div id="list"></div>
<footer><b>Scope.</b> This explains; it does not forecast. Every predictive test in this
programme came back below an empirically-calibrated noise floor (~0.002 incremental R²,
linear and non-linear). Control = <span id="cf"></span>σ, the mean idiosyncratic move on a
random non-event day. Classes that fail the control (<span id="fl"></span>) can never
support an explanation.<br>
Built from <code>news_immediate_risk.py</code> priors · <code>classification/</code> ·
verbatim quotes from the causal graph.</footer>
</div>
<script>
const D=__DATA__;
document.getElementById('cf').textContent=D.ctrl;
document.getElementById('fl').textContent=D.fails.join(', ');
const secs=[...new Set(D.rows.map(r=>r.sec))].sort();
const cls=[...new Set(D.rows.map(r=>r.cls))].sort();
document.getElementById('fs').innerHTML='<option value="">all</option>'+secs.map(s=>`<option>${s}</option>`).join('');
document.getElementById('fc').innerHTML='<option value="">all</option>'+cls.map(s=>`<option>${s}</option>`).join('');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function draw(){
 const s=fs.value,c=fc.value,u=fu.checked,g=fg.checked,q=fq.value.toLowerCase();
 const rs=D.rows.filter(r=>(!s||r.sec===s)&&(!c||r.cls===c)&&(!u||r.unsched)&&(!g||r.ok)
   &&(!q||r.tk.toLowerCase().includes(q)||(r.head[0]||'').toLowerCase().includes(q)));
 cnt.textContent=`${rs.length} of ${D.rows.length} events`;
 const pass=rs.filter(r=>r.ok).length;
 stat.innerHTML=`<span><b>${pass}</b> explained</span>
  <span><b>${rs.length-pass}</b> no explanation</span>
  <span><b>${rs.length?(rs.reduce((a,r)=>a+r.obs,0)/rs.length).toFixed(2):'—'}</b>σ mean move</span>
  <span><b>${D.ctrl}</b>σ normal day</span>`;
 const mx=Math.max(1,...rs.map(r=>r.obs));
 list.innerHTML=rs.map(r=>`<div class="ev${r.ok?'':' no'}">
  <div><div class="tick">${r.tk}</div><div class="meta">${r.d}</div>
   <div class="meta">${r.sec}</div><div class="chip${r.ok?'':' x'}">${esc(r.cls).replace(/_/g,' ')}</div></div>
  <div><div class="head">${esc(r.head[0]||'(no headline captured)')}</div>
   ${r.q.length?`<div class="q">“${esc(r.q[0])}”</div>`:''}
   ${r.ok?'':`<div class="no-exp" style="margin-top:5px">⚠ no explanation — ${esc(r.why.join('; '))}</div>`}
   ${r.nb.length?`<div class="nb">also exposed: ${r.nb.map(n=>`${n[1]} <span class="sm">${n[0]}</span>`).join(' · ')}</div>`:''}
   <div class="track"><div class="fill" style="width:${r.obs/mx*100}%;background:${r.dir==='up'?'var(--up)':'var(--down)'}"></div>
    <div class="ref" style="left:${r.exp/mx*100}%"></div></div>
   <div class="sm" style="margin-top:3px">observed ${r.obs}σ · expected for ${esc(r.cls).replace(/_/g,' ')} ${r.exp}σ (amber)</div></div>
  <div class="sc"><div class="big" style="color:${r.sur>0?'var(--ink)':'var(--muted)'}">${r.sur>0?'+':''}${r.sur}σ</div>
   <div class="sm">surprise</div>
   ${r.vz!==null?`<div class="sm" style="margin-top:5px">volume ${r.vz>0?'+':''}${r.vz}z</div>`:''}
   ${r.url?`<div class="sm" style="margin-top:4px"><a href="${esc(r.url)}" style="color:var(--acc)">source ↗</a></div>`:''}</div>
 </div>`).join('')||'<div style="color:var(--muted);padding:30px;text-align:center">no events match</div>';
}
[fs,fc,fu,fg,fq].forEach(el=>el.addEventListener('input',draw));draw();
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light'}
</script></body></html>"""


if __name__ == "__main__":
    main()
