# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Build a self-contained HTML visualisation of the news->risk findings.

Renders the two results that survived scrutiny, each against its control:

  1. IMMEDIATE RISK BY EVENT CLASS (news_immediate_risk.py) — same-day
     idiosyncratic |move| in sigma and >2-sigma tail rate, per event class,
     against a random-non-event-day control. This is the strongest positive
     finding in the programme: 11 of 13 classes clear the control, earnings
     carries a 4.7x jump-risk multiplier.

  2. RISK CLUSTERING (news_risk_clusters_nograph.py + news_riskweighted_
     clusters.py) — within-cluster idiosyncratic correlation for news-embedding
     clusters vs GICS sectors vs random, on all days vs high-risk event windows.
     Shows both that risk-weighting and event-conditioning help, AND that GICS
     still wins — the honest negative alongside the positive.

Design follows the project's data-viz rules: magnitude -> single sequential hue
(not categorical, these are one measure across categories); the control is a
REFERENCE LINE, not another series, because it is a baseline not a comparison
member; values in text tokens never series colour; dark mode is a selected
palette rather than an inverted one; hover tooltips throughout; a table view is
inherent since every figure is labelled.

Usage:
    uv run scripts/build_risk_viz.py     # writes <graph>/news_risk_viz.html
"""
from __future__ import annotations

import json
from pathlib import Path

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
OUT = G / "news_risk_viz.html"

# --- measured values (scripts/news_immediate_risk.py) ---
CONTROL_MOVE, CONTROL_TAIL = 0.712, 0.055
RISK_ROWS = [
    # class, n, |move| sigma, tail rate, vol-z, clears control (bootstrap CI)
    ("earnings", 448, 1.621, 0.259, 1.051, True),
    ("employment", 64, 1.386, 0.203, 0.784, True),
    ("debt_issuance", 53, 1.318, 0.208, 0.454, True),
    ("econ_indicator", 41, 1.257, 0.171, 0.614, True),
    ("guidance", 436, 1.242, 0.181, 0.480, True),
    ("other", 1041, 1.161, 0.157, 0.417, True),
    ("legal_regulatory", 116, 1.153, 0.155, 0.525, True),
    ("monetary_policy", 316, 1.118, 0.158, 0.455, True),
    ("rating_action", 181, 1.063, 0.166, 0.351, True),
    ("growth", 72, 1.053, 0.153, 0.630, True),
    ("m_and_a", 1120, 0.912, 0.093, 0.274, True),
    ("credit_event", 45, 0.797, 0.089, -0.062, False),
    ("election", 84, 0.779, 0.036, 0.226, False),
]

# --- clustering (news_riskweighted_clusters.py, matched universe n=78) ---
CLUSTER_ROWS = [
    ("Clusters: high-risk news", 0.0591, 0.0938),
    ("Clusters: all news", 0.0564, 0.0782),
    ("GICS sectors", 0.0893, 0.1059),
    ("Random (same sizes)", 0.0016, -0.0046),
]

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>News → risk: what the data actually shows</title>
<style>
  :root {
    --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --seq:#3987e5; --seq-dim:#1c5cab; --warn:#fab219; --good:#0ca30c; --crit:#d03b3b;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#898781;
  }
  html[data-theme="light"] {
    --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
    --seq:#2a78d6; --seq-dim:#9ec5f4;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#898781;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--page); color:var(--ink);
         font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }
  .wrap { max-width:1080px; margin:0 auto; padding:28px 22px 60px }
  h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.01em }
  .sub { color:var(--ink2); font-size:13px; margin:0 0 22px; max-width:70ch }
  .card { background:var(--surface); border:1px solid var(--border);
          border-radius:10px; padding:18px 20px; margin:18px 0 }
  h2 { font-size:15px; margin:0 0 2px }
  .note { color:var(--muted); font-size:12px; margin:0 0 16px; max-width:78ch }
  .row { display:grid; grid-template-columns:150px 1fr 62px 1fr 54px;
         align-items:center; gap:10px; padding:3px 0 }
  .row:hover { background:rgba(128,128,128,.07) }
  .lab { font-size:12.5px; color:var(--ink2); text-align:right;
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
  .val { font-size:12px; color:var(--ink2); font-variant-numeric:tabular-nums }
  .track { position:relative; height:17px; background:transparent }
  .bar { position:absolute; left:0; top:2px; height:13px; border-radius:0 4px 4px 0 }
  .ref { position:absolute; top:0; width:2px; height:17px; background:var(--warn) }
  .hdr { display:grid; grid-template-columns:150px 1fr 62px 1fr 54px; gap:10px;
         font-size:10.5px; text-transform:uppercase; letter-spacing:.06em;
         color:var(--muted); padding-bottom:7px; margin-bottom:5px;
         border-bottom:1px solid var(--border) }
  .legend { display:flex; gap:16px; flex-wrap:wrap; font-size:11.5px;
            color:var(--ink2); margin-top:14px }
  .key { display:inline-flex; align-items:center; gap:6px }
  .sw { width:11px; height:11px; border-radius:2px; display:inline-block }
  .gr { display:grid; grid-template-columns:190px 1fr 1fr; gap:12px;
        align-items:center; padding:5px 0 }
  .tog { position:fixed; top:14px; right:16px; background:var(--surface);
         color:var(--ink2); border:1px solid var(--border); border-radius:7px;
         padding:6px 11px; font-size:12px; cursor:pointer }
  .tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
         background:var(--surface); border:1px solid var(--border);
         border-radius:7px; padding:8px 11px; font-size:12px; max-width:290px;
         box-shadow:0 8px 26px rgba(0,0,0,.4); z-index:9 }
  .tip b { color:var(--ink) } .tip div { color:var(--ink2); margin-top:3px }
  footer { color:var(--muted); font-size:11.5px; margin-top:26px; line-height:1.7 }
  code { background:rgba(128,128,128,.15); padding:1px 5px; border-radius:4px;
         font-size:11px }
</style></head><body>
<button class="tog" onclick="tog()">◐ theme</button>
<div class="wrap">
  <h1>News → risk: what the data actually shows</h1>
  <p class="sub">Two surviving results from a programme in which every
  <em>predictive</em> test came back null. Both are contemporaneous measurements,
  each shown against the control that makes it interpretable.</p>

  <div class="card">
    <h2>1 · Immediate risk by news topic</h2>
    <p class="note">Same-day idiosyncratic move (market-model residual, in units of
    the name's own trailing volatility) and the probability of a &gt;2σ move, by
    event class. The <span style="color:var(--warn)">amber line</span> is a
    random non-event day — the number that says whether a bar is actually large.
    Hover any row.</p>
    <div class="hdr"><span style="text-align:right">event class</span>
      <span>same-day |move| (σ)</span><span>×ctrl</span>
      <span>P(&gt;2σ move)</span><span>×ctrl</span></div>
    <div id="risk"></div>
    <div class="legend">
      <span class="key"><span class="sw" style="background:var(--seq)"></span>
        clears control (bootstrap 95% CI)</span>
      <span class="key"><span class="sw" style="background:var(--seq-dim)"></span>
        does not clear</span>
      <span class="key"><span class="sw" style="background:var(--warn);width:3px"></span>
        random-day control</span>
    </div>
  </div>

  <div class="card">
    <h2>2 · Does news clustering group companies by shared risk?</h2>
    <p class="note">Mean within-group correlation of <em>idiosyncratic</em>
    returns (market factor removed), on a matched universe of 78 names. Higher =
    the grouping really does capture shared company-specific risk. Two takeaways:
    risk-weighting and event-conditioning both help — and GICS still wins.</p>
    <div class="gr" style="font-size:10.5px;text-transform:uppercase;
         letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border);
         padding-bottom:7px"><span></span><span>all days</span>
         <span>high-risk event windows</span></div>
    <div id="clu"></div>
    <div class="legend">
      <span class="key"><span class="sw" style="background:var(--s1)"></span>news embedding</span>
      <span class="key"><span class="sw" style="background:var(--s2)"></span>GICS sectors</span>
      <span class="key"><span class="sw" style="background:var(--s4)"></span>random baseline</span>
    </div>
  </div>

  <footer>
    <b>Read with care.</b> These are <em>contemporaneous measurements</em>, not
    forecasts — risk being realised as the news lands. Every attempt to predict
    beyond the price was null: incremental R² below an empirically-calibrated
    noise floor of ~0.002, linear and non-linear alike.<br>
    Sources: <code>news_immediate_risk.py</code>,
    <code>news_risk_clusters_nograph.py</code>,
    <code>news_riskweighted_clusters.py</code> · Bloomberg 2010-12, 670 US names
    with resolved identifiers.
  </footer>
</div>
<div class="tip" id="tip"></div>
<script>
const RISK = __RISK__, CLU = __CLU__, CTRL = __CTRL__, CTAIL = __CTAIL__;
const tip = document.getElementById('tip');
function show(e, html){ tip.innerHTML = html; tip.style.opacity = 1;
  const r = 14; tip.style.left = Math.min(e.clientX + r, innerWidth - 310) + 'px';
  tip.style.top = (e.clientY + r) + 'px'; }
function hide(){ tip.style.opacity = 0; }

const maxMove = Math.max(...RISK.map(r => r[2])) * 1.06;
const maxTail = Math.max(...RISK.map(r => r[3])) * 1.06;
document.getElementById('risk').innerHTML = RISK.map(r => {
  const [cls, n, mv, tl, vz, ok] = r;
  const col = ok ? 'var(--seq)' : 'var(--seq-dim)';
  return `<div class="row" onmousemove='show(event, ${JSON.stringify(
      `<b>${cls}</b><div>n = ${n} events</div>` +
      `<div>same-day |move| ${mv.toFixed(2)}σ — ${(mv/CTRL).toFixed(2)}× a random day</div>` +
      `<div>P(&gt;2σ) ${(tl*100).toFixed(1)}% — ${(tl/CTAIL).toFixed(2)}× control</div>` +
      `<div>abnormal volume z ${vz>=0?'+':''}${vz.toFixed(2)}</div>` +
      `<div>${ok ? 'clears the control' : 'does NOT clear the control'}</div>`
    )})' onmouseleave="hide()">
    <span class="lab">${cls}</span>
    <span class="track"><span class="bar" style="width:${mv/maxMove*100}%;background:${col}"></span>
      <span class="ref" style="left:${CTRL/maxMove*100}%"></span></span>
    <span class="val">${(mv/CTRL).toFixed(2)}×</span>
    <span class="track"><span class="bar" style="width:${tl/maxTail*100}%;background:${col};opacity:.62"></span>
      <span class="ref" style="left:${CTAIL/maxTail*100}%"></span></span>
    <span class="val">${(tl/CTAIL).toFixed(1)}×</span></div>`;
}).join('');

const lo = Math.min(0, ...CLU.flatMap(c => [c[1], c[2]]));
const hi = Math.max(...CLU.flatMap(c => [c[1], c[2]]));
const span = hi - lo;
const colOf = n => n.startsWith('GICS') ? 'var(--s2)'
  : n.startsWith('Random') ? 'var(--s4)' : 'var(--s1)';
document.getElementById('clu').innerHTML = CLU.map(c => {
  const [name, a, b] = c;
  const bar = v => `<span class="track"><span class="bar" style="width:${
      Math.abs(v)/span*100}%;background:${colOf(name)};${v<0?'opacity:.4':''}"></span>
      </span><span class="val" style="position:absolute;right:6px;top:0">${
      v>=0?'+':''}${v.toFixed(4)}</span>`;
  const cell = v => `<span style="position:relative;display:block">${bar(v)}</span>`;
  return `<div class="gr" onmousemove='show(event, ${JSON.stringify(
      `<b>${name}</b><div>all days: ${a>=0?'+':''}${a.toFixed(4)}</div>` +
      `<div>event windows: ${b>=0?'+':''}${b.toFixed(4)}</div>` +
      `<div>event-window lift: ${(b-a)>=0?'+':''}${(b-a).toFixed(4)}</div>`
    )})' onmouseleave="hide()">
    <span class="lab" style="text-align:left">${name}</span>
    ${cell(a)}${cell(b)}</div>`;
}).join('');

function tog(){ const h = document.documentElement;
  h.dataset.theme = h.dataset.theme === 'light' ? 'dark' : 'light'; }
</script></body></html>"""


def main() -> None:
    html = (HTML.replace("__RISK__", json.dumps(RISK_ROWS))
                .replace("__CLU__", json.dumps(CLUSTER_ROWS))
                .replace("__CTRL__", str(CONTROL_MOVE))
                .replace("__CTAIL__", str(CONTROL_TAIL)))
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html):,} bytes, self-contained, no external refs)")


if __name__ == "__main__":
    main()
