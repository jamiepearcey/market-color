# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx"]
# ///
"""
DAY EXPLAIN — decompose a day's moves, drill into what formed each layer, and
show the GATED answer beside the NAIVE RAG answer.

This is the comprehensive view the individual pages each showed a slice of.

WHAT IT DOES
  1. For a chosen day, decompose every name's return:
         total = market (equal-weighted factor) + sector + idiosyncratic
     and report how the cross-sectional variance splits across those layers.
     Most days are mostly market+sector. Company-specific news is expected to
     explain primarily the IDIOSYNCRATIC residual, which is usually the smallest
     piece — which is why article-only attribution frequently overstates the
     role of company news.
  2. Drill into a SECTOR or an individual NAME and get an answer formed by an
     LLM that is handed the decomposition, the measured class priors, and the
     gate verdict — and is instructed to treat the narrative as CONTRIBUTORY
     RATHER THAN EXPLANATORY when the reaction is within the class's normal
     historical range.
  3. Beside it, the NAIVE RAG answer: the same article chunks, no decomposition,
     no priors, no gate — "here are the articles, explain the move". This is the
     control, and it is the point of the whole exhibit.

THE GATE — a NARRATIVE CONFIDENCE TEST, not a test of truth or causality. It
asks whether a candidate explanation is strong enough to be treated as PRIMARY:
  - the idiosyncratic move should exceed an ordinary non-event day (~0.75 sigma)
  - the event class should itself produce reactions distinguishable from
    ordinary days in-sample (election/credit_event do not)
  - surprise = observed - class_prior should be > 0: a 1.6-sigma reaction on an
    earnings day is within the normal historical range for earnings (1.62), so
    the announcement is insufficient to account for the magnitude
Crucially the gate NEVER asserts what DID cause a move. Rejecting one
explanation is not evidence for another; a failing verdict says the narrative is
CONTRIBUTORY, not that it is false or that its impact was zero.

Full article chunks are carried into both prompts and into the UI, so the
evidence a reader sees is the evidence the model saw.

Usage:
    source ../data/tmp/groq.env && uv run scripts/build_day_explain.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# SIC (SEC EDGAR), not GICS. GICS is proprietary so gics.py was ~140 tickers
# assigned BY HAND — 12% of the universe, unverifiable, and flat. SIC is public
# domain, assigned per filer by the SEC itself (74% coverage here), and genuinely
# hierarchical: division -> major group -> 4-digit industry, which is what the
# drill-down needs. SIC is an older and coarser taxonomy than GICS; it is used
# because it is real and checkable, which beats a better-shaped invented one.
from sic import sector as gics_sector, subsector, industry  # noqa: E402,F401
from gate import Gate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
OUT = G / "day_explain.html"
GROQ = "https://api.groq.com/openai/v1/chat/completions"
KEY = os.environ.get("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
BETA_WIN, TRAIL = 250, 60
N_DAYS = 6
# The gate is NOT re-implemented here. It lives in gate.py and reads TRAILING
# priors: a rolling estimate showed class priors drift twice as much as the
# baseline and REORDER (legal_regulatory spans rank 1 to 7 across the sample), so
# a pooled prior mis-gates 3.6% of events in both directions — 2.0% of ordinary
# days waved through as SUFFICIENT, which is exactly where a confident but
# unsupported narrative gets generated.
GATE = Gate()


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


def llm(system: str, user: str, max_tokens: int = 2200) -> str:
    """NOTE gpt-oss-120b is a REASONING model: reasoning tokens are drawn from the
    same budget, so a small max_tokens silently returns EMPTY content rather than
    erroring. 2200 leaves room for reasoning plus the <=75-word answer. This trap
    has bitten this project before — do not lower it."""
    if not KEY:
        return "(no GROQ_API_KEY — source ../data/tmp/groq.env)"
    try:
        r = httpx.post(GROQ, headers={"Authorization": f"Bearer {KEY}"},
                       json={"model": MODEL, "temperature": 0.2,
                             "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]},
                       timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"(LLM error: {exc})"


NAIVE_SYS = ("You are a market commentator. You are given news articles about a company or "
             "sector and the size of its price move. Explain what drove the move. Be concise "
             "(<=70 words) and confident, in the style of a market wrap.")
GATED_SYS = (
    "You are a disciplined attribution analyst. You assess whether a candidate news narrative "
    "is STRONG ENOUGH to be treated as the primary explanation for a price move. This is a "
    "NARRATIVE CONFIDENCE TEST, not a test of truth or causality.\n"
    "You are given a residual decomposition, MEASURED historical priors for the news class, and "
    "a gate verdict.\n"
    "RULES:\n"
    "1. If the verdict is INSUFFICIENT you must NOT present the news as the explanation. State "
    "that the observed reaction falls within the normal historical range for that event class, "
    "give the numbers, and describe the narrative as CONTRIBUTORY RATHER THAN EXPLANATORY.\n"
    "2. NEVER claim to know what DID cause the move. Rejecting one explanation is not evidence "
    "for another. Do not write 'hence the move stems from X'. The correct formulation is: the "
    "move remains largely idiosyncratic, but the observed news does not provide sufficient "
    "evidence to attribute it to this event.\n"
    "3. Do not say the news is false, or that it had zero impact — neither is established. Say "
    "it is insufficient to account for the magnitude.\n"
    "4. If the verdict is SUFFICIENT, explain using only the supplied evidence and state how far "
    "the reaction exceeded the historical norm for that class.\n"
    "5. MATCH YOUR CONFIDENCE TO THE STATED TIER. SUFFICIENT is not binary permission to sound "
    "certain. On a WEAK tier the move only barely cleared its class prior — say so explicitly and "
    "hedge; do not write that the news 'explains' or 'drove' the move. Reserve firm language for "
    "the STRONG tier.\n"
    "6. Never bridge an evidence gap with 'likely' or 'investors saw'.\n"
    "Be concise (<=75 words).")


def main() -> None:
    # ---------- load graph ----------
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
    chunks = collections.defaultdict(list)
    for l in open(G / "lake" / "chunk.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") in docmeta and j.get("text"):
            chunks[j["doc_id"]].append((j.get("seq", 0), j["text"]))
    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])
    # (ticker, day) -> docs
    cells = collections.defaultdict(lambda: {"types": set(), "docs": []})
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        meta = docmeta.get(doc)
        if e not in ent or not meta:
            continue
        c = cells[(ent[e], meta[0])]
        c["types"] |= (doctypes.get(doc) or {"other"})
        if doc not in c["docs"]:
            c["docs"].append(doc)

    # ---------- prices, factors, residuals ----------
    cache = G / "prices"
    rets = {}
    for tk in {t for t, _ in cells}:
        p = cache / f"{tk.replace('/', '-')}.json"
        if not p.exists():
            continue
        try:
            res = json.loads(p.read_text())["chart"]["result"][0]
            ts = res["timestamp"]; ind = res["indicators"]
            cl = (ind.get("adjclose", [{}])[0].get("adjclose")
                  if "adjclose" in ind else None) or ind["quote"][0]["close"]
        except Exception:
            continue
        dd, px = [], []
        for t, c in zip(ts, cl):
            if c and c > 0:
                dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
                px.append(float(c))
        r = {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}
        if not _tradeable(r):
            continue
        if len(r) > 600:
            rets[tk] = r
    dc = collections.Counter()
    for r in rets.values():
        dc.update(r.keys())
    alldays = sorted([d for d, c in dc.items() if c >= max(30, int(.3 * len(rets)))])
    dayidx = {d: i for i, d in enumerate(alldays)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])

    # per-sector factor = equal-weighted sector return, orthogonal to market
    secs = collections.defaultdict(list)
    for tk in rets:
        s = gics_sector(tk)
        if s != "UNK":
            secs[s].append(tk)
    SF = {}
    for s, mem in secs.items():
        if len(mem) < 3:
            continue
        v = np.array([np.mean([rets[t][d] for t in mem if d in rets[t]] or [np.nan])
                      for d in alldays])
        ok = np.isfinite(v) & np.isfinite(F)
        b = float(np.polyfit(F[ok], v[ok], 1)[0]) if ok.sum() > 100 else 1.0
        SF[s] = v - b * F           # sector factor net of market
    resid, beta_m, beta_s = {}, {}, {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < 600:
            continue
        s = gics_sector(tk)
        sf = SF.get(s)
        e = np.full(len(alldays), np.nan)
        bm = np.full(len(alldays), np.nan); bs = np.full(len(alldays), np.nan)
        for i in range(BETA_WIN, len(alldays)):
            sl = slice(i - BETA_WIN, i); m = ok[sl]
            if m.sum() < 150 or not ok[i]:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean(); den = float(fc @ fc)
            if den <= 0:
                continue
            b1 = float((yy - yy.mean()) @ fc / den)
            r1 = y[i] - b1 * F[i]
            b2 = 0.0
            if sf is not None:
                m2 = m & np.isfinite(sf[sl])
                if m2.sum() > 100:
                    ss = sf[sl][m2]; sc = ss - ss.mean()
                    d2 = float(sc @ sc)
                    if d2 > 0:
                        yr = y[sl][m2] - b1 * F[sl][m2]
                        b2 = float((yr - yr.mean()) @ sc / d2)
            e[i] = r1 - (b2 * sf[i] if sf is not None and np.isfinite(sf[i]) else 0.0)
            bm[i] = b1 * F[i]
            bs[i] = (b2 * sf[i]) if sf is not None and np.isfinite(sf[i]) else 0.0
        resid[tk] = e; beta_m[tk] = bm; beta_s[tk] = bs
    print(f"tickers: {len(resid)}, sectors: {len(SF)}")

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
    for _ in range(6000):
        tk = tl[rng.integers(len(tl))]
        i = int(rng.integers(BETA_WIN + TRAIL + 5, len(alldays) - 2))
        if (tk, alldays[i]) in evset:
            continue
        m = msig(tk, i)
        if m:
            ctrl.append(m[0])
    CTRL = float(np.mean(ctrl))
    print(f"control {CTRL:.3f} sigma")

    # ---------- choose days: most event-bearing, spread across the window ----------
    per_day = collections.Counter(d for (_, d) in cells)
    cand = [d for d in alldays[BETA_WIN + TRAIL + 5:len(alldays) - 2] if per_day[d] >= 8]
    step = max(1, len(cand) // N_DAYS)
    days = cand[::step][:N_DAYS]
    print(f"days: {days}")

    payload = {"ctrl": round(CTRL, 3), "prior": GATE.pooled, "days": []}
    for d in days:
        i = dayidx[d]
        rows = []
        for tk in resid:
            m = msig(tk, i)
            if m is None:
                continue
            tot = rets[tk].get(d)
            if tot is None:
                continue
            rows.append({"tk": tk, "sec": gics_sector(tk), "tot": tot,
                         "mkt": float(beta_m[tk][i]) if np.isfinite(beta_m[tk][i]) else 0.0,
                         "sct": float(beta_s[tk][i]) if np.isfinite(beta_s[tk][i]) else 0.0,
                         "idi": float(resid[tk][i]), "sig": m[0], "signed": m[1]})
        if len(rows) < 40:
            continue
        # variance shares across the cross-section
        v = lambda k: float(np.var([r[k] for r in rows]))
        vt = max(v("tot"), 1e-12)
        shares = {"market": round(v("mkt") / vt, 3), "sector": round(v("sct") / vt, 3),
                  "idio": round(v("idi") / vt, 3)}
        # sector aggregation
        secagg = collections.defaultdict(lambda: {"n": 0, "sct": 0.0, "idi": 0.0, "tot": 0.0})
        for r in rows:
            a = secagg[r["sec"]]
            a["n"] += 1; a["sct"] += r["sct"]; a["idi"] += abs(r["idi"]); a["tot"] += r["tot"]
        sect = sorted(([s, a["n"], round(a["tot"] / a["n"], 5), round(a["sct"] / a["n"], 5),
                        round(a["idi"] / a["n"], 5)] for s, a in secagg.items() if s != "UNK"),
                      key=lambda x: -abs(x[3]))
        # notable names: biggest idio moves that have news
        withnews = [r for r in rows if (r["tk"], d) in cells]
        withnews.sort(key=lambda r: -r["sig"])
        picks = withnews[:3]
        cases = []
        for r in picks:
            c = cells[(r["tk"], d)]
            # pick the class this day's news most strongly implies, ranked by the
            # prior IN FORCE ON THAT DATE — not a pooled constant
            cls = max(c["types"], key=lambda t: (GATE.as_of(d, t)["prior"] or 0.0))
            v = GATE.evaluate(d, cls, r["sig"])
            exp, sur, why = v["prior"], v["surprise"], v["reasons"]
            ctrl_d = v["control"]          # control as of this date, not pooled
            docs = c["docs"][:3]
            texts = []
            for doc in docs:
                head = docmeta[doc][1]
                body = " ".join(t for _, t in sorted(chunks.get(doc, [])))[:900]
                texts.append({"head": head, "body": body})
            evidence = "\n\n".join(f"[{t['head']}]\n{t['body']}" for t in texts) or "(no article text)"
            share_i = abs(r["idi"]) / max(abs(r["tot"]), 1e-9)
            naive = llm(NAIVE_SYS,
                        f"{r['tk']} moved {r['tot']*100:+.2f}% today ({d}).\n"
                        f"News articles:\n\n{evidence}\n\nWhy did it move?")
            verdict = v["status"]
            gated = llm(GATED_SYS,
                        f"Name: {r['tk']} on {d}\n"
                        f"Total move {r['tot']*100:+.2f}% decomposes as: market "
                        f"{r['mkt']*100:+.2f}%, sector {r['sct']*100:+.2f}%, "
                        f"IDIOSYNCRATIC {r['idi']*100:+.2f}% "
                        f"({share_i:.0%} of the total).\n"
                        f"Idiosyncratic move = {r['sig']:.2f} sigma of this name's own "
                        f"trailing residual volatility. A random non-event day in the same "
                        f"period averages {ctrl_d:.2f} sigma.\n"
                        f"News class: {cls}. MEASURED prior for that class AS OF "
                        f"{v['asof']}: {exp:.2f} sigma (what this class of news was producing "
                        f"in the trailing window — this prior moves over time, it is not a "
                        f"constant).\n"
                        f"Surprise (observed - prior) = {sur:+.2f} sigma.\n"
                        f"GATE VERDICT: {verdict} (narrative-confidence test)\n"
                        f"CONFIDENCE TIER: {v['confidence']} — {v['strength_note']}"
                        + (f"\nReasons: {'; '.join(why)}" if why else "")
                        + f"\n\nEvidence:\n{evidence}\n\nExplain, following your rules.")
            cases.append({"tk": r["tk"], "nm": name_of.get(r["tk"], r["tk"]),
                          "sec": r["sec"], "tot": round(r["tot"], 5),
                          "mkt": round(r["mkt"], 5), "sct": round(r["sct"], 5),
                          "idi": round(r["idi"], 5), "sig": round(r["sig"], 2),
                          "cls": cls, "exp": round(exp, 2), "sur": round(sur, 2),
                          "asof": v["asof"], "ctrl": round(ctrl_d, 2),
                          "conf": v["confidence"], "z": v["surprise_z"],
                          "pooled": GATE.pooled.get(cls),
                          "ok": v["status"] == "SUFFICIENT", "why": why, "texts": texts,
                          "naive": naive, "gated": gated})
            print(f"  {d} {r['tk']:6} {verdict:12} prior {exp:.2f}σ as-of {v['asof']} "
                  f"(pooled {GATE.pooled.get(cls)})")
        payload["days"].append({"d": d, "n": len(rows), "shares": shares,
                                "sect": sect[:8], "cases": cases,
                                "mkt": round(float(F[i]), 5)})
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"\nwrote {OUT} ({OUT.stat().st_size:,} bytes)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day explain — decomposition, drill-down, and the naive-RAG control</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --border:rgba(255,255,255,.10);--up:#0ca30c;--down:#e66767;--warn:#fab219;--acc:#3987e5;
  --bad:#d03b3b}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
  --border:rgba(11,11,11,.10);--up:#006300;--down:#e34948;--acc:#2a78d6}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1300px;margin:0 auto;padding:20px 18px 60px}
 h1{font-size:18px;margin:0 0 3px}h2{font-size:13.5px;margin:0 0 3px}
 .sub{color:var(--ink2);font-size:12.5px;margin:0 0 14px;max-width:88ch}
 .bar{display:flex;gap:12px;align-items:center;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;padding:10px 12px;margin-bottom:14px}
 select{background:var(--page);color:var(--ink);border:1px solid var(--border);
  border-radius:6px;padding:6px 9px;font-size:13px}
 .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:15px 17px;margin-bottom:14px}
 .note{color:var(--muted);font-size:11.5px;margin:0 0 12px;max-width:90ch}
 .wf{display:flex;height:34px;border-radius:5px;overflow:hidden;margin:10px 0 6px}
 .wf div{display:flex;align-items:center;justify-content:center;font-size:11.5px;color:#fff}
 .lg{display:flex;gap:16px;font-size:11.5px;color:var(--ink2)}
 .sec{display:grid;grid-template-columns:130px 1fr 1fr;gap:10px;align-items:center;
  padding:3px 0;font-size:12px}
 .trk{position:relative;height:13px;background:rgba(128,128,128,.14);border-radius:3px}
 .fil{position:absolute;height:13px;border-radius:3px}
 .case{border:1px solid var(--border);border-radius:9px;padding:13px 15px;margin-bottom:11px}
 .ch{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
 .tk{font-size:17px;font-weight:600}
 .chip{font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;padding:2px 6px;
  border-radius:3px;background:rgba(57,135,229,.16);color:var(--acc)}
 .chip.x{background:rgba(250,178,25,.16);color:var(--warn)}
 .dec{font-size:11.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
 .ab{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
 .ans{border-radius:8px;padding:11px 13px;font-size:12.5px;line-height:1.6}
 .naive{background:rgba(208,59,59,.09);border:1px solid rgba(208,59,59,.3)}
 .gated{background:rgba(57,135,229,.08);border:1px solid rgba(57,135,229,.3)}
 .ans h3{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;margin:0 0 6px;font-weight:600}
 .naive h3{color:var(--bad)}.gated h3{color:var(--acc)}
 .ev{margin-top:9px}
 details{font-size:11.5px;color:var(--ink2)}summary{cursor:pointer;color:var(--muted);font-size:11px}
 .art{border-left:2px solid var(--border);padding-left:9px;margin-top:7px}
 .art b{color:var(--ink);font-size:11.5px}
 .art p{margin:3px 0 0;font-size:11px;color:var(--ink2);line-height:1.55}
 .tog{position:fixed;top:12px;right:14px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
 footer{color:var(--muted);font-size:11px;margin-top:20px;line-height:1.7}
</style></head><body>
<button class="tog" onclick="tg()">◐</button><div class="wrap">
<h1>Day explain — what moved, which layer carried it, and what actually explains it</h1>
<p class="sub">Every move splits into <b>market</b> + <b>sector</b> + <b>idiosyncratic</b>.
Company-specific news is expected to explain primarily the <b>idiosyncratic residual</b>, which is
usually the smallest piece — which is why article-only attribution frequently overstates the role of
company news. Each name shows the <span style="color:var(--bad)">article-only answer</span> (same
articles, no decomposition, no priors) beside the <span style="color:var(--acc)">gated answer</span>.
The gate is a <b>narrative-confidence test</b>: it asks whether a candidate explanation is strong
enough to be treated as primary, not whether it is true.</p>
<div class="bar"><label style="font-size:12px;color:var(--ink2)">day <select id="sel"></select></label>
<span id="hint" style="font-size:11.5px;color:var(--muted)"></span></div>
<div class="card"><h2>Where the day's variance sat</h2>
 <p class="note">Share of cross-sectional return variance by layer. Company-specific news is
 expected to speak primarily to the idiosyncratic slice.</p>
 <div class="wf" id="wf"></div><div class="lg" id="lg"></div></div>
<div class="card"><h2>By GICS sector</h2>
 <p class="note">Mean sector-factor move (net of market) and mean absolute idiosyncratic move per name.</p>
 <div id="sect"></div></div>
<div id="cases"></div>
<footer><b>Method.</b> Market = equal-weighted factor; sector = equal-weighted sector return net of
market; idiosyncratic = what remains after rolling 250-day betas on both. Moves standardised by each
name's own trailing 60-day residual vol. Control = <span id="cf"></span>σ, the mean idiosyncratic move
on a random non-event day. The gated model is handed the decomposition, the measured class prior and
the gate verdict, and is instructed to treat a narrative as <b>contributory rather than explanatory</b> when the observed
reaction falls within the normal historical range for its class. Rejecting one explanation is not
evidence for another: the gate never asserts what <em>did</em> cause a move. The article-only model
gets the same articles and nothing else.</footer>
</div>
<script>
const D=__DATA__;
document.getElementById('cf').textContent=D.ctrl;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=v=>(v*100).toFixed(2)+'%';
sel.innerHTML=D.days.map((d,i)=>`<option value="${i}">${d.d}</option>`).join('');
function draw(){
 const day=D.days[+sel.value];
 hint.textContent=`${day.n} names · market ${pct(day.mkt)}`;
 const S=day.shares,cols={market:'var(--acc)',sector:'#d95926',idio:'#199e70'};
 wf.innerHTML=['market','sector','idio'].map(k=>
  `<div style="width:${S[k]*100}%;background:${cols[k]}">${(S[k]*100).toFixed(0)}%</div>`).join('');
 lg.innerHTML=['market','sector','idio'].map(k=>
  `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${cols[k]};margin-right:5px"></span>${k} ${(S[k]*100).toFixed(0)}%</span>`).join('');
 const mx=Math.max(...day.sect.flatMap(s=>[Math.abs(s[3]),s[4]]),1e-6);
 sect.innerHTML=`<div class="sec" style="color:var(--muted);font-size:10.5px"><span></span>
  <span>sector factor (net of market)</span><span>mean |idiosyncratic|</span></div>`+
  day.sect.map(([s,n,tot,sc,idi])=>`<div class="sec">
   <span style="text-align:right;color:var(--ink2)">${s} <span style="color:var(--muted)">${n}</span></span>
   <span class="trk"><span class="fil" style="width:${Math.abs(sc)/mx*100}%;background:${sc>=0?'var(--up)':'var(--down)'}"></span></span>
   <span class="trk"><span class="fil" style="width:${idi/mx*100}%;background:#199e70;opacity:.75"></span></span>
  </div>`).join('');
 cases.innerHTML=day.cases.map(c=>`<div class="card case">
  <div class="ch"><span class="tk">${c.tk}</span>
   <span style="color:var(--ink2);font-size:12px">${esc(c.nm)} · ${c.sec}</span>
   <span class="chip${c.ok?'':' x'}">${esc(c.cls).replace(/_/g,' ')}</span>
   <span class="dec">total ${pct(c.tot)} = market ${pct(c.mkt)} + sector ${pct(c.sct)} +
    <b style="color:var(--ink)">idio ${pct(c.idi)}</b></span>
   <span class="dec">idio ${c.sig}σ · control ${c.ctrl}σ ·
    <span title="Class priors drift and reorder over time, so this is the trailing estimate in force on this date — not a pooled constant.">prior <b style="color:var(--ink)">${c.exp}σ</b>
    <span style="color:var(--muted)">as of ${c.asof}${c.pooled&&Math.abs(c.pooled-c.exp)>=0.08?` · pooled would say ${c.pooled.toFixed(2)}σ`:''}</span></span> ·
    <b style="color:${c.sur>0?'var(--ink)':'var(--warn)'}">surprise ${c.sur>0?'+':''}${c.sur}σ</b></span>
  </div>
  <div style="font-size:11.5px;margin-bottom:6px;color:${c.ok?(c.conf==='WEAK'?'var(--warn)':'var(--ink2)'):'var(--warn)'}">
   ${c.ok?`${c.conf==='WEAK'?'◔':c.conf==='MODERATE'?'◑':'●'} gate: SUFFICIENT · confidence <b>${c.conf}</b> — surprise is ${c.z} of this class's own standard deviation${c.conf==='WEAK'?', so this only barely clears the prior':''}`
        :`◯ gate: INSUFFICIENT — ${esc(c.why.join('; '))}`}</div>
  <div class="ab">
   <div class="ans naive"><h3>article-only — no decomposition, no priors</h3>${esc(c.naive)}</div>
   <div class="ans gated"><h3>gated — decomposition + measured prior + confidence test</h3>${esc(c.gated)}</div>
  </div>
  <div class="ev"><details><summary>source articles (${c.texts.length}) — the same evidence both models saw</summary>
   ${c.texts.map(t=>`<div class="art"><b>${esc(t.head)}</b><p>${esc(t.body)}</p></div>`).join('')}
  </details></div></div>`).join('');
}
sel.addEventListener('change',draw);draw();
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light'}
</script></body></html>"""


if __name__ == "__main__":
    main()
