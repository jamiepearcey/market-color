# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""LIVE decomposition ROUTER — splits each recent activation's move into macro / sector / idio, then routes:
idio-dominant → firm-grounded retrieval; macro/sector-dominant → the sector/macro theme overview.
Minimal 2-factor model on data already on hand: market beta (ACWI) + same-sector peer average (sector.json)."""
import json
import math
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent / "data" / "eg_runs" / "eg_live2"
SECTOR = json.load(open(G / "sector.json"))


def closes(sym):
    p = G / "prices" / f"{sym}.json"
    if not p.exists():
        return {}
    out = {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], cl):
            if c is not None:
                out[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception:
        pass
    return out


def rets(sym):
    c = closes(sym); ds = sorted(c)
    return {ds[i]: math.log(c[ds[i]] / c[ds[i - 1]]) for i in range(1, len(ds)) if c[ds[i - 1]] > 0 and c[ds[i]] > 0}


MKT = rets("ACWI")
# sector -> member tickers (with price data)
sec_members = defaultdict(list)
for tk, s in SECTOR.items():
    if (G / "prices" / f"{tk}.json").exists():
        sec_members[s].append(tk)
_ret_cache = {}
def R(t):
    if t not in _ret_cache:
        _ret_cache[t] = rets(t)
    return _ret_cache[t]


def beta(y, x, days):
    xs = np.array([x[d] for d in days]); ys = np.array([y[d] for d in days])
    return float(np.cov(xs, ys)[0, 1] / np.var(xs)) if np.var(xs) > 1e-12 else 0.0


def decompose(sym, day):
    r = R(sym)
    if day not in r or day not in MKT:
        return None
    trail = [d for d in r if d < day and d in MKT][-60:]
    if len(trail) < 20:
        return None
    bm = beta(r, MKT, trail)
    tot = r[day]; mkt = MKT[day]
    macro = bm * mkt
    # sector = same-sector peer average return that day, beta-scaled on the residual
    peers = [p for p in sec_members.get(SECTOR.get(sym, "?"), []) if p != sym and day in R(p)]
    sec = 0.0
    if len(peers) >= 3:
        sret = {d: float(np.mean([R(p)[d] for p in peers if d in R(p)])) for d in trail + [day]}
        resid_tr = {d: r[d] - bm * MKT[d] for d in trail if d in sret}
        bs = beta(resid_tr, {d: sret[d] for d in resid_tr}, list(resid_tr)) if len(resid_tr) >= 20 else 0.0
        sec = bs * sret[day]
    idio = tot - macro - sec
    parts = {"macro": macro, "sector": sec, "idio": idio}
    dom = max(parts, key=lambda k: abs(parts[k]))
    return {"tot": tot, **parts, "dom": dom, "beta": bm, "npeers": len(peers)}


ACTS = [json.loads(l)["activation"] for l in open(G / "activation_first_report.jsonl")]
print(f"{'firm':6}{'day':12}{'move':>8}{'macro':>8}{'sector':>8}{'idio':>8}   ROUTE")
print("-" * 70)
for a in ACTS:
    d = decompose(a["sym"], a["day"])
    if not d:
        print(f"{a['sym']:6}{a['day']:12}{'—':>8}  (insufficient price history)")
        continue
    route = "firm-grounded retrieval" if d["dom"] == "idio" else "→ SECTOR/MACRO theme overview"
    print(f"{a['sym']:6}{a['day']:12}{d['tot']:>+8.1%}{d['macro']:>+8.1%}{d['sector']:>+8.1%}{d['idio']:>+8.1%}   "
          f"{d['dom']}-dominant · {route}")
