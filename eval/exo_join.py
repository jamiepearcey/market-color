#!/usr/bin/env python3
"""Exogenous join: news facts <-> price series. The operation chunk-RAG
cannot express — requires structured keys (entity -> instrument, event_time,
direction).

  confirm  For every price_move fact mappable to an instrument with a
           normalized event date: does the actual 1-day return sign match the
           claimed direction? Reports confirmation rate + z-score stats,
           overall and per instrument. (Basis for a market-confirmation
           ranking prior.)
  explore  Reverse hypothesis exploration: the biggest |zscore_20d| moves in
           the fact window, each joined to candidate causes from the fact
           base (dense + entity match near the event date) — the analyst's
           "what explains this move?" workflow, automated.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from strat_blind import _embed_all  # noqa: E402
from weighted_walk_test import norm_event_time  # noqa: E402

INSTRUMENT_KEYWORDS = {
    "cl.f": ["wti", "west texas"],
    "cb.f": ["brent"],
    "ng.f": ["natural gas", "henry hub"],
    "rb.f": ["gasoline", "rbob"],
    "gc.f": ["gold"],
    "si.f": ["silver"],
    "hg.f": ["copper"],
    "pl.f": ["platinum"],
    "btcusd": ["bitcoin", "btc"],
    "ethusd": ["ether", "ethereum"],
    "solusd": ["solana"],
    "^spx": ["s&p 500", "s&p500", "sp500"],
}


def _load():
    import duckdb
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    con = duckdb.connect()
    px = con.sql("select symbol, date, ret_1d, zscore_20d from "
                 f"'{ROOT}/data/prices/prices.parquet'").fetchall()
    series: dict[str, dict[str, tuple]] = collections.defaultdict(dict)
    for sym, date, ret, z in px:
        series[sym][str(date)] = (ret, z)
    return facts, series


def _instrument_of(f):
    hay = (str(f.get("subject") or "") + " " +
           " ".join(f.get("entities") or [])).lower()
    for sym, kws in INSTRUMENT_KEYWORDS.items():
        if any(k in hay for k in kws):
            return sym
    return None


def _ret_around(series_sym, d, tol_days=1):
    """Return (ret,z) at date d, else nearest trading day within tolerance."""
    for off in range(tol_days + 1):
        for cand in (d + dt.timedelta(days=off), d - dt.timedelta(days=off)):
            v = series_sym.get(cand.isoformat())
            if v and v[0] is not None:
                return v
    return None


def cmd_confirm(args):
    facts, series = _load()
    rows = []
    for f in facts:
        if f.get("predicate") != "price_move":
            continue
        if str(f.get("direction")) not in ("up", "down"):
            continue
        sym = _instrument_of(f)
        if not sym or sym not in series:
            continue
        et = norm_event_time(f.get("time"), f.get("published_date"))
        if not et:
            continue
        v = _ret_around(series[sym], et)
        if not v:
            continue
        ret, z = v
        claimed_up = f["direction"] == "up"
        confirmed = (ret > 0) == claimed_up
        rows.append({"sym": sym, "confirmed": confirmed, "ret": ret,
                     "z": z, "conf": float(f.get("confidence") or 0.5)})
    by_sym = collections.defaultdict(list)
    for r in rows:
        by_sym[r["sym"]].append(r)
    print(f"{'instrument':<10} {'n':>4} {'confirm%':>9} {'mean|z|':>8}")
    for sym, rs in sorted(by_sym.items(), key=lambda kv: -len(kv[1])):
        zs = [abs(r["z"]) for r in rs if r["z"] is not None]
        print(f"{sym:<10} {len(rs):>4} {100*sum(r['confirmed'] for r in rs)/len(rs):>8.0f}% "
              f"{(sum(zs)/len(zs)) if zs else 0:>8.2f}")
    n = len(rows)
    conf_rate = sum(r["confirmed"] for r in rows) / n if n else 0
    # does extraction confidence predict market confirmation?
    hi = [r for r in rows if r["conf"] >= 0.9]
    lo = [r for r in rows if r["conf"] < 0.9]
    print(json.dumps({
        "joined_price_move_facts": n,
        "market_confirmation_rate": round(conf_rate, 3),
        "conf>=0.9 confirmation": round(sum(r["confirmed"] for r in hi) / len(hi), 3) if hi else None,
        "conf<0.9 confirmation": round(sum(r["confirmed"] for r in lo) / len(lo), 3) if lo else None,
    }, indent=2))


def cmd_explore(args):
    facts, series = _load()
    print("[explore] embedding facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    for f in facts:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
    inst_names = {s: k[0] for s, k in INSTRUMENT_KEYWORDS.items()}
    # biggest |z| moves inside the fact window
    window_lo, window_hi = dt.date(2026, 6, 26), dt.date(2026, 7, 2)
    moves = []
    for sym, dd in series.items():
        for date, (ret, z) in dd.items():
            d = dt.date.fromisoformat(date)
            if window_lo <= d <= window_hi and z is not None and ret is not None:
                moves.append((abs(z), sym, d, ret, z))
    moves.sort(reverse=True)
    for absz, sym, d, ret, z in moves[:5]:
        name = inst_names.get(sym, sym)
        q = f"why did {name} move {'up' if ret > 0 else 'down'} sharply"
        qv = _embed_all([q])[0]
        sims = fmat @ qv
        cands = []
        for i in np.argsort(-sims):
            f = facts[int(i)]
            if not f["_etime"] or abs((f["_etime"] - d).days) > 2:
                continue
            hay = (str(f.get("subject") or "") + " "
                   + " ".join(f.get("entities") or [])).lower()
            if not any(k in hay for k in INSTRUMENT_KEYWORDS.get(sym, [])):
                continue
            cands.append(f)
            if len(cands) >= 3:
                break
        print(f"\n== {sym} {d} ret={ret:+.2%} z={z:+.1f} — candidate causes:")
        for f in cands:
            cause = f" ⟵ {f['cause']}" if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else ""
            print(f"   [{f.get('source_name')}] {str(f.get('claim'))[:90]}{cause}")
        if not cands:
            print("   (no fact-base explanation within ±2 days — UNEXPLAINED move flag)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["confirm", "explore"])
    args = p.parse_args()
    {"confirm": cmd_confirm, "explore": cmd_explore}[args.stage](args)


if __name__ == "__main__":
    main()
