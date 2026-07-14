#!/usr/bin/env python3
"""News-lag event study — do unexplained price spikes predict news/returns?

PRE-REGISTERED DESIGN (fixed before any results):
  Event: |zscore_20d| >= 2.0 on instrument i, day T (within the fact-corpus
  window). Regional flag from instrument desks (asia/fx vs core).
  Explanation state at info-date D (publication-aware — only facts with
  published_date <= D count; event_time within [T-1, T]):
    2 = causal fact (instrument-matched, direction-matched, cause present)
    1 = mention (instrument-matched price_move, no cause)
    0 = nothing
  UNEXPLAINED at T = state < 2 using facts published <= T.
  Outcomes:
    A. CATCH-UP: P(state reaches 2 using facts published in (T, T+2]) —
       unexplained vs explained events.
    B. CONTINUATION: sign(ret_T) * ret_{T+1} — unexplained vs explained.
  No post-hoc thresholds; z>=2 and the state definitions are frozen.

Run: uv run ... python eval/newslag.py study
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from weighted_walk_test import norm_event_time  # noqa: E402

INSTRUMENT_KEYWORDS = {
    "cl.f": ["wti", "west texas"], "cb.f": ["brent"],
    "ng.f": ["natural gas", "henry hub"], "rb.f": ["gasoline", "rbob"],
    "gc.f": ["gold"], "si.f": ["silver"], "hg.f": ["copper"],
    "pl.f": ["platinum"], "btcusd": ["bitcoin", "btc"],
    "ethusd": ["ether", "ethereum"], "solusd": ["solana"],
    "^spx": ["s&p 500", "s&p500", "sp500", "s&p"],
    "^ndq": ["nasdaq"], "^dji": ["dow jones", "dow "],
    "^vix": ["vix", "volatility index"],
    "eurusd": ["eur/usd", "euro"], "usdjpy": ["usd/jpy", "yen"],
    "gbpusd": ["gbp/usd", "pound", "sterling"],
    "usdcny": ["usd/cny", "yuan", "renminbi"],
    "audusd": ["aud/usd", "australian dollar"],
    "usdkrw": ["usd/krw", "korean won", "won"],
    "dx.f": ["dollar index", "dxy"],
    "10usy.b": ["10-year", "10 year", "treasury yield"],
    "2usy.b": ["2-year", "two-year", "2 year"],
    "30usy.b": ["30-year", "30 year", "long bond"],
    "^nkx": ["nikkei"], "^hsi": ["hang seng"],
    "^shc": ["shanghai composite", "shanghai"], "^kospi": ["kospi"],
}
REGIONAL = {"usdkrw", "usdcny", "^nkx", "^hsi", "^shc", "^kospi", "usdjpy", "audusd"}


def _load():
    import duckdb
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    for f in facts:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
        try:
            f["_pub"] = dt.date.fromisoformat(str(f.get("published_date"))[:10])
        except Exception:
            f["_pub"] = None
        hay = (str(f.get("subject") or "") + " " + str(f.get("claim") or "")[:60]
               + " " + " ".join(f.get("entities") or [])).lower()
        f["_inst"] = None
        for sym, kws in INSTRUMENT_KEYWORDS.items():
            if any(k in hay for k in kws):
                f["_inst"] = sym
                break
    con = duckdb.connect()
    px = con.sql("select symbol, date, ret_1d, zscore_20d from "
                 f"'{ROOT}/data/prices/prices.parquet' order by symbol, date").fetchall()
    series = collections.defaultdict(list)
    for sym, date, ret, z in px:
        series[sym].append((dt.date.fromisoformat(str(date)), ret, z))
    return facts, series


def _state(facts, sym, T, pub_lo, pub_hi, direction):
    best = 0
    for f in facts:
        if f["_inst"] != sym or not f["_pub"] or not f["_etime"]:
            continue
        if not (pub_lo < f["_pub"] <= pub_hi):
            continue
        if not (T - dt.timedelta(days=1) <= f["_etime"] <= T + dt.timedelta(days=1)):
            continue
        if str(f.get("direction")) in ("up", "down") \
                and (f["direction"] == "up") != (direction > 0):
            continue
        has_cause = f.get("cause") and str(f["cause"]).lower() not in ("none", "null")
        best = max(best, 2 if has_cause else 1)
        if best == 2:
            break
    return best


def cmd_study(args):
    facts, series = _load()
    pubs = [f["_pub"] for f in facts if f["_pub"]]
    corpus_lo, corpus_hi = min(pubs), max(pubs)
    print(f"fact corpus publication span: {corpus_lo} .. {corpus_hi}", file=sys.stderr)
    events = []
    for sym, rows in series.items():
        for i, (d, ret, z) in enumerate(rows):
            if z is None or abs(z) < 2.0 or ret is None:
                continue
            # event must sit where we HAVE news for T and T+2
            if not (corpus_lo <= d <= corpus_hi - dt.timedelta(days=2)):
                continue
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            explained_T = _state(facts, sym, d, dt.date(2000, 1, 1), d, ret)
            catchup = _state(facts, sym, d, d, d + dt.timedelta(days=2), ret)
            events.append({
                "sym": sym, "date": d.isoformat(), "ret": ret, "z": z,
                "regional": sym in REGIONAL,
                "explained_T": explained_T,
                "catchup_state": catchup,
                "ret_next": nxt[1] if nxt else None,
            })
    unexp = [e for e in events if e["explained_T"] < 2]
    expl = [e for e in events if e["explained_T"] == 2]

    def cont(evs):
        vals = [(1 if e["ret"] > 0 else -1) * e["ret_next"]
                for e in evs if e["ret_next"] is not None]
        return round(sum(vals) / len(vals), 5) if vals else None

    for e in sorted(events, key=lambda e: -abs(e["z"])):
        print(f"  {e['sym']:<8} {e['date']} z={e['z']:+.1f} "
              f"explained_T={e['explained_T']} catchup={e['catchup_state']} "
              f"{'REGIONAL' if e['regional'] else ''}")
    print(json.dumps({
        "events": len(events),
        "unexplained_at_T": len(unexp),
        "explained_at_T": len(expl),
        "A_catchup_rate_unexplained": round(
            sum(1 for e in unexp if e["catchup_state"] == 2) / len(unexp), 3) if unexp else None,
        "B_continuation_unexplained": cont(unexp),
        "B_continuation_explained": cont(expl),
        "regional_events": sum(1 for e in events if e["regional"]),
        "regional_unexplained_pct": round(100 * sum(
            1 for e in unexp if e["regional"]) / max(len(unexp), 1)),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["study"])
    args = p.parse_args()
    cmd_study(args)


if __name__ == "__main__":
    main()
