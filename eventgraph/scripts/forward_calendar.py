# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Forward catalyst calendar — attribution.py's forward twin, built on the Polymarket
propositions (polymarket_ingest.py).

Where the news attribution surface is BACKWARD ("what moved & why"), this is FORWARD:
dated catalysts, their market-implied probability, the linked tradeable names, and the
mechanism reliability borrowed from the news graph. Three views:

  calendar          upcoming resolution dates, each with its markets, implied prob,
                    linked US names, volume (attention) and 1d prob move.
  name  TICKER      the forward catalysts attached to one name.
  synth             SYNTHESISE event nodes conforming to the calendar: one dated,
                    two-scenario (Yes/No) event per resolved market, with implied
                    probabilities + borrowed mechanism reliability. Writes
                    lake/synthetic_event.jsonl -- a forward event layer the risk /
                    vol machinery can consume alongside the realised news edges.

"Profitable" candidates surfaced by --edge: markets whose implied prob is moving
(|1d change|) on rising volume (attention arriving) while the resolution date is near
-- the attention-before-realisation setup. This is a WATCHLIST, not a claim; the vol
test decides whether the attention actually predicts realised movement.

Usage:
  uv run eventgraph/scripts/forward_calendar.py calendar --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/forward_calendar.py name NVDA --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/forward_calendar.py synth  --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/forward_calendar.py calendar --edge --graph-dir eventgraph/data/eg_live
"""
import argparse, json, collections, datetime as dt
from pathlib import Path

# measured name-level directional reliability (from signal_value.py; kept in sync with attribution.py).
S_DIR = {"monetary_policy": 0.43, "competition": 1.13, "rating_action": 1.02, "demand_change": 0.30,
         "earnings": 0.46, "regulation": 0.17, "geopolitics": 0.26, "other": 0.22, "guidance": 0.09,
         "rate_decision": 0.16, "supply_shock": 0.05, "default": 0.04, "mergers_acquisitions": -0.02,
         "data_surprise": -0.39, "contagion": 0.30}
# map a proposition category -> the news mechanism whose reliability we borrow.
CAT_MECH = {"crypto": "other", "macro": "rate_decision", "company": "earnings",
            "commodity": "supply_shock", "other": "other"}


def load_props(gd):
    p = gd / "lake" / "proposition.jsonl"
    if not p.exists():
        raise SystemExit(f"no {p} -- run polymarket_ingest.py first")
    return [json.loads(l) for l in open(p) if l.strip()]


def reliability(cat):
    m = CAT_MECH.get(cat, "other")
    return m, S_DIR.get(m, 0.1)


def edge_score(p):
    """attention-before-realisation heuristic: prob moving on rising volume, resolution near."""
    try:
        chg = abs(float(p.get("one_day_price_change") or 0))
    except Exception:
        chg = 0.0
    vol24 = p.get("volume24hr") or 0.0
    return chg * (1.0 + (vol24 / max(p.get("volume") or 1.0, 1.0)))


def fmt_prob(p):
    v = p.get("prob_yes")
    return f"{v*100:4.0f}%" if isinstance(v, (int, float)) else "  ? "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calendar", "name", "synth"])
    ap.add_argument("ticker", nargs="?", default="")
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--resolved-only", action="store_true", help="only markets linked to a ticker")
    ap.add_argument("--edge", action="store_true", help="rank calendar by attention-before-realisation score")
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    props = load_props(gd)
    if a.resolved_only or a.cmd in ("name", "synth"):
        props = [p for p in props if p.get("symbols")]

    if a.cmd == "name":
        t = a.ticker.upper()
        rows = [p for p in props if t in (p.get("symbols") or [])]
        if not rows:
            print(f"no forward catalysts linked to {t}"); return
        rows.sort(key=lambda p: p.get("resolution_date") or "9999")
        print(f"=== forward catalysts for {t} ===  ({len(rows)} markets)")
        for p in rows:
            mech, rel = reliability(p["category"])
            print(f"  {p.get('resolution_date','?'):10}  P(yes) {fmt_prob(p)}  vol ${p['volume']/1e6:6.2f}M  "
                  f"[{p['category']}/{mech} rel {rel:+.2f}]")
            print(f"       {p['question'][:100]}")
        return

    if a.cmd == "synth":
        out = gd / "lake" / "synthetic_event.jsonl"; n = 0
        with open(out, "w") as f:
            for p in props:
                if not p.get("resolution_date"):
                    continue
                mech, rel = reliability(p["category"])
                py = p.get("prob_yes")
                ev = {
                    "event_id": "synev_" + p["prop_id"][3:],
                    "source": "polymarket_synth", "contract_ref": p["contract_ref"],
                    "event_date": p["resolution_date"], "title": p["question"],
                    "category": p["category"], "mechanism": mech, "reliability": rel,
                    "symbols": p["symbols"], "volume": p["volume"], "liquidity": p.get("liquidity"),
                    # two scenarios; probability from the market, direction unknown a priori
                    "scenarios": [
                        {"outcome": "yes", "probability": py, "dir": "up"},
                        {"outcome": "no", "probability": (1 - py) if isinstance(py, (int, float)) else None, "dir": "down"},
                    ],
                    "resolution_criteria": p.get("resolution_criteria"),
                }
                f.write(json.dumps(ev) + "\n"); n += 1
        print(f"synthesised {n} forward event nodes -> {out}")
        print("  each: dated catalyst + 2 scenarios (market-implied prob) + borrowed mechanism reliability,")
        print("  linked to US tickers -- consumable by the vol / risk machinery alongside realised news edges.")
        return

    # calendar
    rows = [p for p in props if p.get("resolution_date")]
    if a.edge:
        rows.sort(key=lambda p: -edge_score(p))
        print(f"=== forward calendar :: attention-before-realisation watchlist ===  (top {a.top})")
        print("    (prob moving on rising 24h volume, resolution ahead -- WATCHLIST not a claim)\n")
        for p in rows[:a.top]:
            mech, rel = reliability(p["category"])
            syms = ",".join(p.get("symbols") or []) or "-"
            print(f"  {p.get('resolution_date','?'):10} score {edge_score(p):5.2f}  P {fmt_prob(p)} "
                  f"1d {float(p.get('one_day_price_change') or 0):+.2f}  ${p['volume24hr']/1e6:5.2f}M/24h  "
                  f"[{syms}]  {p['question'][:60]}")
        return

    by_date = collections.defaultdict(list)
    for p in rows:
        by_date[p["resolution_date"]].append(p)
    print(f"=== forward catalyst calendar ===  {len(rows)} markets across {len(by_date)} dates")
    shown = 0
    for d in sorted(by_date):
        if shown >= a.top:
            break
        grp = sorted(by_date[d], key=lambda p: -(p.get("volume") or 0))
        print(f"\n  {d}  ({len(grp)} market{'s' if len(grp) != 1 else ''})")
        for p in grp[:6]:
            mech, rel = reliability(p["category"])
            syms = ",".join(p.get("symbols") or []) or "-"
            print(f"    P {fmt_prob(p)}  ${p['volume']/1e6:6.2f}M  [{p['category']}, {syms}]  {p['question'][:70]}")
            shown += 1


if __name__ == "__main__":
    main()
