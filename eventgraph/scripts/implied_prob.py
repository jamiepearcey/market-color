# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Implied-probability feed (lake/0011_market_quote.sql): free prediction-market
quotes -> eg.market_quote, closing the loop 0004's v_prob_divergence was built
for (narrative probability_annotation vs market-implied prob per proposition).

  kalshi      Kalshi public market data (no auth). YES implied prob from the
              *_dollars fields (already 0..1): mid(yes_bid, yes_ask) if two-sided
              else last_price. --series KXFED,KXCPI... to target specific event
              series (recommended -- the firehose is mostly illiquid).
  polymarket  Polymarket Gamma API (no auth). YES prob = first entry of the
              stringified outcomePrices array; natural key = conditionId.

Both append eg.market_quote rows (market_ref, title, as_of, implied_prob,
instrument, source, source_ref, ingested_at, close_time) to
<graph-dir>/lake/market_quote.jsonl. Append-only + full lineage, same as the
calendar/options feeds. --min-liquidity drops untraded 0-prob markets.

Matching quotes to narrative propositions is the eg.proposition_contract table
(populate from Postgres proposition.contract_ref, or a text-match pass); once
mapped, `INSERT INTO eg.market_implied_prob SELECT * FROM eg.v_market_implied`.

Usage:
  uv run scripts/implied_prob.py kalshi --graph-dir /tmp/eg_6k --series KXFED,KXCPIYOY
  uv run scripts/implied_prob.py polymarket --graph-dir /tmp/eg_6k --query fed,rate --limit 200
"""
import argparse, json, re, datetime as dt
from pathlib import Path
import httpx

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLY = "https://gamma-api.polymarket.com/markets"


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(gd, rows):
    p = Path(gd) / "lake" / "market_quote.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  +{len(rows)} -> {p}")


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0


def do_kalshi(a):
    now = now_iso()
    series = a.series.split(",") if a.series else [None]
    rows, client = [], httpx.Client(timeout=30, headers={"Accept": "application/json"})
    for s in series:
        cursor = None
        while True:
            params = {"limit": 100, "status": a.status}
            if s: params["series_ticker"] = s
            if cursor: params["cursor"] = cursor
            d = client.get(KALSHI, params=params).json()
            for m in d.get("markets", []):
                yb, ya = fnum(m.get("yes_bid_dollars")), fnum(m.get("yes_ask_dollars"))
                last = fnum(m.get("last_price_dollars"))
                prob = (yb + ya) / 2 if (yb > 0 and ya > 0) else last
                if prob < a.min_liquidity:  # untraded / no market
                    continue
                rows.append({"market_ref": m["ticker"], "title": m.get("title"),
                             "as_of": now, "implied_prob": round(prob, 4),
                             "instrument": "kalshi", "source": "kalshi",
                             "source_ref": f"kalshi:{m['ticker']}", "ingested_at": now,
                             "close_time": m.get("close_time")})
            cursor = d.get("cursor")
            if not cursor or not a.paginate:
                break
        print(f"{s or 'ALL'}: running total {len(rows)}")
    append(a.graph_dir, rows)


def do_polymarket(a):
    now = now_iso()
    # word-boundary match so 'rate' doesn't hit 'Pirates' (the 'infratel' trap)
    terms = [re.compile(rf"\b{re.escape(t.strip())}\b", re.I) for t in (a.query.split(",") if a.query else [])]
    rows = []
    with httpx.Client(timeout=30) as client:
        offset = 0
        while offset < a.limit:
            batch = client.get(POLY, params={"limit": 100, "offset": offset, "closed": "false",
                                             "order": "volume", "ascending": "false"}).json()
            if not batch:
                break
            for m in batch:
                q = (m.get("question") or "")
                if terms and not any(rx.search(q) for rx in terms):
                    continue
                try:
                    prices = json.loads(m.get("outcomePrices") or "[]")
                    prob = float(prices[0])   # YES / first outcome
                except (json.JSONDecodeError, IndexError, ValueError):
                    continue
                if prob < a.min_liquidity:
                    continue
                rows.append({"market_ref": m.get("conditionId"), "title": q,
                             "as_of": now, "implied_prob": round(prob, 4),
                             "instrument": "polymarket", "source": "polymarket",
                             "source_ref": f"polymarket:{m.get('conditionId')}", "ingested_at": now,
                             "close_time": m.get("endDate")})
            offset += 100
    append(a.graph_dir, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kalshi"); k.add_argument("--graph-dir", required=True)
    k.add_argument("--series", help="comma list of Kalshi series tickers (e.g. KXFED)")
    k.add_argument("--status", default="open")
    k.add_argument("--paginate", action="store_true", help="follow cursor (all pages)")
    k.add_argument("--min-liquidity", type=float, default=0.001)
    p = sub.add_parser("polymarket"); p.add_argument("--graph-dir", required=True)
    p.add_argument("--query", help="comma keywords to filter question text")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--min-liquidity", type=float, default=0.001)
    a = ap.parse_args()
    {"kalshi": do_kalshi, "polymarket": do_polymarket}[a.cmd](a)
