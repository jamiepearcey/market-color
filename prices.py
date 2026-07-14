#!/usr/bin/env python3
"""market-color daily price layer.

Fetches daily OHLC for the instrument universe in config/instruments.json
(one row per instrument per trading day) and derives the two signals the
brief renderer needs: 1-day return and its trailing-20-observation z-score.
Everything is keyless.

Sources (per instrument, in order, under the default --source auto):
  1. Stooq daily CSV  (https://stooq.com/q/d/l/?s=<sym>&i=d)  -- canonical
     symbols live in instruments.json `symbol`. NOTE: as of 2026-07-01
     stooq.com AND stooq.pl serve a JavaScript proof-of-work challenge to
     non-browser clients on this endpoint; when that is detected the script
     marks Stooq blocked (once, not per-instrument) and falls through.
  2. Yahoo Finance via the `yfinance` package (keyless; handles Yahoo's
     cookie/crumb + TLS requirements which plain httpx cannot) -- tickers
     live in instruments.json `yahoo`.

The parquet `symbol` column is ALWAYS the canonical instruments.json
`symbol` (stooq-style), whichever source actually served the data.

Output (atomic full refresh): data/prices/prices.parquet with columns
  symbol VARCHAR, name VARCHAR, desks LIST<VARCHAR>, date VARCHAR
  (YYYY-MM-DD), open/high/low/close DOUBLE, ret_1d DOUBLE
  (close/prev_close - 1), zscore_20d DOUBLE (ret_1d z-scored over the
  trailing <=20 ret_1d obs including the current one; NULL when fewer
  than 10 obs are available).

Run:
  uv run --with httpx --with pyarrow --with duckdb --with yfinance \
      python prices.py fetch --days 120
  uv run --with httpx --with pyarrow --with duckdb \
      python prices.py movers --desk energy
  ... python prices.py movers --desk fx --date 2026-06-30 --z 1.5 --limit 5

`fetch` prints a JSON summary (per-instrument source + failures, row count,
date range). Per-instrument failures never abort the run. `movers` prints
JSON {desk, date, movers: [...], quiet: [...]}: movers are the desk's
instruments at `date` with |zscore_20d| >= --z (instruments with NULL
zscore fall back to |ret_1d| >= 0.75%), sorted by |zscore| desc; `quiet`
carries ALL remaining desk instruments (latest row at-or-before `date`,
nulls if none) so callers can render "no significant move".
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "instruments.json"
OUT_PATH = ROOT / "data" / "prices" / "prices.parquet"

STOOQ_URL = "https://stooq.com/q/d/l/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20.0
POLITE_DELAY_S = 0.6
RET_FALLBACK_ABS = 0.0075  # |ret_1d| threshold when zscore is NULL


# ---------------------------------------------------------------- config

def load_instruments() -> list[dict]:
    cfg = json.loads(CONFIG_PATH.read_text())
    instruments = cfg["instruments"]
    if not instruments:
        raise SystemExit("config/instruments.json has no instruments")
    return instruments


# ---------------------------------------------------------------- fetch

class StooqBlocked(Exception):
    """Stooq served its JS proof-of-work challenge instead of CSV."""


def fetch_stooq(client, symbol: str, d1: date, d2: date) -> list[dict]:
    """Return [{date, open, high, low, close}] from Stooq's daily CSV."""
    params = {
        "s": symbol,
        "i": "d",
        "d1": d1.strftime("%Y%m%d"),
        "d2": d2.strftime("%Y%m%d"),
    }
    resp = client.get(STOOQ_URL, params=params)
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("<"):
        raise StooqBlocked("stooq returned an HTML JS-challenge page, not CSV")
    if not text.lower().startswith("date,open,high,low,close"):
        raise ValueError(f"stooq returned no data for {symbol!r}: {text[:80]!r}")
    bars = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            bars.append(
                {
                    "date": row["Date"],
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed rows (e.g. 'N/D')
    if not bars:
        raise ValueError(f"stooq CSV for {symbol!r} contained no parseable rows")
    return bars


def fetch_yahoo(ticker: str, d1: date, d2: date) -> list[dict]:
    """Return [{date, open, high, low, close}] via yfinance (keyless)."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yfinance not installed -- run via "
            "`uv run --with httpx --with pyarrow --with duckdb --with yfinance "
            "python prices.py ...`"
        ) from exc
    df = yf.Ticker(ticker).history(
        start=d1.isoformat(),
        end=(d2 + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
    )
    bars: dict[str, dict] = {}
    for idx, row in df.iterrows():
        close = row.get("Close")
        if close is None or (isinstance(close, float) and math.isnan(close)):
            continue
        def _f(v):
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        day = idx.date().isoformat()
        bars[day] = {  # keyed by day => dedupes intraday partial rows
            "date": day,
            "open": _f(row.get("Open")),
            "high": _f(row.get("High")),
            "low": _f(row.get("Low")),
            "close": float(close),
        }
    if not bars:
        raise ValueError(f"yahoo returned no data for {ticker!r}")
    return [bars[k] for k in sorted(bars)]


def derive_signals(bars: list[dict]) -> list[dict]:
    """Sort by date, add ret_1d and zscore_20d (trailing 20 obs, min 10)."""
    bars = sorted(bars, key=lambda b: b["date"])
    rets: list[float | None] = [None]
    for prev, cur in zip(bars, bars[1:]):
        if prev["close"]:
            rets.append(cur["close"] / prev["close"] - 1.0)
        else:
            rets.append(None)
    for i, bar in enumerate(bars):
        bar["ret_1d"] = rets[i]
        z = None
        if rets[i] is not None:
            window = [r for r in rets[max(0, i - 19) : i + 1] if r is not None]
            if len(window) >= 10:
                mean = statistics.fmean(window)
                sd = statistics.stdev(window)
                if sd > 1e-12:
                    z = (rets[i] - mean) / sd
        bar["zscore_20d"] = z
    return bars


def write_parquet(rows: list[dict], out_path: Path) -> None:
    """Atomic full-refresh parquet write (tmp file + os.replace)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("name", pa.string()),
            pa.field("desks", pa.list_(pa.string())),
            pa.field("date", pa.string()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("ret_1d", pa.float64()),
            pa.field("zscore_20d", pa.float64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=out_path.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cmd_fetch(args: argparse.Namespace) -> int:
    import httpx

    instruments = load_instruments()
    d2 = date.today()
    d1 = d2 - timedelta(days=args.days)

    all_rows: list[dict] = []
    ok: list[dict] = []
    failed: list[dict] = []
    stooq_blocked = False

    with httpx.Client(
        timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        for inst in instruments:
            symbol, yahoo = inst["symbol"], inst.get("yahoo")
            bars, source, errors = None, None, []

            if args.source in ("auto", "stooq") and not stooq_blocked:
                try:
                    bars, source = fetch_stooq(client, symbol, d1, d2), "stooq"
                except StooqBlocked as exc:
                    stooq_blocked = True  # challenge is host-wide; stop retrying
                    errors.append(f"stooq: {exc}")
                except Exception as exc:
                    errors.append(f"stooq: {exc}")
                time.sleep(POLITE_DELAY_S)
            elif args.source == "stooq" and stooq_blocked:
                errors.append("stooq: blocked by JS challenge (host-wide)")

            if bars is None and args.source in ("auto", "yfinance") and yahoo:
                try:
                    bars, source = fetch_yahoo(yahoo, d1, d2), "yfinance"
                except Exception as exc:
                    errors.append(f"yfinance({yahoo}): {exc}")
                time.sleep(POLITE_DELAY_S)

            if bars is None:
                failed.append({"symbol": symbol, "error": "; ".join(errors) or "no source attempted"})
                print(f"[fetch] FAIL {symbol}: {'; '.join(errors)}", file=sys.stderr)
                continue

            bars = derive_signals(bars)
            for bar in bars:
                all_rows.append(
                    {"symbol": symbol, "name": inst["name"], "desks": inst["desks"], **bar}
                )
            ok.append({"symbol": symbol, "source": source, "rows": len(bars), "last_date": bars[-1]["date"]})
            print(f"[fetch] ok   {symbol:<9} {source:<8} {len(bars):>4} rows -> {bars[-1]['date']}", file=sys.stderr)

    if not all_rows:
        print(json.dumps({"error": "no instrument returned data", "failed": failed}, indent=2))
        return 1

    write_parquet(all_rows, OUT_PATH)
    dates = [r["date"] for r in all_rows]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out": str(OUT_PATH),
        "days_requested": args.days,
        "instruments_fetched": len(ok),
        "instruments_failed": len(failed),
        "stooq_blocked": stooq_blocked,
        "rows": len(all_rows),
        "date_min": min(dates),
        "date_max": max(dates),
        "ok": ok,
        "failed": failed,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 0  # partial success is still success


# ---------------------------------------------------------------- movers

def cmd_movers(args: argparse.Namespace) -> int:
    import duckdb

    if not OUT_PATH.exists():
        print(json.dumps({"error": f"{OUT_PATH} not found -- run `prices.py fetch` first"}))
        return 1

    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT symbol, name, date, close, ret_1d, zscore_20d
        FROM read_parquet(?)
        WHERE list_contains(desks, ?)
        ORDER BY symbol, date
        """,
        [str(OUT_PATH), args.desk],
    ).fetchall()
    if not rows:
        print(json.dumps({"desk": args.desk, "date": None, "movers": [], "quiet": [],
                          "note": "no instruments mapped to this desk"}))
        return 1

    target = args.date or max(r[2] for r in rows)

    by_symbol: dict[str, list] = {}
    for r in rows:
        by_symbol.setdefault(r[0], []).append(r)

    candidates, quiet = [], []
    for symbol, series in by_symbol.items():
        at_date = next((r for r in series if r[2] == target), None)
        if at_date is not None:
            _, name, _, close, ret, z = at_date
            item = {"symbol": symbol, "name": name, "close": close,
                    "ret_1d": ret, "zscore_20d": z}
            significant = (
                (z is not None and abs(z) >= args.z)
                or (z is None and ret is not None and abs(ret) >= RET_FALLBACK_ABS)
            )
            (candidates if significant else quiet).append((item, at_date[2]))
        else:  # no row at target date: latest at-or-before, else nulls
            prior = [r for r in series if r[2] <= target]
            if prior:
                _, name, d, close, ret, z = prior[-1]
            else:
                name, d, close, ret, z = series[0][1], None, None, None, None
            quiet.append(({"symbol": symbol, "name": name, "close": close,
                           "ret_1d": ret, "zscore_20d": z}, d))

    def sort_key(entry):
        item = entry[0]
        z, ret = item["zscore_20d"], item["ret_1d"]
        # zscore-bearing items rank first (by |z|), NULL-zscore fallbacks after (by |ret|)
        return (0, -abs(z)) if z is not None else (1, -abs(ret or 0.0))

    candidates.sort(key=sort_key)
    movers = [item for item, _ in candidates[: args.limit]]
    quiet.extend(candidates[args.limit :])  # over-limit significants are still reported
    quiet.sort(key=sort_key)
    quiet_out = [dict(item, date=d) for item, d in quiet]

    print(json.dumps({
        "desk": args.desk,
        "date": target,
        "z_threshold": args.z,
        "ret_fallback_abs": RET_FALLBACK_ABS,
        "movers": movers,
        "quiet": quiet_out,
    }, indent=2))
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download daily OHLC -> data/prices/prices.parquet")
    p_fetch.add_argument("--days", type=int, default=120, help="calendar-day lookback (default 120)")
    p_fetch.add_argument("--source", choices=["auto", "stooq", "yfinance"], default="auto",
                         help="auto = stooq first, yfinance fallback (default)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_movers = sub.add_parser("movers", help="biggest |zscore| moves for a desk, as JSON")
    p_movers.add_argument("--desk", required=True,
                          help="desk key (rates|fx|energy|metals|crypto|equities|geopolitics|macro|asia)")
    p_movers.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest date for the desk)")
    p_movers.add_argument("--z", type=float, default=1.0, help="|zscore_20d| significance threshold (default 1.0)")
    p_movers.add_argument("--limit", type=int, default=8, help="max movers returned (default 8)")
    p_movers.set_defaults(func=cmd_movers)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
