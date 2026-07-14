#!/usr/bin/env python3
"""Hourly OHLC for all instruments -> data/prices/prices_hourly.parquet.

The intraday event anchor for the EM prospective benchmark: hourly bars give
exact spike times (|z| on 1h returns), news gives attribution times — the
move never needs to be in print. yfinance 1h bars reach back ~730 days.

  uv run --with yfinance --with pyarrow --with pandas python fetch_hourly.py \
      [--days 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "prices" / "prices_hourly.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()

    import pandas as pd
    import yfinance as yf

    cfg = HERE / "config" / "instruments.json"
    if not cfg.exists():
        cfg = HERE / "instruments.json"
    ins = json.load(open(cfg))
    items = ins if isinstance(ins, list) else ins.get("instruments", [])

    frames = []
    for it in items:
        yt = it.get("yahoo")
        if not yt:
            continue
        try:
            df = yf.Ticker(yt).history(period=f"{args.days}d", interval="1h",
                                       auto_adjust=False)
        except Exception as exc:
            print(f"  {it['symbol']:<10} FAIL {exc}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"  {it['symbol']:<10} empty", file=sys.stderr)
            continue
        df = df.reset_index()
        tcol = "Datetime" if "Datetime" in df.columns else "Date"
        out = pd.DataFrame({
            "symbol": it["symbol"],
            "ts_utc": pd.to_datetime(df[tcol], utc=True),
            "open": df["Open"], "high": df["High"],
            "low": df["Low"], "close": df["Close"],
        }).dropna(subset=["close"])
        out["ret_1h"] = out["close"].pct_change()
        mu = out["ret_1h"].rolling(120, min_periods=40).mean()
        sd = out["ret_1h"].rolling(120, min_periods=40).std()
        out["zscore_1h"] = (out["ret_1h"] - mu) / sd
        frames.append(out)
        print(f"  {it['symbol']:<10} bars={len(out)} "
              f"span={out.ts_utc.min():%m-%d}..{out.ts_utc.max():%m-%d}",
              file=sys.stderr)
    if not frames:
        sys.exit("no data fetched")
    allb = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allb.to_parquet(OUT, index=False)
    spikes = int((allb.zscore_1h.abs() >= 3).sum())
    print(json.dumps({"instruments": int(allb.symbol.nunique()),
                      "bars": len(allb), "abs_z_ge_3": spikes,
                      "out": str(OUT)}))


if __name__ == "__main__":
    main()
