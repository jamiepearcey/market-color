# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi"]
# ///
"""
Daily Yahoo options-chain snapshotter.

Captures an UN-BACKFILLABLE time series: Yahoo only serves the *current*
option chain, so a missed day of history is gone forever. This script grabs
calls+puts across the nearest few expiries for a small watchlist and appends
one dated snapshot per ticker, so we can later compute implied moves
(ATM implied vol / straddle price) and compare them to realized moves around
earnings and macro events.

Why curl_cffi and not plain httpx/requests
-------------------------------------------
Yahoo's v7 options endpoint requires a session cookie + "crumb" token. As of
2026, Yahoo fingerprints the TLS/HTTP2 handshake itself (not just headers)
to hand out that cookie/crumb -- a plain httpx client gets a real HTTP 200
from finance.yahoo.com but only a throwaway "dflow" cookie, and the crumb
endpoint then 401s with "Invalid Cookie". curl_cffi's `impersonate="chrome"`
reproduces a real Chrome TLS fingerprint, which reliably gets the session
cookie the crumb endpoint wants. This was verified empirically before
building this script (plain httpx: 401 Invalid Crumb/Invalid Cookie;
curl_cffi impersonate=chrome: 200 with real chain data).

Usage
-----
    uv run scripts/options_snapshot.py
    uv run scripts/options_snapshot.py --tickers SPY,AAPL,NVDA --max-expiries 3
    uv run scripts/options_snapshot.py --out-dir data/eg_live/options_snapshots --sleep 0.8

Output
------
data/eg_live/options_snapshots/{TICKER}/{YYYY-MM-DD}.jsonl
  one JSON object per line = one option contract, self-describing (carries
  ticker/expiry/spot/capture_ts so no separate header/index is needed).
  Re-running the same day OVERWRITES that day's file only (idempotent
  per ticker/date; history for other days is untouched).

Schedule daily (after US options settle, ~21:15 UTC = 17:15 America/New_York)
------------------------------------------------------------------------------
cron:
    15 21 * * 1-5 cd /Users/jamiepearcey/projects/research/market-color/eventgraph && /opt/homebrew/bin/uv run scripts/options_snapshot.py >> data/eg_live/options_snapshots/cron.log 2>&1
(Do NOT install this automatically -- documented here for the operator to add.)
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

from curl_cffi import requests as cf

# The universe lives in a DATA FILE, not here: widening it is the single highest-
# leverage change for any earnings-vol study and should not require a code edit.
UNIVERSE_FILE = Path(__file__).resolve().parents[1] / "data" / "eg_live" / "options_universe.txt"
FALLBACK_TICKERS = ["SPY", "TLT", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM"]


def default_tickers() -> list[str]:
    if UNIVERSE_FILE.exists():
        out = [l.strip() for l in UNIVERSE_FILE.read_text().splitlines()]
        out = [t for t in out if t and not t.startswith("#")]
        if out:
            return out
    return FALLBACK_TICKERS
IMPERSONATE = "chrome"
CONTRACT_FIELDS = [
    "contractSymbol", "strike", "bid", "ask", "lastPrice",
    "impliedVolatility", "volume", "openInterest", "inTheMoney",
]


class YahooOptionsSession:
    """Cookie+crumb-authenticated session against the v7 options endpoint."""

    def __init__(self):
        self.s = cf.Session(impersonate=IMPERSONATE)
        self.crumb = None
        self.warm_error = None

    def warm(self):
        """Two-step dance: load a finance.yahoo.com page to get a session
        cookie, then trade it for a crumb. Both steps need the browser TLS
        fingerprint (impersonate=chrome) or Yahoo rejects the cookie."""
        try:
            self.s.get("https://finance.yahoo.com/quote/SPY/", timeout=20)
            r = self.s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=20)
        except Exception as e:
            self.warm_error = f"network error during cookie/crumb warmup: {e}"
            return False
        if r.status_code != 200 or not r.text or r.text.lstrip().startswith("{"):
            self.warm_error = f"crumb endpoint returned http {r.status_code}: {r.text[:150]!r}"
            return False
        self.crumb = r.text.strip()
        return True

    def chain(self, ticker, date=None):
        """Fetch one optionChain result (expirations + one expiry's calls/puts
        + underlying quote). Returns (result_dict, error_str)."""
        params = {}
        if self.crumb:
            params["crumb"] = self.crumb
        if date:
            params["date"] = date
        url = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
        try:
            r = self.s.get(url, params=params, timeout=20)
        except Exception as e:
            return None, f"network error: {e}"
        if r.status_code == 401:
            return None, f"http 401 unauthorized: {r.text[:150]}"
        if r.status_code == 429:
            return None, "http 429 rate-limited"
        if r.status_code != 200:
            return None, f"http {r.status_code}: {r.text[:150]}"
        try:
            j = r.json()
        except Exception:
            return None, "response was not valid JSON"
        result = j.get("optionChain", {}).get("result")
        if not result:
            err = j.get("optionChain", {}).get("error")
            return None, f"empty optionChain.result (error={err})"
        return result[0], None


def epoch_to_date(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.UTC).strftime("%Y-%m-%d")


def extract_rows(ticker, res, capture_ts, spot):
    """Flatten one optionChain result (one expiry) into per-contract rows."""
    if not res.get("options"):
        return []
    opt = res["options"][0]
    expiry_epoch = opt.get("expirationDate")
    expiry = epoch_to_date(expiry_epoch) if expiry_epoch else None
    rows = []
    for side, key in (("call", "calls"), ("put", "puts")):
        for c in opt.get(key, []):
            row = {
                "ticker": ticker,
                "capture_ts": capture_ts,
                "expiry": expiry,
                "expiry_epoch": expiry_epoch,
                "spot": spot,
                "contract_type": side,
            }
            for f in CONTRACT_FIELDS:
                row[f] = c.get(f)
            rows.append(row)
    return rows


def atm_iv(rows, spot, contract_type="call"):
    """ATM implied vol for the front expiry: contract with strike nearest spot."""
    cands = [r for r in rows if r["contract_type"] == contract_type and r.get("strike") is not None
             and r.get("impliedVolatility") is not None]
    if not cands or spot is None:
        return None
    best = min(cands, key=lambda r: abs(r["strike"] - spot))
    return best["impliedVolatility"], best["strike"]


def capture_ticker(sess, ticker, max_expiries, sleep_s):
    capture_ts = dt.datetime.now(dt.UTC).isoformat()
    res0, err = sess.chain(ticker)
    if err:
        return None, err
    quote = res0.get("quote", {}) or {}
    spot = quote.get("regularMarketPrice")
    expirations = res0.get("expirationDates") or []
    if not expirations:
        return None, "no expirationDates in response"
    expirations = sorted(expirations)[:max_expiries]

    all_rows = extract_rows(ticker, res0, capture_ts, spot)
    front_expiry_epoch = expirations[0] if expirations else None
    # res0 already covers the first expiry only if it matches expirations[0]
    got_epochs = {r["expiry_epoch"] for r in all_rows}

    for exp in expirations:
        if exp in got_epochs:
            continue
        time.sleep(sleep_s)
        res_e, err_e = sess.chain(ticker, date=exp)
        if err_e:
            print(f"    [{ticker}] expiry {epoch_to_date(exp)}: {err_e}")
            continue
        all_rows.extend(extract_rows(ticker, res_e, capture_ts, spot))

    return {
        "capture_ts": capture_ts,
        "spot": spot,
        "expirations": expirations,
        "rows": all_rows,
        "front_expiry_epoch": front_expiry_epoch,
    }, None


def write_snapshot(out_dir, ticker, day, rows):
    tdir = out_dir / ticker
    tdir.mkdir(parents=True, exist_ok=True)
    fp = tdir / f"{day}.jsonl"
    # overwrite: idempotent per (ticker, date) -- re-running today replaces
    # today's file only, never touches other days' history.
    with fp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None,
                    help="comma-separated; defaults to options_universe.txt")
    ap.add_argument("--max-expiries", type=int, default=4)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "eg_live" / "options_snapshots"))
    ap.add_argument("--sleep", type=float, default=0.8, help="seconds between per-expiry/per-ticker requests")
    a = ap.parse_args()
    tickers = ([t.strip().upper() for t in a.tickers.split(",") if t.strip()]
               if a.tickers else default_tickers())
    out_dir = Path(a.out_dir)
    day = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    sess = YahooOptionsSession()
    if not sess.warm():
        print("YAHOO OPTIONS ENDPOINT BLOCKED")
        print(f"  cookie/crumb negotiation failed: {sess.warm_error}")
        print("  Workaround needed: a real-browser cookie jar (this script already tries")
        print("  curl_cffi impersonate=chrome; if that itself is blocked, try refreshing")
        print("  from an unrelated IP, or pull a browser-exported yahoo.com cookie file, or")
        print("  switch to a delayed-data vendor, e.g. CBOE DataShop / Polygon.io / Tradier).")
        print("  No snapshot was written this run.")
        return

    print(f"yahoo options session warmed (crumb ok)  tickers={tickers}  max_expiries={a.max_expiries}  day={day}")
    summary = []
    for i, ticker in enumerate(tickers):
        if i:
            time.sleep(a.sleep)
        cap, err = capture_ticker(sess, ticker, a.max_expiries, a.sleep)
        if err:
            print(f"  {ticker:6} FAILED: {err}")
            summary.append((ticker, None, None, None, None))
            continue
        rows = cap["rows"]
        fp = write_snapshot(out_dir, ticker, day, rows)
        front_rows = [r for r in rows if r["expiry_epoch"] == cap["front_expiry_epoch"]]
        iv_res = atm_iv(front_rows, cap["spot"], "call") or atm_iv(front_rows, cap["spot"], "put")
        iv, strike = iv_res if iv_res else (None, None)
        n_exp = len({r["expiry_epoch"] for r in rows})
        summary.append((ticker, cap["spot"], n_exp, len(rows), iv))
        iv_s = f"{iv:.1%}" if iv is not None else "n/a"
        print(f"  {ticker:6} spot={cap['spot']!s:>10}  expiries={n_exp}  contracts={len(rows):4}  "
              f"front-ATM IV={iv_s} (K={strike})  -> {fp}")

    ok = [s for s in summary if s[1] is not None]
    print(f"\ndone: {len(ok)}/{len(tickers)} tickers captured, "
          f"{sum(s[3] for s in ok)} total contracts, day={day}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
