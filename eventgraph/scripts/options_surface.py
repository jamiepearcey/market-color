# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb", "httpx"]
# ///
"""
Formal-plane options layer (see lake/0006_options.sql): free prepped datasets ->
eg.option_iv (name-level daily IV) + eg.option_surface (underlier surface summary).

  hf-iv   gauss314/options-IV-SP500 (HuggingFace, Apache-2.0): daily moneyness-
          ladder IVs + HV terms + VIX for ~3.9k US symbols, 2019-10..2023-07.
          Downloads (cached), types, writes <graph-dir>/lake/option_iv.parquet.

  chains  Full EOD option chains (e.g. Kaggle 'SPY Options EOD 2010-2023',
          optionsDX) -> one eg.option_surface row per (underlier, date):
          front/next ATM IV, put skew (~0.95-moneyness put minus ATM), term
          slope, ATM-straddle implied move. Column names vary per vendor ->
          map with --col 'ours=theirs' (defaults fit the common Kaggle/optionsDX
          layout). Writes/appends <graph-dir>/lake/option_surface.parquet.
          These datasets need a manual download (Kaggle login) -- point --glob
          at the files.

Both are idempotent full rewrites of their parquet (source data is static
snapshots, not a stream). Load into DuckLake with lake/0006_options.sql then
`INSERT INTO eg.option_iv SELECT * FROM read_parquet('.../option_iv.parquet')`.

Usage:
  uv run scripts/options_surface.py hf-iv  --graph-dir /tmp/eg_6k
  uv run scripts/options_surface.py chains --graph-dir /tmp/eg_6k \
      --glob '~/data/spy_eod/*.parquet' --underlier SPY
"""
import argparse, sys
from pathlib import Path
import duckdb

HF_URL = "https://huggingface.co/datasets/gauss314/options-IV-SP500/resolve/main/data_IV_USA.csv"
CACHE = Path.home() / ".cache" / "eventgraph"

# our canonical chain columns -> common vendor spellings (first match wins)
CHAIN_COLS = {
    "date":      ["quote_date", "QUOTE_DATE", "date", "quotedate", "t_date"],
    "expiry":    ["expiration", "EXPIRE_DATE", "expiry", "expiration_date"],
    "type":      ["option_type", "type", "call_put", "cp_flag", "right"],
    "strike":    ["strike", "STRIKE", "strike_price"],
    "bid":       ["bid", "C_BID", "bid_price", "best_bid"],
    "ask":       ["ask", "C_ASK", "ask_price", "best_offer"],
    "iv":        ["implied_volatility", "C_IV", "iv", "impl_volatility", "implied_vol"],
    "underlying":["underlying_last", "UNDERLYING_LAST", "underlying_price", "active_underlying_price", "spot"],
}


def hf_rev(a):
    # pin the exact dataset revision: HF sha for the main ref (best-effort)
    try:
        import httpx
        sha = httpx.get("https://huggingface.co/api/datasets/gauss314/options-IV-SP500",
                        timeout=15).json().get("sha", "unknown")
    except Exception:
        sha = "unknown"
    return f"{HF_URL}@{sha[:12]}"


def do_hf_iv(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    csv = CACHE / "data_IV_USA.csv"
    if not csv.exists():
        import httpx
        print(f"downloading {HF_URL} ...")
        with httpx.stream("GET", HF_URL, follow_redirects=True, timeout=120) as r, open(csv, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    out = Path(a.graph_dir) / "lake" / "option_iv.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"""
      COPY (SELECT symbol, date,
                   DITM_IV AS ditm_iv, ITM_IV AS itm_iv, sITM_IV AS sitm_iv,
                   ATM_IV AS atm_iv, sOTM_IV AS sotm_iv, OTM_IV AS otm_iv, DOTM_IV AS dotm_iv,
                   calls_contracts_traded AS calls_traded, puts_contracts_traded AS puts_traded,
                   calls_open_interest AS calls_oi, puts_open_interest AS puts_oi,
                   contracts_number AS contracts, expirations_number AS expirations, strikes_spread,
                   hv_20, hv_40, hv_60, hv_90, hv_180, VIX AS vix,
                   'hf:gauss314/options-IV-SP500' AS source,
                   '{hf_rev(a)}' AS source_ref, current_timestamp AS ingested_at
            FROM read_csv_auto('{csv}')
            WHERE ATM_IV IS NOT NULL)
      TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n, d0, d1, s = duckdb.sql(f"SELECT count(*), min(date), max(date), count(DISTINCT symbol) FROM read_parquet('{out}')").fetchone()
    print(f"{out}: {n} rows, {s} symbols, {d0}..{d1}")


def do_chains(a):
    con = duckdb.connect()
    cols = {c.split("=", 1)[0]: c.split("=", 1)[1] for c in (a.col or [])}
    have = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{a.glob}') LIMIT 0"
                                  if a.glob.endswith(".parquet") or "*.parquet" in a.glob
                                  else f"DESCRIBE SELECT * FROM read_csv_auto('{a.glob}') LIMIT 0").fetchall()}
    def col(ours):
        if ours in cols: return cols[ours]
        for cand in CHAIN_COLS[ours]:
            if cand in have: return cand
        sys.exit(f"cannot find a '{ours}' column in {sorted(have)[:20]}... -- map it with --col {ours}=THEIRS")
    src = f"read_parquet('{a.glob}')" if ".parquet" in a.glob else f"read_csv_auto('{a.glob}')"
    c = {k: col(k) for k in CHAIN_COLS}
    out = Path(a.graph_dir) / "lake" / "option_surface.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # one pass: nearest-to-ATM per expiry -> front/next expiry picks -> surface row
    con.sql(f"""
    CREATE TEMP TABLE surf AS
    WITH base AS (
      SELECT CAST({c['date']} AS DATE) AS date, CAST({c['expiry']} AS DATE) AS expiry,
             upper(left(CAST({c['type']} AS VARCHAR),1)) AS cp,
             CAST({c['strike']} AS DOUBLE) AS strike, CAST({c['iv']} AS DOUBLE) AS iv,
             (CAST({c['bid']} AS DOUBLE)+CAST({c['ask']} AS DOUBLE))/2 AS mid,
             CAST({c['underlying']} AS DOUBLE) AS spot
      FROM {src}
      WHERE {c['iv']} IS NOT NULL AND CAST({c['iv']} AS DOUBLE) > 0.005 AND CAST({c['expiry']} AS DATE) > CAST({c['date']} AS DATE)
    ), atm AS (  -- nearest-to-spot strike per (date, expiry, cp)
      SELECT *, row_number() OVER (PARTITION BY date, expiry, cp ORDER BY abs(strike-spot)) AS rk_atm
      FROM base
    ), skewleg AS (  -- ~0.95-moneyness put per (date, expiry)
      SELECT date, expiry, iv AS put95_iv,
             row_number() OVER (PARTITION BY date, expiry ORDER BY abs(strike-0.95*spot)) AS rk
      FROM base WHERE cp = 'P'
    ), per_exp AS (
      SELECT a.date, a.expiry, any_value(a.spot) AS spot,
             avg(a.iv) FILTER (rk_atm=1) AS atm_iv,
             sum(a.mid) FILTER (rk_atm=1) AS straddle,     -- ATM call mid + put mid
             any_value(s.put95_iv) AS put95_iv,
             count(*) AS n
      FROM atm a LEFT JOIN skewleg s ON s.date=a.date AND s.expiry=a.expiry AND s.rk=1
      GROUP BY a.date, a.expiry
      HAVING count(*) FILTER (rk_atm=1) = 2                -- need both legs for the straddle
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY date ORDER BY expiry) AS rk_exp
      FROM per_exp WHERE datediff('day', date, expiry) >= {a.min_dte}
    )
    SELECT '{a.underlier}' AS underlier, f.date, f.spot,
           f.expiry AS front_expiry, datediff('day', f.date, f.expiry) AS front_dte,
           f.atm_iv AS atm_iv_front, n.atm_iv AS atm_iv_next,
           n.atm_iv - f.atm_iv AS term_slope,
           f.put95_iv - f.atm_iv AS put_skew,
           f.straddle / f.spot AS implied_move,
           f.n AS n_contracts, '{a.source}' AS source,
           '{a.glob}' AS source_ref, current_timestamp AS ingested_at
    FROM ranked f LEFT JOIN ranked n ON n.date=f.date AND n.rk_exp=2
    WHERE f.rk_exp = 1""")
    con.sql(f"COPY surf TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n, d0, d1 = con.sql("SELECT count(*), min(date), max(date) FROM surf").fetchone()
    print(f"{out}: {n} surface rows, {d0}..{d1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf-iv"); h.add_argument("--graph-dir", required=True)
    ch = sub.add_parser("chains")
    ch.add_argument("--graph-dir", required=True)
    ch.add_argument("--glob", required=True, help="parquet/csv glob of EOD chain files")
    ch.add_argument("--underlier", required=True)
    ch.add_argument("--source", default="kaggle")
    ch.add_argument("--min-dte", type=int, default=5, help="skip expiries closer than this (weeklies noise)")
    ch.add_argument("--col", action="append", metavar="ours=theirs", help=f"override column mapping; ours in {list(CHAIN_COLS)}")
    a = ap.parse_args()
    {"hf-iv": do_hf_iv, "chains": do_chains}[a.cmd](a)
