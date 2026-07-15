# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "pyarrow", "numpy"]
# ///
"""
Compact daily market panel for the regime receptor (C2). Reduces
data/prices/prices.parquet to one row per date with a few headline series and a
composite risk-on/off score (equities up + vol down + dollar down = risk-on).

Output: receptors/data/market_daily.jsonl
  {date, spx_ret, vix_ret, dxy_ret, y10_ret, risk_on}
"""
import json
from pathlib import Path
import duckdb, numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "data"
PRICES = ROOT / "data" / "prices" / "prices.parquet"

con = duckdb.connect()
# pivot the headline symbols into daily returns
rows = con.execute(f"""
    SELECT date, symbol, ret_1d
    FROM '{PRICES.as_posix()}'
    WHERE symbol IN ('^spx','^vix','dx.f','10usy.b')
    ORDER BY date
""").fetchall()

by_date = {}
for date, sym, ret in rows:
    by_date.setdefault(date, {})[sym] = (ret if ret is not None else 0.0)

def z(vals):
    a = np.array(vals, dtype=float)
    s = a.std() or 1.0
    return (a - a.mean()) / s

dates = sorted(by_date)
spx = [by_date[d].get('^spx', 0.0) for d in dates]
vix = [by_date[d].get('^vix', 0.0) for d in dates]
dxy = [by_date[d].get('dx.f', 0.0) for d in dates]
y10 = [by_date[d].get('10usy.b', 0.0) for d in dates]
# risk-on composite: equities up, vol down, dollar down
risk = z(spx) - z(vix) - 0.5 * z(dxy)

with open(OUT / "market_daily.jsonl", "w") as f:
    for i, d in enumerate(dates):
        f.write(json.dumps({
            "date": d,
            "spx_ret": spx[i], "vix_ret": vix[i],
            "dxy_ret": dxy[i], "y10_ret": y10[i],
            "risk_on": float(risk[i]),
        }) + "\n")
print(f"wrote {OUT/'market_daily.jsonl'} ({len(dates)} days)")
