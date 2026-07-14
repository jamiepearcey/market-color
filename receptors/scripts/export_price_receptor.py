# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""Feasibility + training data for a PRICE-GROUNDED receptor.

The target relation is NOT in any single document: "article A preceded a large
move in symbol S". We build it by joining news (with desks + timestamp) to
prices (symbol moves), so it is a genuinely non-self-explaining signal.

Outputs (data/):
  symbols.npy      (Ns, 384) embedded symbol names
  symbols.jsonl    Ns lines: {symbol, name, desks[]}
  price_pairs.jsonl grounded (doc_id, symbol, sign, zscore, move_date) pairs:
                    article in [move_date - LAG, move_date], desks overlap.
"""
import json
from pathlib import Path
import duckdb, numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "data"
FACTS = (ROOT / "facts_work" / "facts.parquet").as_posix()
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()
PRICES = (ROOT / "data" / "prices" / "prices.parquet").as_posix()

Z = 1.5      # |zscore_20d| threshold for a "notable move"
LAG = 3      # article must be published within LAG days before the move

con = duckdb.connect()

# --- symbols: name + desks ---
syms = con.execute(f"""
    SELECT symbol, any_value(name) AS nm, any_value(desks) AS dks
    FROM '{PRICES}' GROUP BY symbol ORDER BY symbol
""").fetchall()
symbols = [{"symbol": s[0], "name": s[1], "desks": list(s[2] or [])} for s in syms]
print(f"symbols: {len(symbols)}")

# --- symbol -> keywords (for a TIGHT, causal-ish article<->symbol match) ---
KW = {
    "10usy.b": ["treasury","yield","bond","fed","federal reserve","interest rate","fomc","rates"],
    "2usy.b": ["treasury","yield","bond","fed","federal reserve","interest rate","fomc","rates"],
    "30usy.b": ["treasury","yield","bond","fed","federal reserve","interest rate","fomc","rates"],
    "^dji": ["dow","dow jones"],
    "^hsi": ["hang seng","hong kong"],
    "^kospi": ["kospi","korea","korean"],
    "^ndq": ["nasdaq"],
    "^nkx": ["nikkei"],
    "^shc": ["shanghai","csi 300","china stocks"],
    "^spx": ["s&p 500","s&p","sp500"],
    "^vix": ["vix","volatility"],
    "audusd": ["australian dollar","aud","australia"],
    "btcusd": ["bitcoin","btc"],
    "cb.f": ["brent","crude","oil"],
    "cl.f": ["wti","crude","oil"],
    "dx.f": ["dollar index","dxy","us dollar","greenback"],
    "ethusd": ["ethereum","ether","eth"],
    "eurusd": ["euro","eur"],
    "gbpusd": ["pound","sterling","gbp"],
    "gc.f": ["gold"],
    "hg.f": ["copper"],
    "ng.f": ["natural gas","lng","henry hub"],
    "pl.f": ["platinum"],
    "rb.f": ["gasoline","rbob","diesel","fuel"],
    "si.f": ["silver"],
    "solusd": ["solana","sol"],
    "usdcny": ["yuan","renminbi","china","cny","pboc"],
    "usdjpy": ["yen","japan","jpy","boj","bank of japan"],
    "usdkrw": ["won","korea","krw"],
}

# --- doc -> entities + publish date (docs.jsonl already has normalized entities) ---
docmeta = {}
for line in (OUT / "docs.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    docmeta[o["doc_id"]] = {"ents": set(o.get("entities", [])), "pdate": o.get("date", "")}
con.execute("CREATE TEMP TABLE docdesk AS SELECT NULL")  # placeholder (unused)

# --- notable price moves within the news window ---
dmin = min(m["pdate"] for m in docmeta.values() if m["pdate"])
dmax = max(m["pdate"] for m in docmeta.values() if m["pdate"])
moves = con.execute(f"""
    SELECT p.symbol, p.date, p.ret_1d, p.zscore_20d
    FROM '{PRICES}' p
    WHERE abs(p.zscore_20d) >= {Z} AND p.date >= '{dmin}' AND p.date <= '{dmax}'
    GROUP BY p.symbol, p.date, p.ret_1d, p.zscore_20d
""").fetchall()
print(f"notable moves (|z|>={Z}) in news window: {len(moves)}")

# --- grounded pairs: article ABOUT the symbol precedes its move (entity match) ---
from datetime import date
def d(s):
    y,m,dd = map(int,str(s)[:10].split("-")); return date(y,m,dd)
def matches(ents, keywords):
    return any(any(kw == e or kw in e or e in kw for e in ents) for kw in keywords)

pairs, by_sym = [], {}
for sym, mdate, ret, z in moves:
    kws = KW.get(sym, [])
    if not kws:
        continue
    md = d(mdate)
    for doc_id, meta_d in docmeta.items():
        if not meta_d["pdate"]:
            continue
        if 0 <= (md - d(meta_d["pdate"])).days <= LAG and matches(meta_d["ents"], kws):
            pairs.append({"doc_id": doc_id, "symbol": sym,
                          "sign": 1 if z > 0 else -1, "zscore": float(z),
                          "move_date": str(mdate)})
            by_sym[sym] = by_sym.get(sym, 0) + 1
print(f"grounded (article ABOUT moved-symbol) pairs: {len(pairs)}")
print("pairs per symbol (top):", sorted(by_sym.items(), key=lambda x:-x[1])[:10])

# --- embed symbol names ---
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = np.array(list(model.embed([s["name"] for s in symbols])), dtype=np.float32)
vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
np.save(OUT / "symbols.npy", vec)
(OUT / "symbols.jsonl").write_text("\n".join(json.dumps(s) for s in symbols) + "\n")
(OUT / "price_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
print(f"wrote symbols.npy {vec.shape}, symbols.jsonl, price_pairs.jsonl")
