# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Beta-correlated PEER SUBSETS — "only a subset of the sector actually co-moves".

For each analysed firm, compute pairwise return correlation against every other firm over the
attribution window and keep the empirically co-moving SUBSET (corr >= threshold, capped). This is
what a sector-dominant move should be researched against — not the coarse SPDR-correlation bucket.

Emits {ticker: [peer, ...]} for `retrieval_router.py --peers`.
Reads the price cache produced by the eventgraph attribution scripts (cache-only; no network).

  uv run peer_subsets.py --out data/eg_runs/eg100k_graph/peer_subsets.json
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE / "data" / "eg_runs" / "eg100k_graph"


def read_closes(cache: Path, sym: str) -> dict[str, float]:
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}
    out: dict[str, float] = {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], cl):
            if c is not None:
                out[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception:
        pass
    return out


def logret(cl: dict[str, float], lo: str, hi: str) -> dict[str, float]:
    ds = sorted(d for d in cl if lo <= d <= hi)
    return {ds[i]: math.log(cl[ds[i]] / cl[ds[i - 1]])
            for i in range(1, len(ds)) if cl[ds[i - 1]] > 0 and cl[ds[i]] > 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--out", type=Path, default=DEFAULT_GRAPH / "peer_subsets.json")
    ap.add_argument("--from-date", default="2010-01-01")
    ap.add_argument("--to-date", default="2012-12-31")
    ap.add_argument("--min-corr", type=float, default=0.35)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-overlap", type=int, default=120, help="min common trading days to trust a corr")
    args = ap.parse_args()

    cache = args.graph / "prices"
    attr = json.load(open(args.graph / "attribution.json"))
    firms = [(f["t"], f["sec"]) for f in attr["firms"]]
    sec_of = {t: s for t, s in firms}

    rets = {}
    missing = []
    for t, _ in firms:
        r = logret(read_closes(cache, t), args.from_date, args.to_date)
        if len(r) >= args.min_overlap:
            rets[t] = r
        else:
            missing.append(t)

    tickers = sorted(rets)
    out: dict[str, list[str]] = {}
    meta: dict[str, list] = {}
    for a in tickers:
        ra = rets[a]
        cors = []
        for b in tickers:
            if b == a:
                continue
            common = [d for d in ra if d in rets[b]]
            if len(common) < args.min_overlap:
                continue
            x = np.array([ra[d] for d in common])
            y = np.array([rets[b][d] for d in common])
            if x.std() and y.std():
                c = float(np.corrcoef(x, y)[0, 1])
                if c >= args.min_corr:
                    cors.append((c, b))
        cors.sort(reverse=True)
        chosen = [b for _, b in cors[:args.top_k]]
        out[a] = chosen
        meta[a] = [(b, round(c, 2), "same-sec" if sec_of.get(b) == sec_of.get(a) else "cross-sec")
                   for c, b in cors[:args.top_k]]

    args.out.write_text(json.dumps(out, indent=1))
    # report
    print(f"[peers] {len(tickers)} firms with returns ({len(missing)} skipped: {missing[:8]}{'…' if len(missing) > 8 else ''})")
    sizes = [len(v) for v in out.values()]
    print(f"[peers] subset size: mean {np.mean(sizes):.1f}, median {int(np.median(sizes))}, "
          f"empty {sum(1 for s in sizes if s == 0)}")
    print(f"[peers] wrote {args.out}")
    for t in ("WFC", "BP", "BAC", "AAPL", "VWSYF"):
        if t in meta:
            print(f"  {t} ({sec_of.get(t)}): {meta[t][:6]}")


if __name__ == "__main__":
    main()
