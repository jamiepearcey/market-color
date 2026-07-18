# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Conditional-signal explorer: markets are efficient ON AVERAGE, but a pooled statistic
integrates over a mixture and washes out the small conditional inefficiencies. This
looks for them by BUCKETING on ex-ante conditioners — with the discipline that makes a
bucket result credible rather than data-mined:

  * conditioners are ECONOMIC and ex-ante (liquidity, disagreement, time-to-resolution,
    vol regime) — never peek at the target;
  * the credible object is a MONOTONIC dose-response across ordered buckets, not one hot
    cell (report Spearman(bucket_index, bucket_signal) = the gradient);
  * bootstrap CI per bucket + explicit count of buckets tried (multiplicity);
  * signal is the PARTIAL IC (feature vs forward vol, controlling recent vol) so we never
    re-discover vol autocorrelation.

Demonstrated on the prediction-market attention data (Polymarket implied-prob paths vs
Yahoo realised vol; same obs as market_vol_signal.py). Feature = prob-churn; target =
forward realised vol; control = recent realised vol. Conditioners: market liquidity,
disagreement (|p-0.5|), days-to-resolution, baseline-vol regime, category.

Usage:
  uv run eventgraph/scripts/conditional_signal.py --graph-dir eventgraph/data/eg_live
  ... --q 5 --h 10 --w 10        # buckets / forward horizon / churn window
"""
import argparse, json, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np


def yahoo_logret(sym, cache, p1, p2):
    p = cache / f"{sym.replace('/', '_')}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                          f"?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            txt = r.text if r.status_code == 200 else ""
        except Exception:
            txt = ""
        p.write_text(txt); time.sleep(0.1)
    cl = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]; ind = res["indicators"]
        px = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], px):
            if c is not None and c > 0:
                cl[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception:
        pass
    ds = sorted(cl)
    return ds, {ds[i]: math.log(cl[ds[i]] / cl[ds[i - 1]]) for i in range(1, len(ds))}


def rv(ds, lr, d0, d1):
    xs = [lr[d] for d in ds if d0 < d <= d1 and d in lr]
    return (float(np.std(xs)) if len(xs) >= 3 else None)


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def partial_ic(x, y, z):
    """rank corr of x,y after removing control z (all in rank space)."""
    if len(x) < 8:
        return 0.0
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    if min(rx.std(), ry.std(), rz.std()) == 0:
        return 0.0
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(x, y, z, seed_rows, nb=400):
    """bootstrap CI of the partial IC by resampling rows (indices in seed_rows)."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n = len(x)
    if n < 12:
        return (float("nan"), float("nan"))
    ics = []
    rng = np.random.default_rng(seed_rows)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        ics.append(partial_ic(x[idx], y[idx], z[idx]))
    return (float(np.percentile(ics, 5)), float(np.percentile(ics, 95)))


def build_rows(gd, w, h):
    lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    props = {json.loads(l)["prop_id"]: json.loads(l) for l in open(lake / "proposition.jsonl") if l.strip()}
    paths = collections.defaultdict(dict)
    for l in open(lake / "proposition_price.jsonl"):
        j = json.loads(l); paths[j["prop_id"]][j["d"]] = j["p"]
    syms = sorted({s for pid in paths for s in (props.get(pid, {}).get("symbols") or [])})
    p1 = int(dt.datetime(2023, 1, 1, tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp())
    hist = {s: yahoo_logret(s, cache, p1, p2) for s in syms}

    rows = []
    for pid, path in paths.items():
        p = props.get(pid)
        if not p or not p.get("symbols") or len(path) < 25:
            continue
        s = p["symbols"][0]; ds, lr = hist.get(s, ([], {}))
        if not ds:
            continue
        pdays = sorted(path); resd = p.get("resolution_date") or pdays[-1]
        vol = float(p.get("volume") or 0.0)
        for T in pdays[w::max(h, 5)]:
            if T >= ds[-1] or T <= ds[h]:
                continue
            if resd and (dt.date.fromisoformat(min(resd, ds[-1])) - dt.date.fromisoformat(T)).days < h:
                continue
            Td = dt.date.fromisoformat(T)
            seg = [(d, path[d]) for d in pdays if (Td - dt.timedelta(days=w)).isoformat() <= d <= T]
            if len(seg) < 4:
                continue
            ai = ds.index(min(ds, key=lambda d: abs((dt.date.fromisoformat(d) - Td).days)))
            fv = rv(ds, lr, T, ds[min(ai + h, len(ds) - 1)])
            bv = rv(ds, lr, ds[max(ai - h, 0)], T)
            if not fv or not bv:
                continue
            churn = float(np.std(np.diff([v for _, v in seg])))
            pT = seg[-1][1]
            dtr = (dt.date.fromisoformat(resd) - Td).days if resd else 999
            rows.append({"s": s, "cat": p.get("category") or "?", "churn": churn,
                         "fv": math.log(fv), "bv": math.log(bv),
                         "liq": math.log(vol + 1), "disagree": -abs(pT - 0.5),
                         "dtr": float(dtr), "basevol": math.log(bv)})
    return rows


def bucket_report(rows, cond, label, q):
    vals = np.array([r[cond] for r in rows])
    # quantile edges (dedup)
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        print(f"  [{label}] too few distinct values"); return
    bidx = np.clip(np.digitize(vals, edges[1:-1]), 0, len(edges) - 2)
    print(f"\n  condition on {label}  ({len(edges)-1} buckets, low→high):")
    bucket_ics = []
    for b in range(len(edges) - 1):
        sel = [r for r, bi in zip(rows, bidx) if bi == b]
        if len(sel) < 15:
            bucket_ics.append(np.nan); print(f"    b{b}: n={len(sel):4}  (too few)"); continue
        x = [r["churn"] for r in sel]; y = [r["fv"] for r in sel]; z = [r["bv"] for r in sel]
        ic = partial_ic(x, y, z); lo, hi = boot_ci(x, y, z, seed_rows=1000 + b)
        bucket_ics.append(ic)
        rng = f"[{np.min([r[cond] for r in sel]):+.2f},{np.max([r[cond] for r in sel]):+.2f}]"
        flag = "  <-- excl 0" if (lo > 0 or hi < 0) else ""
        print(f"    b{b}: n={len(sel):4}  {label}∈{rng:18}  NET IC {ic:+.3f}  90%CI[{lo:+.3f},{hi:+.3f}]{flag}")
    valid = [(i, v) for i, v in enumerate(bucket_ics) if not math.isnan(v)]
    if len(valid) >= 3:
        gi, gv = zip(*valid)
        mono = np.corrcoef(_rank(gi), _rank(gv))[0, 1]
        print(f"    → MONOTONICITY (bucket-order vs IC): {mono:+.2f}  "
              f"[{'gradient' if abs(mono) >= 0.8 else 'weak/none'}] — the credible object, not any single cell")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--q", type=int, default=5); ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    a = ap.parse_args()
    rows = build_rows(Path(a.graph_dir), a.w, a.h)
    print(f"{len(rows)} (market,anchor) obs across {len({r['s'] for r in rows})} instruments")
    print("signal = partial IC(prob-churn, forward-vol | recent-vol); pooled is ~0 — hunting conditional pockets")
    ntested = 0
    for cond, label in [("liq", "market liquidity (log volume)"), ("disagree", "disagreement (-|p-0.5|)"),
                        ("dtr", "days-to-resolution"), ("basevol", "recent-vol regime")]:
        bucket_report(rows, cond, label, a.q)
        ntested += a.q
    # one economic 2D interaction: low-liquidity × high-disagreement (the a-priori inefficiency corner)
    liqm = np.median([r["liq"] for r in rows]); dism = np.median([r["disagree"] for r in rows])
    corner = [r for r in rows if r["liq"] <= liqm and r["disagree"] >= dism]
    if len(corner) >= 15:
        ic = partial_ic([r["churn"] for r in corner], [r["fv"] for r in corner], [r["bv"] for r in corner])
        lo, hi = boot_ci([r["churn"] for r in corner], [r["fv"] for r in corner], [r["bv"] for r in corner], 7)
        print(f"\n  a-priori inefficiency corner (low-liq × high-disagreement): n={len(corner)}  "
              f"NET IC {ic:+.3f}  90%CI[{lo:+.3f},{hi:+.3f}]")
    print(f"\n  [multiplicity: ~{ntested+1} buckets/cells inspected — trust GRADIENTS + CI-excludes-0, "
          f"discount lone cells]")


if __name__ == "__main__":
    main()
