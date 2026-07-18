# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Calibration / favorite-longshot test — the canonical prediction-market inefficiency.

Efficient markets are CALIBRATED: things priced at p happen ~p of the time. The
documented, persistent deviation is the FAVORITE-LONGSHOT bias — longshots (low p)
are OVER-priced (realise LESS than priced) and favorites (high p) UNDER-priced. That
is a 'small inefficiency in an on-average-efficient market' of exactly the kind we're
hunting, and unlike churn->vol / ladder-arb (both came back efficient) it is where the
literature actually finds an edge.

Method (resolved Polymarket markets w/ ground-truth outcome + CLOB implied-prob path):
  * take each market's implied prob at LEAD days BEFORE resolution (not the settled 0/1);
  * bin by that prob; per bin realised yes-frequency vs mean predicted prob, with a
    binomial (Wilson) CI; the credible object is a MONOTONIC deviation curve
    (realised-minus-predicted rising in p), not one bin — same discipline as
    conditional_signal.py. Report the favorite-longshot slope + a naive edge.

Usage:
  uv run eventgraph/scripts/calibration_test.py --graph-dir eventgraph/data/eg_live
  ... --lead 7 --max-markets 2500 --min-volume 20000
"""
import argparse, json, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

CLOB = "https://clob.polymarket.com"


def hist(token, cache, sess):
    p = cache / f"{token}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return []
    try:
        r = sess.get(f"{CLOB}/prices-history", params={"market": token, "interval": "max", "fidelity": 1440}, timeout=25)
        h = r.json().get("history", []) if r.status_code == 200 else []
    except Exception:
        h = []
    p.write_text(json.dumps(h)); time.sleep(0.05)
    return h


def prob_at(h, target_epoch):
    """implied prob at the path point nearest to (and at/before) target_epoch."""
    best = None
    for pt in h:
        if pt["t"] <= target_epoch and (best is None or pt["t"] > best["t"]):
            best = pt
    return best["p"] if best else None


def wilson(k, n, z=1.64):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n; d = 1 + z * z / n
    c = ph + z * z / (2 * n); half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--lead", type=int, default=7, help="days before resolution to read the implied prob")
    ap.add_argument("--max-markets", type=int, default=2500)
    ap.add_argument("--min-volume", type=float, default=20000.0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--exclude-daily", action="store_true", help="drop ~50/50 'up or down today' dailies")
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume]
    if a.exclude_daily:
        res = [r for r in res if "up or down" not in (r["question"] or "").lower()]
    res.sort(key=lambda r: -(r.get("volume") or 0))
    res = res[:a.max_markets]
    print(f"{len(res)} resolved markets (vol>={a.min_volume:.0f}); reading implied prob {a.lead}d pre-resolution ...")

    obs = []                                    # (p_lead, y, volume, category)
    for i, r in enumerate(res):
        try:
            rd = dt.datetime.fromisoformat(r["resolution_date"] + "T00:00:00+00:00")
        except Exception:
            continue
        tgt = int((rd - dt.timedelta(days=a.lead)).timestamp())
        h = hist(r["clob_token_yes"], cache, sess)
        p = prob_at(h, tgt)
        if p is None or p <= 0.005 or p >= 0.995:   # need an interior pre-resolution quote
            continue
        obs.append((p, 1 if r["resolved_outcome"] == "yes" else 0, r.get("volume") or 0, r.get("category")))
        if (i + 1) % 500 == 0:
            print(f"  fetched {i+1}/{len(res)} -> {len(obs)} usable")
    if len(obs) < 50:
        print(f"only {len(obs)} usable obs"); return

    P = np.array([o[0] for o in obs]); Y = np.array([o[1] for o in obs])
    base = Y.mean(); brier = float(np.mean((P - Y) ** 2))
    print(f"\n{len(obs)} markets | base rate yes={base:.3f} | Brier={brier:.4f} "
          f"(vs {base*(1-base):.4f} for always-base-rate)\n")
    print(f"  {'prob bin':13} {'n':>5} {'mean p':>7} {'realised':>9} {'dev':>7}  95% CI on realised")
    edges = np.linspace(0, 1, a.bins + 1)
    devs, mids = [], []
    for b in range(a.bins):
        sel = (P >= edges[b]) & (P < edges[b + 1] if b < a.bins - 1 else P <= edges[b + 1])
        n = int(sel.sum())
        if n < 10:
            continue
        mp = float(P[sel].mean()); rf = float(Y[sel].mean()); lo, hi = wilson(int(Y[sel].sum()), n)
        dev = rf - mp; devs.append(dev); mids.append(mp)
        flag = "  <-- excl 0-dev" if (lo > mp or hi < mp) else ""
        print(f"  [{edges[b]:.1f},{edges[b+1]:.1f}]   {n:5} {mp:7.3f} {rf:9.3f} {dev:+7.3f}  [{lo:.3f},{hi:.3f}]{flag}")

    # favorite-longshot = deviation (realised-predicted) RISES with p (neg at low p, pos at high p)
    if len(devs) >= 4:
        rp = np.argsort(np.argsort(mids)); rd = np.argsort(np.argsort(devs))
        slope = float(np.corrcoef(rp, rd)[0, 1])
        print(f"\n  favorite-longshot slope (dev vs p, rank): {slope:+.2f}  "
              f"[{'FL bias present' if slope>=0.6 else 'no monotone FL bias'}]")
        print(f"  low-p bins mean dev {np.mean([d for d,m in zip(devs,mids) if m<0.5]):+.3f} "
              f"(longshots; <0 => overpriced)   "
              f"high-p bins mean dev {np.mean([d for d,m in zip(devs,mids) if m>=0.5]):+.3f} "
              f"(favorites; >0 => underpriced)")
    print(f"\n  [bins tested: {a.bins}; trust the monotone slope + CI-excludes-dev0 bins, not lone cells]")


if __name__ == "__main__":
    main()
