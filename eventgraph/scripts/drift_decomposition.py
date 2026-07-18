# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Lead-sensitivity + drift decomposition — settle whether the calibration pocket
(favorites priced 70-90% at 7d out realise 93-98%) is TRADEABLE UNDERREACTION or just
mechanical CONVERGENCE.

Two decisive reads, both off the CACHED CLOB paths (graph-dir/pm_hist/, no refetch):

  1. LEAD-SENSITIVITY: the high-prob (0.7-0.9) realised-minus-predicted gap at leads
     -14/-7/-3/-1d. If the gap SHRINKS toward 0 as lead->0, the -7d underpricing is
     just not-yet-converged (weak: must hold to resolution). If it PERSISTS near
     resolution, it's a real static mispricing.
  2. MOMENTUM / UNDERREACTION: Spearman( Δp[-14,-7] , Δp[-7,-1] ). Positive
     autocorrelation of prob changes = the market UNDERREACTS (a move begets a further
     move) = tradeable momentum. ~0 or negative = efficient / mean-reverting = the
     pocket is just convergence, not edge.

Usage:
  uv run eventgraph/scripts/drift_decomposition.py --graph-dir eventgraph/data/eg_live
  ... --min-volume 15000 --max-markets 3000   # widen (fetches uncached paths)
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


def prob_at(h, target_epoch, tol_days=3):
    """last path point at/before target; None if the nearest is > tol_days away (gap)."""
    best = None
    for pt in h:
        if pt["t"] <= target_epoch and (best is None or pt["t"] > best["t"]):
            best = pt
    if best is None:
        return None
    if (target_epoch - best["t"]) > tol_days * 86400:
        return None
    return best["p"]


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    if len(x) < 8:
        return 0.0, 0
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0, len(x)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def tstat(ic, n):
    return ic * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - ic * ic, 1e-9))


def wilson_half(k, n, z=1.64):
    if n == 0:
        return 0.0
    ph = k / n
    return z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--leads", default="14,7,3,1")
    ap.add_argument("--min-volume", type=float, default=15000.0)
    ap.add_argument("--max-markets", type=int, default=2500)
    a = ap.parse_args()
    leads = [int(x) for x in a.leads.split(",")]
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume]
    res.sort(key=lambda r: -(r.get("volume") or 0))
    res = res[:a.max_markets]
    print(f"{len(res)} resolved markets (vol>={a.min_volume:.0f}); reading cached paths at leads {leads}d\n")

    recs = []
    for i, r in enumerate(res):
        try:
            rd = dt.datetime.fromisoformat(r["resolution_date"] + "T00:00:00+00:00")
        except Exception:
            continue
        h = hist(r["clob_token_yes"], cache, sess)
        if not h:
            continue
        pl = {L: prob_at(h, int((rd - dt.timedelta(days=L)).timestamp())) for L in leads}
        recs.append({"y": 1 if r["resolved_outcome"] == "yes" else 0, "p": pl})
        if (i + 1) % 800 == 0:
            print(f"  processed {i+1}/{len(res)}")

    # 1. LEAD-SENSITIVITY: high-prob (0.7-0.9) realised-minus-predicted at each lead
    print("1) LEAD-SENSITIVITY of the high-prob pocket (predicted 0.70-0.90):")
    print(f"   {'lead':>5} {'n':>5} {'mean p':>7} {'realised':>9} {'dev':>7}  (dev>0 = underpriced)")
    for L in leads:
        band = [(rec["p"][L], rec["y"]) for rec in recs if rec["p"].get(L) is not None and 0.70 <= rec["p"][L] <= 0.90]
        if len(band) < 15:
            print(f"   {L:5} {len(band):5}  (too few)"); continue
        mp = np.mean([p for p, _ in band]); rf = np.mean([y for _, y in band])
        half = wilson_half(int(sum(y for _, y in band)), len(band))
        sig = "  sig" if abs(rf - mp) > half else ""
        print(f"   {L:5} {len(band):5} {mp:7.3f} {rf:9.3f} {rf-mp:+7.3f}  (±{half:.3f}){sig}")
    print("   -> gap SHRINKS toward lead 1 = convergence; PERSISTS = static mispricing")

    # 2. MOMENTUM: does a prior prob move predict the next prob move?
    print("\n2) MOMENTUM / UNDERREACTION  (autocorrelation of prob changes, NON-OVERLAPPING windows):")
    if len(leads) >= 4:
        La, Lb, Lc, Ld = leads[0], leads[1], leads[2], leads[3]   # 14,7,3,1
        # DISTINCT endpoints: prev = p[-Lb]-p[-La] (14->7), next = p[-Ld]-p[-Lc] (3->1).
        # No shared endpoint => no spurious negative autocorrelation from measurement noise.
        pairs = [(rec["p"][Lb] - rec["p"][La], rec["p"][Ld] - rec["p"][Lc])
                 for rec in recs if all(rec["p"].get(L) is not None for L in (La, Lb, Lc, Ld))]
        d_prev = [a for a, _ in pairs]; d_next = [b for _, b in pairs]
        ic, n = spearman(d_prev, d_next)
        print(f"   Spearman( Δp[{La}->{Lb}d] , Δp[{Lc}->{Ld}d] ) = {ic:+.3f}  t≈{tstat(ic,n):+.2f}  (n={n})  [distinct endpoints]")
        print(f"   {'-> POSITIVE = underreaction/momentum (tradeable)' if ic>0.05 else ('-> ~0 = efficient (pocket is just convergence)' if ic>-0.05 else '-> NEGATIVE = mean-reverting')}")

    # 3. does p@-7d UNDER-shoot the eventual outcome, and is drift monotone?
    print("\n3) DRIFT toward outcome by starting band (p@-7d):")
    L7 = 7 if 7 in leads else leads[len(leads) // 2]
    byband = collections.defaultdict(list)
    for rec in recs:
        p = rec["p"].get(L7)
        if p is None:
            continue
        byband[min(int(p * 5), 4)].append(rec["y"] - p)     # realised - predicted (5 bands)
    for b in range(5):
        v = byband.get(b, [])
        if len(v) >= 15:
            print(f"   p@-{L7}d∈[{b/5:.1f},{(b+1)/5:.1f}]  n={len(v):4}  mean(realised-p) {np.mean(v):+.3f}")


if __name__ == "__main__":
    main()
