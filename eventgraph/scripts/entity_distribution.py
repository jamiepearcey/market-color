# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Per-entity implied DISTRIBUTIONS — find where opposing-fan bias does NOT cancel.

Pairwise fandom cancels: two fanbases on opposite sides of ONE contract net out. But a
multi-outcome market (championship winner, nomination field, price ladder) is a set of
digital options = an implied probability DISTRIBUTION over an entity space, and there the
beloved entity is overbet to win against the whole FIELD — its fans have no single
counterparty, so the bias survives. The diffuse, unloved field absorbs the underpricing.

Polymarket 'negRisk' / multi-market events give this for free (mutually exclusive legs,
exactly one resolves yes). For each resolved multi-leg event we:
  * read each leg's implied prob at LEAD days before the event ends (fan-sentiment window),
    NORMALISE across legs -> the implied distribution (removes the non-uniform overround);
  * measure per-leg bias = normalised prob - realised outcome (0/1);
  * condition on FAVORITISM (prob rank) and POPULARITY (the leg's share of event volume =
    how lopsided the crowd's MONEY is on that entity), and on event CONCENTRATION.

Hypothesis: favorite / high-volume-share legs are OVER-priced (fade), the field UNDER-
priced, and the effect grows with crowd lopsidedness — the distribution-level fan bias
that pairwise markets hide.

Usage:
  uv run eventgraph/scripts/entity_distribution.py --graph-dir eventgraph/data/eg_live \
      --max-events 800 --lead 14 --min-legs 4
"""
import argparse, json, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at

GAMMA = "https://gamma-api.polymarket.com"


def settled_yes(m):
    try:
        prices = json.loads(m.get("outcomePrices") or "[]"); outs = json.loads(m.get("outcomes") or "[]")
    except Exception:
        return None
    if not (outs and prices and len(outs) == len(prices)):
        return None
    yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
    try:
        pv = float(prices[yi])
    except Exception:
        return None
    return 1 if pv >= 0.95 else (0 if pv <= 0.05 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-events", type=int, default=800)
    ap.add_argument("--lead", type=int, default=14, help="days before event end to read the distribution")
    ap.add_argument("--anchor-age", type=int, default=0, help="if >0, read at event OPEN + N days "
                    "(EARLY-life window where fan bias lives) instead of end-lead")
    ap.add_argument("--min-legs", type=int, default=4)
    ap.add_argument("--max-legs", type=int, default=40)
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    events, off = [], 0
    while len(events) < a.max_events:
        try:
            page = sess.get(f"{GAMMA}/events", params={"closed": "true", "limit": 100, "offset": off,
                            "order": "volume", "ascending": "false"}, timeout=30).json()
        except Exception:
            break
        if not page:
            break
        events.extend(page); off += 100
    print(f"pulled {len(events)} resolved events")

    legs = []                       # per-leg records across all usable multi-outcome events
    nev = 0
    for e in events:
        ms = e.get("markets") or []
        if not (a.min_legs <= len(ms) <= a.max_legs):
            continue
        outc = {id(m): settled_yes(m) for m in ms}
        wins = [m for m in ms if outc[id(m)] == 1]
        # mutually-exclusive resolved field: exactly one winner, rest resolved no
        if len(wins) != 1 or any(outc[id(m)] is None for m in ms):
            continue
        try:
            end = dt.datetime.fromisoformat((e.get("endDate") or "").replace("Z", "+00:00"))
            if a.anchor_age > 0:
                start = dt.datetime.fromisoformat((e.get("startDate") or "").replace("Z", "+00:00"))
                anchor = start + dt.timedelta(days=a.anchor_age)
                if anchor >= end:                       # skip events shorter than the anchor window
                    continue
                tgt = int(anchor.timestamp())
            else:
                tgt = int((end - dt.timedelta(days=a.lead)).timestamp())
        except Exception:
            continue
        raw = {}
        vols = {}
        for m in ms:
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            if not clobs:
                break
            p = prob_at(hist(clobs[0], cache, sess), tgt, tol_days=7)
            if p is None:
                break
            raw[id(m)] = p
            try:
                vols[id(m)] = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vols[id(m)] = 0.0
        if len(raw) != len(ms):
            continue
        s = sum(raw.values())
        if s <= 0:
            continue
        vtot = sum(vols.values()) or 1.0
        norm = {k: v / s for k, v in raw.items()}
        ranks = sorted(norm, key=lambda k: -norm[k])
        conc = sum(x * x for x in norm.values())        # Herfindahl of the distribution (lopsidedness)
        nev += 1
        for rank, k in enumerate(ranks):
            m = next(mm for mm in ms if id(mm) == k)
            legs.append({"p": norm[k], "raw": raw[k], "won": outc[k], "rank": rank,
                         "is_fav": rank == 0, "volshare": vols[k] / vtot, "conc": conc,
                         "event": e.get("title") or e.get("id")})
    print(f"usable multi-outcome events: {nev}  |  legs: {len(legs)}\n")
    if len(legs) < 40:
        print("insufficient"); return

    def edge(sub, label):
        if len(sub) < 15:
            print(f"  {label:34} n={len(sub):4} (too few)"); return
        byev = collections.defaultdict(list)
        for l in sub:
            byev[l["event"]].append(l["p"] - l["won"])     # bias = implied - realised
        cl = np.array([np.mean(x) for x in byev.values()])
        m = cl.mean(); se = cl.std(ddof=1) / math.sqrt(len(cl)) if len(cl) > 1 else float("nan")
        sig = "SIG" if abs(m) > 2 * se else "n.s."
        print(f"  {label:34} n={len(sub):4} ev={len(cl):3}  bias {m:+.4f} ±{2*se:.4f}  {sig}")

    print("per-leg bias = implied prob - realised (>0 = OVER-priced = fade), cluster-robust by event:")
    edge([l for l in legs if l["is_fav"]], "FAVORITE leg (rank 0)")
    edge([l for l in legs if l["rank"] == 1], "2nd favorite")
    edge([l for l in legs if l["rank"] >= 3], "the FIELD (rank>=3)")
    print()
    # popularity: split favorites by volume-share (how lopsided the money is on that entity)
    favs = [l for l in legs if l["is_fav"]]
    if favs:
        vs_med = np.median([l["volshare"] for l in favs])
        edge([l for l in favs if l["volshare"] >= vs_med], "favorite w/ HIGH volume-share")
        edge([l for l in favs if l["volshare"] < vs_med], "favorite w/ low volume-share")
    print()
    # lopsidedness: favorites in the most vs least concentrated (Herfindahl) distributions
    if favs:
        c_med = np.median([l["conc"] for l in favs])
        edge([l for l in favs if l["conc"] >= c_med], "favorite in LOPSIDED distribution")
        edge([l for l in favs if l["conc"] < c_med], "favorite in even distribution")
    print("\n  fade the high-attention FAVORITE, hold the FIELD — the distribution-level")
    print("  fan bias that pairwise (A-vs-B) markets cancel out.")


if __name__ == "__main__":
    main()
