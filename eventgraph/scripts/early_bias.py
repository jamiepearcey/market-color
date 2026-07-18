# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Information (not statistical) arbitrage: early, DIRECTIONAL crowd bias in prediction
markets.

Prediction markets clear on BELIEF, and belief is not symmetric noise — it's a bet, and
people bet on hope/hype/drama even when clueless. So the exploitable structure is not in
the price-path shape (churn/ladders/momentum all proved efficient-net-of-cost) but in
CONTENT x YOUTH: young markets, before information arrives, should carry a systematic
DIRECTIONAL mispricing that a well-calibrated observer can fade.

Signed error  b = outcome - p   (outcome∈{0,1})
    b < 0  => market priced yes TOO HIGH  (crowd over-bet the affirmative — hope/hype)
    b > 0  => market priced yes TOO LOW   (crowd too bearish)
We measure b as a function of MARKET AGE (days since open, NOT lead-to-resolution — the
early corner is where belief bias lives before it converges), and by CATEGORY / content,
with Wilson-style CIs. Also: does |error| shrink with age (information arriving), and is
the early error DIRECTIONALLY predictable (fadeable) rather than just noisy.

Usage:
  uv run eventgraph/scripts/early_bias.py --graph-dir eventgraph/data/eg_live
  ... --ages 1,3,7,14,30 --min-volume 10000
"""
import argparse, json, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at


def se_mean(xs):
    xs = np.asarray(xs, float)
    return xs.mean(), (xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def hype_flags(q):
    ql = (q or "").lower()
    tags = []
    if any(k in ql for k in ["all-time high", "all time high", "record", "reach $", "hit $", "reach ", "above $"]):
        tags.append("target/ATH")
    if any(k in ql for k in ["up or down", "higher or lower"]):
        tags.append("coinflip")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--ages", default="3,7,14,30")
    ap.add_argument("--min-volume", type=float, default=10000.0)
    ap.add_argument("--max-markets", type=int, default=3000)
    a = ap.parse_args()
    ages = [int(x) for x in a.ages.split(",")]
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume and r.get("start_date")]
    res.sort(key=lambda r: -(r.get("volume") or 0)); res = res[:a.max_markets]

    recs = []
    for r in res:
        try:
            sd = dt.datetime.fromisoformat(r["start_date"] + "T00:00:00+00:00")
        except Exception:
            continue
        h = hist(r["clob_token_yes"], cache, sess)
        if not h:
            continue
        y = 1 if r["resolved_outcome"] == "yes" else 0
        page = {A: prob_at(h, int((sd + dt.timedelta(days=A)).timestamp())) for A in ages}
        recs.append({"y": y, "p": page, "cat": r.get("category") or "?", "q": r.get("question") or "",
                     "hype": hype_flags(r.get("question")), "vol": r.get("volume") or 0})
    print(f"{len(recs)} resolved markets with open-date + path (vol>={a.min_volume:.0f})")
    base = np.mean([r["y"] for r in recs])
    print(f"base rate yes = {base:.3f}\n")

    # 1) aggregate directional bias by MARKET AGE
    print("1) directional bias by market AGE  (b = outcome - price; <0 = yes OVER-priced / hope-bet):")
    print(f"   {'age(d)':>6} {'n':>5} {'mean price':>10} {'realised':>9} {'bias b':>8} {'±SE':>6}")
    for A in ages:
        v = [(r["p"][A], r["y"]) for r in recs if r["p"].get(A) is not None]
        if len(v) < 20:
            continue
        mp = np.mean([p for p, _ in v]); ry = np.mean([y for _, y in v])
        b, se = se_mean([y - p for p, y in v])
        sig = "  *" if abs(b) > 2 * se else ""
        print(f"   {A:6} {len(v):5} {mp:10.3f} {ry:9.3f} {b:+8.3f} {se:6.3f}{sig}")

    # 2) does |error| shrink with age (information arriving)?
    print("\n2) mean |error| by age (falling = information converging):")
    for A in ages:
        v = [abs(r["y"] - r["p"][A]) for r in recs if r["p"].get(A) is not None]
        if len(v) >= 20:
            print(f"   age {A:3}d: mean|err| {np.mean(v):.3f}  (n={len(v)})")

    # 3) directional bias by CATEGORY at the youngest usable age (the fade signal)
    A0 = ages[0]
    print(f"\n3) directional bias by CATEGORY at age {A0}d  (fade candidates where CI excludes 0):")
    bycat = collections.defaultdict(list)
    for r in recs:
        if r["p"].get(A0) is not None:
            bycat[r["cat"]].append(r["y"] - r["p"][A0])
    for cat, v in sorted(bycat.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 25:
            continue
        b, se = se_mean(v)
        flag = "  <-- CI excl 0" if abs(b) > 2 * se else ""
        print(f"   {cat:12} n={len(v):4}  bias {b:+.3f}  ±{2*se:.3f}{flag}")

    # 4) content: hype/target markets ('reach $X', 'ATH') vs the rest
    print(f"\n4) content bias at age {A0}d (hope/hype 'reach $X / ATH' markets):")
    for label, sel in [("target/ATH", [r for r in recs if "target/ATH" in r["hype"]]),
                       ("coinflip up/down", [r for r in recs if "coinflip" in r["hype"]]),
                       ("all other", [r for r in recs if not r["hype"]])]:
        v = [r["y"] - r["p"][A0] for r in sel if r["p"].get(A0) is not None]
        if len(v) < 20:
            print(f"   {label:18} n={len(v):4} (too few)"); continue
        b, se = se_mean(v)
        flag = "  <-- CI excl 0" if abs(b) > 2 * se else ""
        print(f"   {label:18} n={len(v):4}  bias {b:+.3f}  ±{2*se:.3f}{flag}")

    # 5) DECISIVE: does the early bias SURVIVE in liquid markets? (unlike momentum, which
    #    was a thin-book slippage artifact). If it persists where price CAN move, it's a
    #    genuine belief mispricing = fadeable information arbitrage.
    print(f"\n5) early bias (age {A0}d) by VOLUME quintile — does it survive where price CAN move?")
    vv = [(r["vol"], r["y"] - r["p"][A0]) for r in recs if r["p"].get(A0) is not None and r["vol"] > 0]
    if len(vv) >= 100:
        vols = np.array([math.log(v) for v, _ in vv]); edges = np.unique(np.quantile(vols, np.linspace(0, 1, 6)))
        bidx = np.clip(np.digitize(vols, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = [bb for (v, bb), bi in zip(vv, bidx) if bi == b]
            if len(sel) < 20:
                continue
            m, se = se_mean(sel)
            rng = f"[{math.exp(edges[b]):,.0f},{math.exp(edges[b+1]):,.0f}]"
            flag = "  <-- CI excl 0" if abs(m) > 2 * se else ""
            print(f"   vol {rng:22} n={len(sel):4}  bias {m:+.3f}  ±{2*se:.3f}{flag}")
        print("   -> persists/strengthens in TOP quintile = genuine belief mispricing (fadeable);")
        print("      vanishes in liquid like momentum did = another thin-market artifact")


if __name__ == "__main__":
    main()
