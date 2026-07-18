# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Bucket predictions by EX-ANTE user bias, then fade only where a behavioral MECHANISM
exists — 'a trade is a trade, a punt is a punt'.

A good model does not backtest across averages hunting for a gap; it identifies a priori
WHERE the crowd bets with identity/emotion rather than Bayes, and fades only there. The
canonical case (user): 'people bet on their own team even when they're awful' — the YES
is an IDENTITY, so loyalty over-prices it. We deliberately EXCLUDED those markets at
ingest (the finance filter dropped sports/politics/culture), so this pulls the bias-rich
universe back and tags each market by mechanism:

  loyalty_team  : a specific team/nation winning (sports)  -> fandom over-bets YES
  partisan      : a candidate/party winning/nominated       -> supporters over-bet YES
  dream_lottery : price target / all-time-high / moonshot    -> hope over-bets YES
  celebrity_pop : entertainment/awards/pop-culture person    -> stan over-bets YES
  doom_fear     : recession/crash/war/default drama          -> fear over-bets YES
  neutral_tech  : CPI/Fed/GDP/threshold, no emotional valence -> CONTROL (should be ~0)

Then: fade (short) YES at market age 3d, edge = price - outcome, per bias bucket,
CLUSTER-ROBUST by event (fandom markets within one game co-move). Hypothesis: emotional
buckets show a real positive fade edge; neutral_tech ~0 (efficient = a punt, not a trade).

Usage:
  uv run eventgraph/scripts/bias_buckets.py --graph-dir eventgraph/data/eg_live \
      --max-events 1500 --per-bias 500
"""
import argparse, json, re, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at

GAMMA = "https://gamma-api.polymarket.com"
SPORTS = {"sports", "soccer", "football", "nfl", "nba", "mlb", "nhl", "tennis", "cricket",
          "ufc", "boxing", "f1", "golf", "esports", "fifa world cup", "champions league"}
POLITICS = {"elections", "politics", "global elections", "us election", "president"}
CULTURE = {"pop culture", "entertainment", "awards", "movies", "music", "celebrities", "mention markets"}
WINVERB = re.compile(r"\b(win|beat|defeat|advance|champion|title|cup|final|elected|nominee|nominated|mvp)\b", re.I)
DREAM = re.compile(r"\b(all[- ]time high|record high|reach \$|hit \$|above \$|\$[0-9]|moon|1 ?million)\b", re.I)
DOOM = re.compile(r"\b(recession|crash|collapse|war|nuclear|default|shutdown|pandemic|bankrupt|impeach)\b", re.I)
NEUTRAL = re.compile(r"\b(cpi|inflation|fed|fomc|rate|gdp|jobs|unemployment|payroll|yield|treasury|basis points|bps)\b", re.I)


def tag_bias(tagset, q):
    if tagset & SPORTS and WINVERB.search(q):
        return "loyalty_team"
    if tagset & POLITICS and WINVERB.search(q):
        return "partisan"
    if tagset & CULTURE:
        return "celebrity_pop"
    if NEUTRAL.search(q) and not DREAM.search(q):
        return "neutral_tech"
    if DREAM.search(q):
        return "dream_lottery"
    if DOOM.search(q):
        return "doom_fear"
    if tagset & SPORTS:
        return "loyalty_team"
    if tagset & POLITICS:
        return "partisan"
    return "other"


def resolved_outcome(m):
    try:
        prices = json.loads(m.get("outcomePrices") or "[]"); outs = json.loads(m.get("outcomes") or "[]")
    except Exception:
        return None, None
    if not (outs and prices and len(outs) == len(prices)):
        return None, None
    yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
    try:
        pv = float(prices[yi])
    except Exception:
        return None, None
    if pv >= 0.95:
        return "yes", 1
    if pv <= 0.05:
        return "no", 0
    return None, None                       # not cleanly settled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-events", type=int, default=1500)
    ap.add_argument("--per-bias", type=int, default=500, help="cap markets per bias bucket (by volume)")
    ap.add_argument("--min-volume", type=float, default=10000.0)
    ap.add_argument("--age", type=int, default=3)
    ap.add_argument("--max-hold", type=int, default=120)
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # 1. pull resolved events by volume (ALL domains — do NOT apply the finance filter)
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

    # 2. collect resolved markets, tag bias
    cand = []
    for e in events:
        tags = {t.get("label", "").lower() for t in (e.get("tags") or [])}
        for m in e.get("markets") or []:
            q = m.get("question") or ""
            oc, y = resolved_outcome(m)
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            if oc is None or vol < a.min_volume or not clobs or not m.get("startDate"):
                continue
            cand.append({"bias": tag_bias(tags, q), "y": y, "vol": vol, "token": clobs[0],
                         "start": m.get("startDate"), "res": (m.get("endDate") or "")[:10],
                         "event": e.get("title") or e.get("id"), "q": q})
    bycount = collections.Counter(c["bias"] for c in cand)
    print("candidates by bias:", dict(bycount))

    # 3. cap per bias by volume, fetch paths, compute age-3d fade edge
    rows = collections.defaultdict(list)      # bias -> list of (event, pnl)
    for bias in bycount:
        pool = sorted([c for c in cand if c["bias"] == bias], key=lambda c: -c["vol"])[:a.per_bias]
        for c in pool:
            try:
                sd = dt.datetime.fromisoformat(c["start"].replace("Z", "+00:00"))
                rd = dt.date.fromisoformat(c["res"])
            except Exception:
                continue
            entry = (sd + dt.timedelta(days=a.age)).date()
            if entry >= rd or (rd - entry).days > a.max_hold:
                continue
            h = hist(c["token"], cache, sess)
            if not h:
                continue
            p0 = prob_at(h, int(dt.datetime.combine(entry, dt.time(), dt.timezone.utc).timestamp()))
            if p0 is None or p0 <= 0.02 or p0 >= 0.98:
                continue
            rows[bias].append((c["event"], p0 - c["y"]))     # fade-yes PnL, fixed notional

    # 4. per-bias cluster-robust edge (cluster by event)
    print(f"\nfade-YES edge at age {a.age}d, per bias bucket (cluster-robust by event):")
    print(f"  {'bias':14} {'trades':>6} {'events':>6} {'edge':>8} {'±2SE':>7}  verdict")
    for bias in sorted(rows, key=lambda b: -len(rows[b])):
        v = rows[bias]
        if len(v) < 20:
            continue
        byev = collections.defaultdict(list)
        for ev, pnl in v:
            byev[ev].append(pnl)
        cl = np.array([np.mean(x) for x in byev.values()])
        m = cl.mean(); se = cl.std(ddof=1) / math.sqrt(len(cl)) if len(cl) > 1 else float("nan")
        sig = "SIG (trade)" if abs(m) > 2 * se else "n.s. (punt)"
        print(f"  {bias:14} {len(v):6} {len(cl):6} {m:+8.4f} {2*se:7.4f}  {sig}")
    print("\n  edge>0 = YES over-priced (fandom/hope/fear bets up the affirmative); "
          "emotional buckets SIG vs neutral_tech ~0 = 'trade vs punt' separation")


if __name__ == "__main__":
    main()
