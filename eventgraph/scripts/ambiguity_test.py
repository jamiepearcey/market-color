# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Resolution-RULES ambiguity — an edge that lives OUTSIDE the price path (so a price-only
market-maker can't arbitrage it). Humans price the EVENT, not the contract's fine print.
Markets whose resolution criteria are vague/subjective carry unpriced 'rules risk': the
obvious side loses on a technicality more than its price implies, and outcomes surprise
more than clear-rules markets.

Score ambiguity from the market DESCRIPTION (full untruncated rules text, via Gamma) —
subjective/hedge tokens per 100 words, minus clarity anchors (official source / exact
number / URL). Then two cluster-robust tests (by event):
  DISPERSION : mean |price − outcome| by ambiguity bucket — vague rules => more surprises.
  FADE-FAV   : among favorites (entry price >= 0.65), bias = price − outcome by ambiguity —
               vague rules => the favorite is OVER-priced (rules-risk premium unpriced).

Prices: early (first traded price within 7d of open) from the CLOB daily cache (reused).

Usage:
  uv run eventgraph/scripts/ambiguity_test.py --graph-dir eventgraph/data/eg_live
"""
import argparse, json, re, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SUBJ = re.compile(r"\b(deem|considered|reasonabl|discretion|generally|substanti|material|approximat|"
                  r"credible|widely|primar|effectively|significant|appropriate|judgment|judgement|"
                  r"interpret|subjective|intend|attempt|meaningful|notable|clearly|consensus|"
                  r"resolver|ambiguous|dispute|unclear|good faith|spirit of)\b", re.I)
HEDGE = re.compile(r"\b(unless|except|provided that|in the event|however|otherwise|subject to|"
                   r"edge case|if necessary|to the extent|as determined|may be|might)\b", re.I)
CLEAR = re.compile(r"(according to|https?://|\.com|\.gov|official|closing price|exactly|resolves? (yes|no) if|"
                   r"on or before|\d{1,2}:\d{2}|utc|\bet\b|1[- ]minute candle|binance|coinbase|per the)", re.I)


def ambiguity(text):
    words = max(len(re.findall(r"\w+", text)), 1)
    subj = len(SUBJ.findall(text)); hedge = len(HEDGE.findall(text)); clear = len(CLEAR.findall(text))
    return 100.0 * (subj + hedge - 0.5 * clear) / words   # rate per 100 words, length-controlled


def settled(m):
    try:
        prices = json.loads(m.get("outcomePrices") or "[]"); outs = json.loads(m.get("outcomes") or "[]")
    except Exception:
        return None
    if not (outs and prices and len(outs) == len(prices)):
        return None
    yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
    try:
        v = float(prices[yi])
    except Exception:
        return None
    return 1 if v >= 0.95 else (0 if v <= 0.05 else None)


def daily_hist(token, cache, sess):
    p = cache / f"{token}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return []
    h = []
    for att in range(2):
        try:
            r = sess.get(f"{CLOB}/prices-history", params={"market": token, "interval": "max", "fidelity": 1440}, timeout=25)
            if r.status_code == 200:
                h = r.json().get("history", []); break
            time.sleep(0.4)
        except Exception:
            time.sleep(0.4)
    time.sleep(0.05)
    if h:
        p.write_text(json.dumps(h))
    return h


def first_price(h, t0, days=7):
    cand = [pt for pt in h if pt["t"] <= t0 + days * 86400]
    return min(cand, key=lambda pt: pt["t"])["p"] if cand else None


def cluster(sub, key):
    byu = collections.defaultdict(list)
    for ev, v in sub:
        byu[ev].append(v)
    cl = np.array([np.mean(x) for x in byu.values()])
    if len(cl) < 4:
        return None
    return cl.mean(), cl.std(ddof=1) / math.sqrt(len(cl)), len(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-fetch", type=int, default=6000)
    ap.add_argument("--per-event", type=int, default=8)
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--q", type=int, default=5)
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    WINDOWS = [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"),
               ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-08-01")]
    events = []
    for lo, hi in WINDOWS:
        off = 0
        while True:
            try:
                page = sess.get(f"{GAMMA}/events", params={"closed": "true", "limit": 100, "offset": off,
                                "end_date_min": lo + "T00:00:00Z", "end_date_max": hi + "T00:00:00Z"}, timeout=30).json()
            except Exception:
                break
            if not isinstance(page, list) or not page:
                break
            events.extend([e for e in page if isinstance(e, dict)]); off += 100
            if off >= 2000:
                break
    print(f"pulled {len(events)} resolved events")

    obs = []; fetched = 0
    for e in events:
        legs = 0
        for m in e.get("markets") or []:
            if fetched >= a.max_fetch or legs >= a.per_event:
                break
            desc = m.get("description") or ""
            if len(desc) < 60:
                continue
            y = settled(m)
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            if y is None or vol < a.min_volume or not clobs or not m.get("startDate"):
                continue
            try:
                sd = dt.datetime.fromisoformat(m["startDate"].replace("Z", "+00:00"))
            except Exception:
                continue
            h = daily_hist(clobs[0], cache, sess); fetched += 1
            if not h:
                continue
            p = first_price(h, int(sd.timestamp()))
            if p is None or not (0.02 < p < 0.98):
                continue
            obs.append({"amb": ambiguity(desc), "p": p, "y": y, "bias": p - y, "abserr": abs(p - y),
                        "ev": e.get("title") or e.get("id"), "vol": vol})
            legs += 1
        if fetched >= a.max_fetch:
            break
    print(f"{len(obs)} resolved markets with rules text + early price ({fetched} fetched)\n")
    if len(obs) < 100:
        print("insufficient"); return

    amb = np.array([o["amb"] for o in obs])
    edges = np.unique(np.quantile(amb, np.linspace(0, 1, a.q + 1)))
    bidx = np.clip(np.digitize(amb, edges[1:-1]), 0, len(edges) - 2)

    print("1) DISPERSION — mean |price−outcome| by ambiguity bucket (cluster-robust by event):")
    print("   (vague rules => bigger surprises => higher |error|)")
    for b in range(len(edges) - 1):
        sub = [(o["ev"], o["abserr"]) for o, bi in zip(obs, bidx) if bi == b]
        c = cluster(sub, None)
        if c:
            m, se, nev = c
            print(f"   amb∈[{edges[b]:+.2f},{edges[b+1]:+.2f}]  n={len(sub):4} ev={nev:3}  |err| {m:.3f} ±{2*se:.3f}")

    print("\n2) FADE-FAVORITE — bias=price−outcome among favorites (entry p>=0.65), by ambiguity:")
    print("   (vague rules => favorite OVER-priced => bias>0, larger for high ambiguity)")
    favs = [(o, bi) for o, bi in zip(obs, bidx) if o["p"] >= 0.65]
    for b in range(len(edges) - 1):
        sub = [(o["ev"], o["bias"]) for o, bi in favs if bi == b]
        c = cluster(sub, None)
        if c:
            m, se, nev = c
            flag = "  <-- CI excl 0" if abs(m) > 2 * se else ""
            print(f"   amb∈[{edges[b]:+.2f},{edges[b+1]:+.2f}]  n={len(sub):4} ev={nev:3}  bias {m:+.3f} ±{2*se:.3f}{flag}")

    # headline contrast: top vs bottom ambiguity tercile
    lo_amb = [o for o in obs if o["amb"] <= np.percentile(amb, 33)]
    hi_amb = [o for o in obs if o["amb"] >= np.percentile(amb, 67)]
    print(f"\n  low-ambiguity Brier {np.mean([o['bias']**2 for o in lo_amb]):.3f}  vs  "
          f"high-ambiguity Brier {np.mean([o['bias']**2 for o in hi_amb]):.3f}  "
          f"(higher = rules-risk adds surprise)")
    lf = [o for o in lo_amb if o["p"] >= 0.65]; hf = [o for o in hi_amb if o["p"] >= 0.65]
    if lf and hf:
        print(f"  favorite over-pricing: low-amb {np.mean([o['bias'] for o in lf]):+.3f}  vs  "
              f"high-amb {np.mean([o['bias'] for o in hf]):+.3f}  (higher = fade ambiguous favorites)")


if __name__ == "__main__":
    main()
