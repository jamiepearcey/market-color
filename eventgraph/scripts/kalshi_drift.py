# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Insider-drift + early-bias tests on KALSHI — where the single-name/event contracts
(earnings, M&A, IPO timing, appointments, economic prints) give MANY independent
underlyings (each company/event = one), the power the Polymarket insider test (65 unds)
lacked.

Two tests per market (settled, with a daily candlestick path):
  * INSIDER-DRIFT continuation: Spearman(p_mid − p_3d, outcome − p_mid), cluster-robust
    by SERIES (company/event type). Positive & surviving = informed leakage you can follow.
  * EARLY-BIAS fade: outcome − p_3d, cluster-robust — does the crowd over/under-price the
    affirmative early (same as Polymarket, now with Kalshi's independent breadth + real
    fees noted).

Tag insider (earnings/KPI/acqui/merger/IPO/nominate/appoint/FDA/approve) vs insider_free
(CPI/inflation/GDP/index-level/price-threshold = scheduled data / underlying-driven).

Public API, paced + 429-retry + cached (kalshi_candles/). Prices normalised to [0,1].

Usage:
  uv run eventgraph/scripts/kalshi_drift.py --graph-dir eventgraph/data/eg_live \
      --per-series 40 --min-dur 7
"""
import argparse, json, re, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

BASE = "https://api.elections.kalshi.com/trade-api/v2"
CATS = ["Companies", "Financials", "Economics", "Politics", "Science and Technology", "Health",
        "World", "Mentions", "Entertainment", "Commodities", "Transportation", "Climate and Weather"]
INSIDER = re.compile(r"\b(kpi|earning|report|revenue|nominat|appoint|acqui|merger|buyout|takeover|"
                     r"fda|approv|ipo|go public|resign|fired|guilty|verdict|settle|launch|deliver)\b", re.I)
INSIDER_FREE = re.compile(r"\b(cpi|inflation|gdp|payroll|unemployment|jobless|index|s&p|nasdaq|russell|"
                          r"dow|ibovespa|kospi|over/under|above|below|between|\$[0-9]|price)\b", re.I)


def ep(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def kget(sess, path, **q):
    for att in range(5):
        try:
            r = sess.get(f"{BASE}{path}", params={k: v for k, v in q.items() if v is not None}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(1.2 * (att + 1)); continue
            return {}
        except Exception:
            time.sleep(0.6 * (att + 1))
    return {}


def cprice(c):
    # prefer an actual TRADED price; the `price` dict is empty when no trades that period.
    pr = c.get("price") or {}
    for f in ("mean_dollars", "close_dollars"):
        if pr.get(f) not in (None, ""):
            v = float(pr[f])
            if 0 < v < 1:
                return v
    # fall back to bid/ask mid ONLY if the spread is tight enough to be a real belief
    # (a 5c-98c phantom spread on an untraded market is meaningless).
    yb = (c.get("yes_bid") or {}).get("close_dollars"); ya = (c.get("yes_ask") or {}).get("close_dollars")
    if yb not in (None, "") and ya not in (None, ""):
        b, a = float(yb), float(ya)
        if 0 < b < a < 1 and (a - b) <= 0.20:
            return (a + b) / 2
    return None


def cspread(c):
    yb = (c.get("yes_bid") or {}).get("close_dollars"); ya = (c.get("yes_ask") or {}).get("close_dollars")
    try:
        b, a = float(yb), float(ya)
        if 0 < b < a < 1:
            return a - b
    except (TypeError, ValueError):
        pass
    return None


def candles(sess, series, ticker, t0, t1, cache):
    f = cache / f"{ticker}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return []
    j = kget(sess, f"/series/{series}/markets/{ticker}/candlesticks", start_ts=t0, end_ts=t1, period_interval=1440)
    cs = j.get("candlesticks", []) if isinstance(j, dict) else []
    out = []
    for c in cs:
        p = cprice(c); t = c.get("end_period_ts") or c.get("ts")
        if p is not None and t and 0 < p < 1:
            out.append({"t": int(t), "p": p, "sp": cspread(c)})
    time.sleep(0.12)
    f.write_text(json.dumps(out))
    return out


def spread_at(cs, tgt):
    best = None
    for c in cs:
        if c["t"] <= tgt + 43200 and (best is None or c["t"] > best["t"]):
            best = c
    return best.get("sp") if best else None


def price_at(cs, tgt):
    best = None
    for c in cs:
        if c["t"] <= tgt + 43200 and (best is None or c["t"] > best["t"]):
            best = c
    return best["p"] if best else None


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    if len(x) < 8:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_boot(recs, key, seed=7, nb=600):
    byu = collections.defaultdict(list)
    for r in recs:
        byu[r["ser"]].append(r)
    us = list(byu.values())
    if len(us) < 5:
        return None
    base = key(recs)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(nb):
        idx = rng.integers(0, len(us), len(us))
        boot.append(key([r for i in idx for r in us[i]]))
    return base, float(np.percentile(boot, 5)), float(np.percentile(boot, 95)), len(recs), len(us)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--per-series", type=int, default=40)
    ap.add_argument("--min-dur", type=int, default=7, help="min market duration (days)")
    ap.add_argument("--early", type=int, default=3)
    ap.add_argument("--max-markets", type=int, default=8000)
    ap.add_argument("--insider-only", action="store_true", help="skip control markets before fetch (faster)")
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "kalshi_candles"; cache.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    series = {}
    for cat in CATS:
        for s in kget(sess, "/series", category=cat).get("series", []):
            if s.get("ticker"):
                series[s["ticker"]] = s.get("title") or ""
    print(f"{len(series)} finance/company/econ series")

    mkcache = gd / "kalshi_markets"; mkcache.mkdir(parents=True, exist_ok=True)

    def settled_markets(ser):
        f = mkcache / f"{ser}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return []
        allm, cursor = [], None
        while len(allm) < a.per_series:
            j = kget(sess, "/markets", series_ticker=ser, status="settled", limit=100, cursor=cursor)
            mk = j.get("markets", []) if isinstance(j, dict) else []
            allm.extend(mk)
            cursor = j.get("cursor") if isinstance(j, dict) else None
            if not cursor or not mk:
                break
        f.write_text(json.dumps(allm[:a.per_series]))
        return allm[:a.per_series]

    recs = collections.defaultdict(list); n = 0
    for si, (ser, title) in enumerate(series.items()):
        if n >= a.max_markets:
            break
        if True:
            mk = settled_markets(ser)
            for m in mk:
                q = " ".join(x for x in [title, m.get("title"), m.get("subtitle")] if x)
                t = "insider" if INSIDER.search(q) else ("insider_free" if INSIDER_FREE.search(q) else "other")
                if a.insider_only and t != "insider":
                    continue
                res = (m.get("result") or "").lower()
                if res not in ("yes", "no") or not m.get("open_time") or not m.get("close_time"):
                    continue
                t0, t1 = ep(m["open_time"]), ep(m["close_time"])
                hold = (t1 - t0) / 86400
                if hold < a.min_dur:
                    continue
                cs = candles(sess, ser, m["ticker"], t0, t1, cache); n += 1
                if len(cs) < 3:
                    continue
                mid_ts = t0 + int(hold / 2) * 86400
                p_e = price_at(cs, t0 + a.early * 86400)
                p_m = price_at(cs, mid_ts)
                if p_e is None or p_m is None or not (0.02 < p_e < 0.98) or not (0.02 < p_m < 0.98):
                    continue
                y = 1 if res == "yes" else 0
                recs[t].append({"m1": p_m - p_e, "fwd": y - p_m, "bias": p_e - y, "ser": ser, "y": y,
                                "pm": p_m, "sp": spread_at(cs, mid_ts), "hold": hold})
        if (si + 1) % 60 == 0:
            print(f"  {si+1}/{len(series)} series, {n} markets fetched")
    print(f"usable: " + ", ".join(f"{k}={len(v)}" for k, v in recs.items()) + f"  ({n} fetched)\n")

    print("INSIDER-DRIFT continuation IC = Spearman(p_mid−p_early, outcome−p_mid), cluster-robust by series:")
    for t in ["insider", "insider_free"]:
        cb = cluster_boot(recs.get(t, []), lambda rs: spearman([r["m1"] for r in rs], [r["fwd"] for r in rs]))
        if not cb:
            print(f"  {t:13} (too few)"); continue
        ic, lo, hi, nn, nu = cb
        excl = "  <-- CI excl 0" if (lo > 0 or hi < 0) else ""
        print(f"  {t:13} IC {ic:+.3f}  90%CI[{lo:+.3f},{hi:+.3f}]  (n={nn}, {nu} series){excl}")

    print("\nFOLLOW-THE-MOVE edge = sign(move1)*fwd, cluster-robust mean by series:")
    for t in ["insider", "insider_free"]:
        sub = [r for r in recs.get(t, []) if abs(r["m1"]) >= 0.02]
        byu = collections.defaultdict(list)
        for r in sub:
            byu[r["ser"]].append(math.copysign(1, r["m1"]) * r["fwd"])
        if len(byu) < 5:
            print(f"  {t:13} (too few)"); continue
        cl = np.array([np.mean(v) for v in byu.values()]); se = cl.std(ddof=1) / math.sqrt(len(cl))
        print(f"  {t:13} edge {cl.mean():+.4f} ±{2*se:.4f}  ({len(cl)} series)  {'SIG' if abs(cl.mean())>2*se else 'n.s.'}")

    # COSTED follow trade: cross HALF the bid-ask to enter + Kalshi fee 0.07*p*(1-p),
    # hold to settlement (no exit fee). Net = sign(m1)*fwd - half_spread - fee.
    print("\nCOSTED follow-the-move (cross half-spread + Kalshi fee, hold to settle), cluster-robust:")
    for t in ["insider", "insider_free"]:
        sub = [r for r in recs.get(t, []) if abs(r["m1"]) >= 0.02 and r.get("sp") is not None]
        byu = collections.defaultdict(list)
        for r in sub:
            fee = 0.07 * r["pm"] * (1 - r["pm"])
            net = math.copysign(1, r["m1"]) * r["fwd"] - r["sp"] / 2 - fee
            byu[r["ser"]].append(net)
        if len(byu) < 5:
            print(f"  {t:13} (too few)"); continue
        cl = np.array([np.mean(v) for v in byu.values()]); se = cl.std(ddof=1) / math.sqrt(len(cl))
        sps = [r["sp"] for r in sub]
        print(f"  {t:13} TAKER net {cl.mean():+.4f} ±{2*se:.4f}  ({len(cl)} series)  "
              f"{'SIG' if abs(cl.mean())>2*se else 'n.s.'}   [median spread {np.median(sps):.3f}]")
        # MAKER: post a limit at mid, pay only the fee (no half-spread) — IF you get filled
        bm = collections.defaultdict(list)
        for r in sub:
            bm[r["ser"]].append(math.copysign(1, r["m1"]) * r["fwd"] - 0.07 * r["pm"] * (1 - r["pm"]))
        cm = np.array([np.mean(v) for v in bm.values()]); sem = cm.std(ddof=1) / math.sqrt(len(cm))
        print(f"  {t:13} MAKER net {cm.mean():+.4f} ±{2*sem:.4f}  ({len(cm)} series)  "
              f"{'SIG' if abs(cm.mean())>2*sem else 'n.s.'}   (limit@mid, fee only)")
        # TIGHT-SPREAD subset: taker net where spread <= 3c (drift may still clear a tight book)
        tight = [r for r in sub if r["sp"] <= 0.03]
        bt = collections.defaultdict(list)
        for r in tight:
            bt[r["ser"]].append(math.copysign(1, r["m1"]) * r["fwd"] - r["sp"] / 2 - 0.07 * r["pm"] * (1 - r["pm"]))
        if len(bt) >= 5:
            ct = np.array([np.mean(v) for v in bt.values()]); set_ = ct.std(ddof=1) / math.sqrt(len(ct))
            print(f"  {t:13} TIGHT net {ct.mean():+.4f} ±{2*set_:.4f}  ({len(ct)} series, {len(tight)} mkts, spread<=3c)  "
                  f"{'SIG' if abs(ct.mean())>2*set_ else 'n.s.'}")

    # SHARPE of the maker follow strategy (hold mid->resolution, ~hold/2 days)
    sub = [r for r in recs.get("insider", []) if abs(r["m1"]) >= 0.02 and r.get("sp") is not None]
    if len(sub) >= 25:
        net = np.array([math.copysign(1, r["m1"]) * r["fwd"] - 0.07 * r["pm"] * (1 - r["pm"]) for r in sub])
        hd = np.array([max(r["hold"] / 2, 1) for r in sub])
        pts = net.mean() / net.std()
        avgh = hd.mean()
        print(f"\nSHARPE (maker follow, insider): per-trade mean {net.mean():+.4f} std {net.std():.3f} "
              f"-> per-trade Sharpe {pts:+.3f} (n={len(net)}, avg hold {avgh:.0f}d)")
        print(f"  NAIVE annualised {pts*math.sqrt(365/avgh):+.2f}  (assumes INDEPENDENT trades — optimistic)")
        # cluster-robust: resample whole SERIES, annualise each resample -> honest CI
        byser = collections.defaultdict(list)
        for r, x in zip(sub, net):
            byser[r["ser"]].append((x, max(r["hold"] / 2, 1)))
        sers = list(byser.values()); rng = np.random.default_rng(3); boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(sers), len(sers))
            xs = np.array([v for i in idx for (v, h) in sers[i]])
            hs = np.array([h for i in idx for (v, h) in sers[i]])
            if xs.std() > 0:
                boot.append(xs.mean() / xs.std() * math.sqrt(365 / hs.mean()))
        print(f"  CLUSTER-ROBUST annualised Sharpe 90%CI [{np.percentile(boot,5):+.2f}, {np.percentile(boot,95):+.2f}] "
              f"(resample by series; {len(sers)} series) {'-> excludes 0' if np.percentile(boot,5)>0 else '-> includes 0'}")

    print("\nEARLY-BIAS fade = price@3d − outcome (>0 = yes overpriced), cluster-robust by series:")
    for t in ["insider", "insider_free"]:
        cb = cluster_boot(recs.get(t, []), lambda rs: float(np.mean([r["bias"] for r in rs])))
        if not cb:
            print(f"  {t:13} (too few)"); continue
        b, lo, hi, nn, nu = cb
        excl = "  <-- CI excl 0" if (lo > 0 or hi < 0) else ""
        print(f"  {t:13} bias {b:+.4f}  90%CI[{lo:+.4f},{hi:+.4f}]  (n={nn}, {nu} series){excl}")


if __name__ == "__main__":
    main()
