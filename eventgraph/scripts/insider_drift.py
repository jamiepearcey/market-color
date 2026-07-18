# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Insider drift: the one inefficiency that fails the OTHER way. Fading the crowd was
efficient everywhere. But in markets where specific people can KNOW the answer early
(nominations, M&A, FDA rulings, court verdicts, resignations), informed money pushes the
price toward truth before the public — and the price often UNDERREACTS to that informed
flow (it leaks in gradually). So the trade is FOLLOW the smart money, not fade the crowd.

Signature: a first-half price MOVE predicts a same-direction second-half move / outcome
(continuation from informed leakage). This must be distinguished from the general
momentum we already showed is a thin-book slippage artifact — so the decisive contrasts:
  * insider-prone categories should show POSITIVE continuation; insider-FREE ones
    (price thresholds, crypto, macro data) should not;
  * and unlike the slippage artifact, informed drift should SURVIVE in LIQUID markets.

Method (distinct endpoints, no shared-endpoint bias): per resolved market,
  move1 = p(mid) - p(3d)         # first-half move (informed accumulation)
  fwd   = outcome - p(mid)       # second-half realised move (what you'd earn following)
continuation IC = cluster-robust Spearman(move1, fwd), split by insider tag & liquidity.

Usage:
  uv run eventgraph/scripts/insider_drift.py --graph-dir eventgraph/data/eg_live
"""
import argparse, json, re, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


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

INSIDER = re.compile(r"\b(nominat|appoint|next (chair|ceo|justice|pope|secretary|coach|manager|director)|"
                     r"acqui|merger|buyout|takeover|acquire|fda|approv|resign|fired|step down|ousted|"
                     r"removed|indict|convict|guilty|verdict|ruling|settle|plea|pardon|ipo|go public|"
                     r"drop out|withdraw|replace)\b", re.I)
INSIDER_FREE = re.compile(r"\b(all[- ]time high|reach \$|hit \$|above \$|below \$|\$[0-9]|price of|"
                          r"cpi|inflation|gdp|payroll|close (above|below|at)|up or down)\b", re.I)


def tag(q):
    if INSIDER_FREE.search(q):
        return "insider_free"
    if INSIDER.search(q):
        return "insider"
    return "other"


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    if len(x) < 8:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_ic(recs):
    """Spearman(move1,fwd) but cluster-robust: bootstrap by underlying for a CI."""
    if len(recs) < 25:
        return None
    x = [r["m1"] for r in recs]; y = [r["fwd"] for r in recs]
    ic = spearman(x, y)
    byu = collections.defaultdict(list)
    for r in recs:
        byu[r["und"]].append(r)
    us = list(byu.values()); rng = np.random.default_rng(7); boot = []
    for _ in range(600):
        idx = rng.integers(0, len(us), len(us))
        samp = [r for i in idx for r in us[i]]
        boot.append(spearman([r["m1"] for r in samp], [r["fwd"] for r in samp]))
    return ic, np.percentile(boot, 5), np.percentile(boot, 95), len(recs), len(byu)


def follow_pnl(recs):
    """tradeable: enter at mid in the direction of move1, hold to resolution. PnL = sign(m1)*fwd,
    cluster-robust mean by underlying."""
    byu = collections.defaultdict(list)
    for r in recs:
        if abs(r["m1"]) < 0.02:
            continue
        byu[r["und"]].append(math.copysign(1, r["m1"]) * r["fwd"])
    if len(byu) < 5:
        return None
    cl = np.array([np.mean(v) for v in byu.values()])
    return cl.mean(), cl.std(ddof=1) / math.sqrt(len(cl)), len(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--early", type=int, default=3, help="first observation age (days after open)")
    ap.add_argument("--max-events", type=int, default=3500)
    ap.add_argument("--max-fetch", type=int, default=6000)
    ap.add_argument("--per-event", type=int, default=8)
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # pull resolved events from Gamma, DATE-WINDOWED to bypass the ~2100 offset cap (each window
    # < cap) so we reach the long tail where insider markets (M&A/FDA/nominations/verdicts) live.
    WINDOWS = [("2023-01-01", "2024-01-01"), ("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"),
               ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-08-01")]
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
    print(f"pulled {len(events)} resolved events (date-windowed)")

    recs = collections.defaultdict(list); fetched = 0
    for e in events:
        legs = 0
        for m in e.get("markets") or []:
            if fetched >= a.max_fetch or legs >= a.per_event:
                break
            q = m.get("question") or ""; t = tag(q)
            if t == "other":
                continue                                    # only insider / insider_free (the control)
            y = settled_yes(m)
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            if y is None or vol < a.min_volume or not clobs or not m.get("startDate") or not m.get("endDate"):
                continue
            try:
                sd = dt.datetime.fromisoformat(m["startDate"].replace("Z", "+00:00"))
                ed = dt.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
            except Exception:
                continue
            hold = (ed.date() - sd.date()).days
            if hold < a.early + 4:
                continue
            h = daily_hist(clobs[0], cache, sess); fetched += 1
            if not h:
                continue
            t0 = int(sd.timestamp())
            p_early = prob_at(h, t0 + a.early * 86400)
            p_mid = prob_at(h, t0 + int(hold / 2) * 86400)
            if p_early is None or p_mid is None or not (0.02 < p_early < 0.98) or not (0.02 < p_mid < 0.98):
                continue
            recs[t].append({"m1": p_mid - p_early, "fwd": y - p_mid, "und": e.get("title") or e.get("id"), "vol": vol})
            legs += 1
        if fetched >= a.max_fetch:
            break
    print(f"fetched {fetched} insider/insider_free markets\n")

    print(f"markets: " + ", ".join(f"{k}={len(v)}" for k, v in recs.items()) + "\n")
    print("CONTINUATION IC = Spearman(first-half move, second-half realised move), cluster-bootstrap CI:")
    for t in ["insider", "insider_free", "other"]:
        ci = cluster_ic(recs.get(t, []))
        if not ci:
            print(f"  {t:13} (too few)"); continue
        ic, lo, hi, n, nu = ci
        excl = "  <-- CI excl 0" if (lo > 0 or hi < 0) else ""
        print(f"  {t:13} IC {ic:+.3f}  90%CI[{lo:+.3f},{hi:+.3f}]  (n={n}, {nu} underlyings){excl}")

    print("\nFOLLOW-THE-MOVE tradeable edge = sign(move1)*fwd, cluster-robust mean by underlying:")
    for t in ["insider", "insider_free", "other"]:
        fp = follow_pnl(recs.get(t, []))
        if not fp:
            print(f"  {t:13} (too few)"); continue
        m, se, nu = fp
        sig = "SIG" if abs(m) > 2 * se else "n.s."
        print(f"  {t:13} edge {m:+.4f} ±{2*se:.4f}  ({nu} underlyings)  {sig}")

    # does insider continuation SURVIVE in liquid? (vs the general momentum slippage artifact)
    ins = recs.get("insider", [])
    if len(ins) >= 40:
        vm = np.median([r["vol"] for r in ins])
        print("\ninsider continuation IC by liquidity (survives in liquid = real informed drift):")
        for lbl, sub in [("liquid (>=median vol)", [r for r in ins if r["vol"] >= vm]),
                         ("thin (<median vol)", [r for r in ins if r["vol"] < vm])]:
            ci = cluster_ic(sub)
            if ci:
                ic, lo, hi, n, nu = ci
                print(f"  {lbl:22} IC {ic:+.3f} 90%CI[{lo:+.3f},{hi:+.3f}] ({nu} und)")


if __name__ == "__main__":
    main()
