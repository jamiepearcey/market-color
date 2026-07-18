# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
The under-sampled tail: does the early crowd bias SURVIVE and become significant in the
ILLIQUID, low-volume markets arbitrageurs don't bother with — where independent-
underlying count is finally HIGH enough to have statistical power?

Every prior 'edge' died on cluster-robustness because there were only ~30 independent
underlyings (crypto-dominated, liquid). The behavioral thesis predicts the OPPOSITE of
what we tested: the bias should be largest exactly where the market is too small to be
worth arbitraging. So pull resolved markets across ALL categories (no finance filter, no
20k cutoff), read the early price (3d after open, daily cache), and bucket by volume —
cluster-robust by EVENT (each game/election/topic is one independent bet).

Prediction (behavioral): bias grows monotonically as volume falls, SIG in the illiquid
buckets with many events; ~0 in the liquid bucket. Prediction (strong-efficiency): flat
~0 everywhere. Either way it's decisive, because the tail finally has the independent N.

Usage:
  uv run eventgraph/scripts/illiquid_tail.py --graph-dir eventgraph/data/eg_live \
      --max-events 3000 --age 3 --min-volume 500
"""
import argparse, json, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

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


def price_at(h, t0, age):
    tgt = t0 + age * 86400; best = None
    for pt in h:
        if pt["t"] <= tgt + 43200 and (best is None or pt["t"] > best["t"]):
            best = pt
    return best["p"] if best else None


def cluster_edge(sub):
    byev = collections.defaultdict(list)
    for ev, b in sub:
        byev[ev].append(b)
    cl = np.array([np.mean(x) for x in byev.values()])
    if len(cl) < 3:
        return None
    m = cl.mean(); se = cl.std(ddof=1) / math.sqrt(len(cl))
    return m, se, len(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-events", type=int, default=3000)
    ap.add_argument("--age", type=int, default=3)
    ap.add_argument("--min-volume", type=float, default=500.0)
    ap.add_argument("--per-event", type=int, default=6, help="cap legs per event (keep events independent)")
    ap.add_argument("--max-fetch", type=int, default=6000)
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # pull a WIDE net of resolved events (order by volume desc gives a full range incl each
    # event's low-vol legs; the tail comes from within-event legs + smaller events deep in the list)
    events, off = [], 0
    while len(events) < a.max_events:
        try:
            page = sess.get(f"{GAMMA}/events", params={"closed": "true", "limit": 100, "offset": off,
                            "order": "volume", "ascending": "false"}, timeout=30).json()
        except Exception:
            break
        if not isinstance(page, list) or not page:
            break
        events.extend([e for e in page if isinstance(e, dict)]); off += 100
    print(f"pulled {len(events)} resolved events")

    obs = []; fetched = 0
    for e in events:
        legs = 0
        for m in e.get("markets") or []:
            if fetched >= a.max_fetch or legs >= a.per_event:
                break
            y = settled_yes(m)
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
            p = price_at(h, int(sd.timestamp()), a.age)
            if p is None or p <= 0.02 or p >= 0.98:
                continue
            obs.append({"ev": e.get("title") or e.get("id"), "bias": p - y, "vol": vol, "p": p, "y": y})
            legs += 1
        if fetched >= a.max_fetch:
            break
    print(f"{len(obs)} resolved markets with a {a.age}d-after-open price, {fetched} fetched\n")
    if len(obs) < 100:
        print("insufficient"); return

    ov = cluster_edge([(o["ev"], o["bias"]) for o in obs])
    print(f"ALL: early bias (price − outcome) {ov[0]:+.4f} ±{2*ov[1]:.4f}  t={ov[0]/ov[1]:+.2f}  "
          f"({ov[2]} independent events, {len(obs)} markets)  {'SIG' if abs(ov[0])>2*ov[1] else 'n.s.'}\n")

    # bucket by volume (log), low → high — the decisive gradient
    print("early bias by VOLUME bucket (cluster-robust by event), low→high:")
    v = np.array([math.log(o["vol"]) for o in obs]); edges = np.unique(np.quantile(v, np.linspace(0, 1, 6)))
    bidx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        sub = [(o["ev"], o["bias"]) for o, bi in zip(obs, bidx) if bi == b]
        ce = cluster_edge(sub)
        if not ce:
            continue
        m, se, nev = ce
        rng = f"${math.exp(edges[b]):,.0f}-{math.exp(edges[b+1]):,.0f}"
        sig = "SIG" if abs(m) > 2 * se else "n.s."
        print(f"  {rng:20} n={len(sub):4} ev={nev:4}  bias {m:+.4f} ±{2*se:.4f}  t={m/se:+.2f}  {sig}")
    print("\n  bias GROWS as volume falls + SIG in illiquid = behavioral inefficiency survives")
    print("  where un-arbitraged (real but un-capturable); FLAT ~0 = strong efficiency even in the tail.")


if __name__ == "__main__":
    main()
