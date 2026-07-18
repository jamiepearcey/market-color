# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Early-decay curve: how fast is the young-market yes-overpricing arbitraged away?

If the accessible-to-humans window (days) is already the picked-over RESIDUAL, the bias
should be FATTEST at the earliest quote (hours after open) and decay steeply. We fetch
HOURLY CLOB history over the opening month (the cached daily paths are too coarse) and
measure bias = implied - realised at a fine age grid from the first quote outward.

Reads: resolved Polymarket binary markets (outcome label + open date + clob token).
Reports mean bias by age (first-quote, 0.5d, 1d, 2d, 3d, 5d, 7d, 14d, 21d, 30d) + the
timestamp of the FIRST quote (does an MM price it near-fair from second 0, or is there a
fat early window?).

Usage:
  uv run eventgraph/scripts/early_decay.py --graph-dir eventgraph/data/eg_live \
      --min-volume 20000 --max-markets 700
"""
import argparse, json, math, time, datetime as dt
from pathlib import Path
import httpx, numpy as np

CLOB = "https://clob.polymarket.com"
AGES = [0.0, 0.5, 1, 2, 3, 5, 7, 10, 14]      # CLOB caps range*fidelity -> 15d hourly window


def hourly(token, t0, t1, cache, sess, fid=120):
    p = cache / f"{token}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return []
    h = []
    for attempt in range(3):
        try:
            r = sess.get(f"{CLOB}/prices-history",
                         params={"market": token, "startTs": t0, "endTs": t1, "fidelity": fid}, timeout=30)
            if r.status_code == 200:
                h = r.json().get("history", []); break
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    time.sleep(0.12)
    if h:
        p.write_text(json.dumps(h))
    return h


def price_at_age(h, t0, age_days):
    tgt = t0 + age_days * 86400
    best = None
    for pt in h:
        if pt["t"] <= tgt + 1800 and (best is None or pt["t"] > best["t"]):
            best = pt
    return best["p"] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--min-volume", type=float, default=20000.0)
    ap.add_argument("--max-markets", type=int, default=700)
    ap.add_argument("--window", type=int, default=15, help="days after open to fetch hourly (CLOB range cap)")
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist_hr"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume and r.get("start_date")]
    res.sort(key=lambda r: -(r.get("volume") or 0)); res = res[:a.max_markets]

    by_age = {A: [] for A in AGES}
    first_ages, first_bias = [], []
    used = 0
    for i, r in enumerate(res):
        try:
            sd = dt.datetime.fromisoformat(r["start_date"] + "T00:00:00+00:00")
        except Exception:
            continue
        t0 = int(sd.timestamp()); t1 = t0 + a.window * 86400
        h = hourly(r["clob_token_yes"], t0, t1, cache, sess)
        if not h:
            continue
        y = 1 if r["resolved_outcome"] == "yes" else 0
        used += 1
        # first quote
        f = min(h, key=lambda p: p["t"])
        first_ages.append((f["t"] - t0) / 86400.0); first_bias.append(f["p"] - y)
        for A in AGES:
            p = price_at_age(h, t0, A)
            if p is not None and 0.02 < p < 0.98:
                by_age[A].append(p - y)
        if (i + 1) % 200 == 0:
            print(f"  fetched {i+1}/{len(res)} -> {used} usable")

    print(f"\n{used} resolved binary markets, hourly opening-month paths (vol>={a.min_volume:.0f})")
    print(f"first quote appears on average {np.mean(first_ages):.2f}d after open "
          f"(median {np.median(first_ages):.2f}d); first-quote bias {np.mean(first_bias):+.3f}\n")
    print(f"  {'age':>7} {'n':>5} {'bias':>8} {'±SE':>6}   decay")
    base = None
    for A in AGES:
        v = np.array(by_age[A])
        if len(v) < 20:
            continue
        m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
        if base is None:
            base = m
        bar = "#" * max(0, int(abs(m) * 200))
        sig = "*" if abs(m) > 2 * se else " "
        lbl = "first" if A == 0 else f"{A:g}d"
        print(f"  {lbl:>7} {len(v):5} {m:+8.3f} {se:6.3f} {sig} {bar}")
    print("\n  fatter+significant at the EARLIEST ages, decaying = human-accessible window is the")
    print("  picked-over residual; FLAT from the first quote = MMs price near-fair from second 0.")


if __name__ == "__main__":
    main()
