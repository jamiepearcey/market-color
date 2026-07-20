# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
CORRELATION-PERSISTENCE SIGNAL — the formalized deliverable (structure x activation).

Validated shape (2010-12, dyadic-robust, sector-controlled): among pairs with elevated
trailing correlation, empirical persistence of the correlation by evidence tier:
    unlinked                 ~20%   (mean-reverts -- diversification comes back)
    shared cause, dormant    ~41%
    shared cause + 1-leg market confirmation   ~63%
    shared cause + both legs confirmed         ~69%   (correlation HOLDS)
A risk manager reads a flag as: "these two positions are correlated for a NAMED reason
the market has confirmed -- don't assume the diversification benefit returns."

  flags     as-of report over a live graph: every elevated-correlation pair with a
            shared cause, its activation tier, calibrated persistence rate, the cause
            entities, edge days + confirmation evidence. JSONL + printed table.
  validate  weekly-scale out-of-sample check on the live window: links formed in the
            first part of the window; forward correlation over the held-out tail;
            persistence by tier vs the unlinked baseline. (SMOKE-SCALE: a ~3-week
            corpus gives a 1-2 week forward window; treat as directional evidence.)

Live returns are proxy-neutralized (ACWI/TLT/UUP/GLD/EEM regression residuals) since
the engine factor panel ends before the live window.

Usage:
  uv run scripts/persistence_signal.py flags    --graph-dir /tmp/eg_live2
  uv run scripts/persistence_signal.py validate --graph-dir /tmp/eg_live2 --split 2026-07-09
"""
import argparse, json, collections, math, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, SKIP

DIR = {"up", "down", "widen", "tighten"}
PROXIES = ["ACWI", "TLT", "UUP", "GLD", "EEM"]
CALIB = {0: 0.20, 1: 0.41, 2: 0.63, 3: 0.69}  # tier -> P(persist), 2010-12 calibration
TIER = {0: "unlinked", 1: "shared-cause (dormant)", 2: "shared-cause + 1-leg confirmed",
        3: "shared-cause + both-legs confirmed"}


def load_graph(gd):
    lake = gd / "lake"
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: (json.loads(l).get("published_at") or "")[:10] for l in open(lake / "document.jsonl")}
    edges = []  # (cause, effect_sym, day, doc_id, chunk_id)
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or e not in sym or not c or j.get("effect_dir") not in DIR: continue
        edges.append((c, sym[e], d, j.get("doc_id"), j.get("chunk_id")))
    return sym, edges


def fetch_returns(names, cache, p1, p2):
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in names}
    return {s: r for s, r in px.items() if len(r) > 40}


def neutralize(px, prox):
    """residual daily returns after regression on proxy returns (live factor model)."""
    pdays = sorted(set.intersection(*[set(prox[p]) for p in prox]))
    out = {}
    for s, r in px.items():
        days = [d for d in sorted(r) if d in set(pdays)]
        if len(days) < 40: continue
        y = np.array([r[d] for d in days]); X = np.column_stack([np.ones(len(days))] + [[prox[p][d] for d in days] for p in prox])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        out[s] = dict(zip(days, res))
    return out


def win_corr(a1, a2, days, min_common=8):
    common = [d for d in days if d in a1 and d in a2]
    if len(common) < min_common: return None
    x = np.array([a1[d] for d in common]); y = np.array([a2[d] for d in common])
    if x.std() == 0 or y.std() == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def vol_z(symn, cache, day):
    p = cache / f"{symn}.json"
    if not p.exists(): return None
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; vol = res["indicators"]["quote"][0].get("volume") or []
    except Exception:
        return None
    dd, lv = [], []
    for t, v in zip(ts, vol):
        if v: dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")); lv.append(math.log(v))
    if day not in dd: return None
    i = dd.index(day)
    if i < 30: return None
    w = np.array(lv[max(0, i-60):i]); sd = w.std()
    return (lv[i] - w.mean()) / sd if sd > 0 else None


def build(gd, split=None):
    """assemble pairs: trailing corr (to split), links + activation, forward corr (after split)."""
    cache = gd / "prices"; cache.mkdir(exist_ok=True)
    sym, edges = load_graph(gd)
    days_used = sorted({d for _, _, d, _, _ in edges})
    d1 = split or days_used[-1]
    names = sorted({s for _, s, _, _, _ in edges})
    p1 = int(dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime.now(dt.UTC).timestamp())
    print(f"{len(names)} US names with causal edges; fetching prices + proxies ...")
    prox_raw = fetch_returns(PROXIES, cache, p1, p2)
    px = fetch_returns(names, cache, p1, p2)
    ar = neutralize(px, {p: prox_raw[p] for p in PROXIES if p in prox_raw})
    print(f"{len(ar)} names with neutralized return series")
    alldays = sorted(set().union(*[set(v) for v in ar.values()]))
    trail_days = [d for d in alldays if d <= d1][-45:]
    fwd_days = [d for d in alldays if d > d1]
    # per-pair: shared causes with edge evidence up to split
    pair_ev = collections.defaultdict(list)   # (i,j) -> [(cause, leg_sym, day, doc, chunk)]
    by_cause = collections.defaultdict(list)
    for c, s, d, doc, ch in edges:
        if s in ar and (split is None or d <= d1): by_cause[c].append((s, d, doc, ch))
    for c, occ in by_cause.items():
        ss = sorted({s for s, _, _, _ in occ})
        for x in range(len(ss)):
            for y in range(x + 1, len(ss)):
                for s, d, doc, ch in occ:
                    if s in (ss[x], ss[y]): pair_ev[(ss[x], ss[y])].append((c, s, d, doc, ch))
    # activation per leg (volume z on edge day; abnormal z from the neutralized series)
    arz = {}
    for s, series in ar.items():
        ds = sorted(series); vals = [series[d] for d in ds]
        for i in range(20, len(ds)):
            w = np.array(vals[max(0, i-60):i]); sd = w.std()
            if sd > 0: arz[(s, ds[i])] = vals[i] / sd
    rows = []
    for pair, ev in pair_ev.items():
        tc = win_corr(ar[pair[0]], ar[pair[1]], trail_days)
        if tc is None: continue
        conf = {pair[0]: False, pair[1]: False}; detail = []
        for c, s, d, doc, ch in ev:
            vz = vol_z(s, cache, d); az = arz.get((s, d))
            ok = (vz is not None and vz >= 2) or (az is not None and abs(az) >= 2)
            conf[s] = conf[s] or ok
            detail.append({"cause": c.split("__")[0], "leg": s, "day": d,
                           "vol_z": None if vz is None else round(vz, 1),
                           "abn_z": None if az is None else round(az, 1), "confirmed": bool(ok),
                           "doc_id": doc, "chunk_id": ch})
        tier = 1 + (1 if conf[pair[0]] else 0) + (1 if conf[pair[1]] else 0)
        fc = win_corr(ar[pair[0]], ar[pair[1]], fwd_days, min_common=5) if fwd_days else None
        rows.append({"pair": list(pair), "trailing_corr": round(tc, 3), "tier": tier,
                     "tier_label": TIER[tier], "persist_calib": CALIB[tier],
                     "causes": sorted({c.split("__")[0] for c, *_ in ev}),
                     "n_edges": len(ev), "evidence": detail,
                     "forward_corr": None if fc is None else round(fc, 3)})
    return rows, ar, trail_days, fwd_days


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["flags", "validate"])
    ap.add_argument("--graph-dir", default="/tmp/eg_live2"); ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--split", default=None, help="validate: link/forward split day YYYY-MM-DD")
    a = ap.parse_args(); gd = Path(a.graph_dir)
    rows, ar, trail_days, fwd_days = build(gd, a.split)

    if a.cmd == "flags":
        flag = [r for r in rows if r["trailing_corr"] >= a.tau]
        flag.sort(key=lambda r: (-r["tier"], -r["trailing_corr"]))
        out = gd / "persistence_flags.jsonl"
        with open(out, "w") as f:
            for r in flag: f.write(json.dumps(r) + "\n")
        print(f"\n=== PERSISTENCE FLAGS (trailing corr >= {a.tau}; window ..{trail_days[-1]}) ===")
        print(f"  {'pair':16} {'corr':>6} {'tier':32} {'P(hold)':>8}  causes")
        for r in flag[:25]:
            print(f"  {'~'.join(r['pair']):16} {r['trailing_corr']:+6.2f} {r['tier_label']:32} {r['persist_calib']:8.0%}  {', '.join(r['causes'][:3])}")
        print(f"\n  {len(flag)} flags -> {out}")
    else:
        if not fwd_days: sys.exit("no forward days after split")
        print(f"\n=== VALIDATE (links to {a.split}, forward {fwd_days[0]}..{fwd_days[-1]}, {len(fwd_days)} days) ===")
        # unlinked baseline: elevated pairs among the same names with no shared cause
        linked_pairs = {tuple(r["pair"]) for r in rows}
        names = sorted(ar); base = []
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                p = (names[x], names[y])
                if p in linked_pairs: continue
                tc = win_corr(ar[p[0]], ar[p[1]], trail_days)
                fc = win_corr(ar[p[0]], ar[p[1]], fwd_days, min_common=5)
                if tc is not None and fc is not None and tc >= a.tau: base.append((tc, fc))
        print(f"  {'category':36} {'n':>5} {'persists':>9} {'fwd_corr':>9} {'from':>7}")
        if base:
            B = np.array(base)
            print(f"  {'unlinked (baseline)':36} {len(B):5} {float(np.mean(B[:,1] >= B[:,0])):9.0%} {B[:,1].mean():+9.3f} {B[:,0].mean():+7.3f}")
        for tier in (1, 2, 3):
            r = [x for x in rows if x["tier"] == tier and x["trailing_corr"] >= a.tau and x["forward_corr"] is not None]
            if not r: continue
            pers = float(np.mean([x["forward_corr"] >= x["trailing_corr"] for x in r]))
            print(f"  {TIER[tier]:36} {len(r):5} {pers:9.0%} {np.mean([x['forward_corr'] for x in r]):+9.3f} {np.mean([x['trailing_corr'] for x in r]):+7.3f}")
        print("\n  calibration (2010-12): unlinked 20% / dormant 41% / 1-leg 63% / both 69%")
        print("  SMOKE-SCALE: ~1.5-week forward window, weekly granularity; directional evidence only.")


if __name__ == "__main__":
    main()
