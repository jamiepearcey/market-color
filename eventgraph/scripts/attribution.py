# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Attribution surface — the shippable product on the validated US single-name signal.

Three views, all provenance-backed (every driver carries its verbatim quote + source):
  movers            — the biggest abnormal US-security movers in the window, each attributed
                      to its news drivers; headline coverage stat "we explain X% of movers".
  explain  TICKER   — a name's abnormal moves, each with ranked drivers (mechanism, direction,
                      cause, confidence, verbatim quote, source).
  drives   TICKER   — the name's driver PROFILE: which mechanisms / cause-entities recur, each
                      annotated with the mechanism's measured name-level reliability.

Signal doctrine (measured): US-listed single names only (foreign-local = timing noise; macro
ETF proxies = null). Abnormal return = residual on liquid Yahoo factor proxies.

Usage:
  uv run eventgraph/scripts/attribution.py movers --graph-dir /tmp/eg_live
  uv run eventgraph/scripts/attribution.py explain NVDA --graph-dir /tmp/eg_live
  uv run eventgraph/scripts/attribution.py drives  GS   --graph-dir /tmp/eg_live
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
MKT = ["ACWI", "TLT", "UUP", "GLD", "EEM"]
SKIP = {"SPY", "IEF", "HYG", "USO", "UUP", "GLD", "EEM", "FXI", "TLT", "DIA", "QQQ", "ACWI"}
# measured name-level directional reliability (signal_value.py); higher = news direction is trustworthy
S_DIR = {"monetary_policy": 0.43, "competition": 1.13, "rating_action": 1.02, "demand_change": 0.30,
         "earnings": 0.46, "regulation": 0.17, "geopolitics": 0.26, "other": 0.22, "guidance": 0.09,
         "rate_decision": 0.16, "supply_shock": 0.05, "default": 0.04, "mergers_acquisitions": -0.02,
         "data_surprise": -0.39, "contagion": 0.30}
CONF_RANK = {"strong": 3, "high": 3, "moderate": 2, "medium": 2, "weak": 1, "low": 1}


def yahoo(sym, cache, p1, p2):
    p = cache / f"{sym.replace('/', '_')}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25); txt = r.text if r.status_code == 200 else ""
        except Exception: txt = ""
        p.write_text(txt); time.sleep(0.1)
    out = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], cl):
            if c is not None: out[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception: pass
    return out


def logret(cl):
    ds = sorted(cl); return {ds[i]: math.log(cl[ds[i]] / cl[ds[i-1]]) for i in range(1, len(ds)) if cl[ds[i-1]] > 0 and cl[ds[i]] > 0}


def load(gd):
    lake = gd / "lake"
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j.get("kind") == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]              # US single names only
    docs = {json.loads(l)["doc_id"]: json.loads(l) for l in open(lake / "document.jsonl")}
    # edges keyed by (symbol, date), full provenance
    edges = collections.defaultdict(list)
    all_for_sym = collections.defaultdict(list)
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); e = j.get("effect_entity"); d = docs.get(j.get("doc_id"), {}).get("published_at")
        if e not in sym or not d or j.get("effect_dir") not in DIR: continue
        s = sym[e]; day = d[:10]
        rec = {"date": day, "mech": j.get("mechanism") or "other", "dir": j["effect_dir"], "ndir": DIR[j["effect_dir"]],
               "cause": (j.get("cause_entity") or "").split("__")[0].replace("_", " "), "quote": j.get("quote") or "",
               "conf": j.get("confidence") or "", "doc": j.get("doc_id"), "modality": j.get("modality") or "happened"}
        edges[(s, day)].append(rec); all_for_sym[s].append(rec)
    return sym, docs, edges, all_for_sym


def abn_series(px, Fmap):
    r = px; ds = [d for d in sorted(r) if d in Fmap]
    if len(ds) < 100: return {}
    F = np.array([Fmap[d] for d in ds]); colok = np.where(np.isfinite(F).mean(axis=0) >= 0.9)[0]
    rk = [k for k in range(len(ds)) if np.all(np.isfinite(F[k, colok]))]
    if len(rk) < 100: return {}
    dd = [ds[k] for k in rk]; y = np.array([r[d] for d in dd])
    X = np.column_stack([np.ones(len(dd)), F[np.ix_(rk, colok)]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ beta; rstd = resid.std()
    if rstd == 0: return {}
    return {dd[k]: (resid[k] / rstd, resid[k]) for k in range(len(dd))}   # (z, pct)


def build_factors(cache, p1, p2):
    mkt = {m: logret(yahoo(m, cache, p1, p2)) for m in MKT}
    fdays = sorted(set().union(*[set(mkt[m]) for m in MKT]))
    return {d: np.array([mkt[m].get(d, np.nan) for m in MKT]) for d in fdays}


def dedupe(recs):
    seen, out = set(), []
    for r in recs:
        k = (r["mech"], r["ndir"], (r["quote"] or "")[:60].lower())
        if k in seen: continue
        seen.add(k); out.append(r)
    return out


def rank_drivers(recs):
    return sorted(dedupe(recs), key=lambda r: (CONF_RANK.get(r["conf"], 1), abs(S_DIR.get(r["mech"], 0.1))), reverse=True)


def fmt_driver(r, move_dir=None):
    rel = S_DIR.get(r["mech"], 0.1)
    agree = "" if move_dir is None else ("  ✓agrees" if (r["ndir"] > 0) == (move_dir > 0) else "  ✗opposes")
    tag = "reliable" if rel >= 0.3 else ("weak" if rel < 0.15 else "med")
    q = (r["quote"][:120] + "…") if len(r["quote"]) > 120 else r["quote"]
    cause = f"{r['cause']} → " if r["cause"] else ""
    return (f"    • [{r['mech']}/{tag}] {cause}{r['dir'].upper()}  ({r['conf'] or 'conf?'}, {r['modality']}){agree}\n"
            f"        “{q}”")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["movers", "explain", "drives"])
    ap.add_argument("ticker", nargs="?", default="")
    ap.add_argument("--graph-dir", default="/tmp/eg_live")
    ap.add_argument("--price-from", type=int, default=2025); ap.add_argument("--price-to", type=int, default=2026)
    ap.add_argument("--z", type=float, default=1.5); ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(); gd = Path(a.graph_dir); cache = gd / "prices"; cache.mkdir(exist_ok=True)
    p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())
    sym, docs, edges, all_for_sym = load(gd)
    tmin = min((d[:10] for d in (x.get("published_at") for x in docs.values()) if d), default="")
    tmax = max((d[:10] for d in (x.get("published_at") for x in docs.values()) if d), default="")
    Fmap = build_factors(cache, p1, p2)

    if a.cmd == "drives":
        t = a.ticker.upper()
        recs = all_for_sym.get(t, [])
        if not recs: print(f"no edges resolve to {t} (US single names only)"); return
        print(f"=== what drives {t} ===  ({len(recs)} news drivers in corpus {tmin}..{tmax})")
        bym = collections.Counter(r["mech"] for r in recs)
        print("  by mechanism (annotated with measured name-level reliability):")
        for m, c in bym.most_common():
            rel = S_DIR.get(m, 0.1); tag = "RELIABLE" if rel >= 0.3 else ("weak" if rel < 0.15 else "med")
            net = sum(r["ndir"] for r in recs if r["mech"] == m)
            print(f"    {m:20} {c:3} edges  net {net:+d}   reliability {rel:+.2f} [{tag}]")
        print("  top cause entities:")
        for cse, c in collections.Counter(r["cause"] for r in recs if r["cause"]).most_common(8):
            print(f"    {cse:30} {c}")
        return

    # need abnormal returns -> fetch prices for the relevant US names
    names = [a.ticker.upper()] if a.cmd == "explain" else sorted({s for (s, _) in edges})
    if a.cmd == "explain" and names[0] not in all_for_sym:
        print(f"no edges resolve to {names[0]} (US single names only)"); return
    fetch = names if a.cmd == "explain" else sorted({s for (s, _) in edges}, key=lambda s: -len(all_for_sym[s]))[:250]
    zmap = {}
    for s in fetch:
        z = abn_series(logret(yahoo(s, cache, p1, p2)), Fmap)
        if z: zmap[s] = z

    if a.cmd == "explain":
        t = names[0]; z = zmap.get(t, {})
        days = sorted(d for d in z if tmin <= d <= tmax and abs(z[d][0]) >= 1.0)
        print(f"=== explain {t} ===  ({tmin}..{tmax}, moves |z|>=1.0)")
        if not days: print("  no notable abnormal days with data in window"); return
        for d in sorted(days, key=lambda d: -abs(z[d][0])):
            zz, pct = z[d]
            drv = edges.get((t, d), []) + edges.get((t, (dt.date.fromisoformat(d) - dt.timedelta(days=1)).isoformat()), [])
            print(f"\n  {d}   abnormal {pct*100:+.2f}%  (z {zz:+.1f})   {'← ' + str(len(drv)) + ' driver(s)' if drv else '← no news in graph'}")
            for r in rank_drivers(drv)[:4]:
                print(fmt_driver(r, move_dir=(1 if zz > 0 else -1)))
        return

    # movers
    rows = []
    for s, z in zmap.items():
        for d in z:
            if tmin <= d <= tmax and abs(z[d][0]) >= a.z:
                drv = edges.get((s, d), []) + edges.get((s, (dt.date.fromisoformat(d) - dt.timedelta(days=1)).isoformat()), [])
                rows.append((abs(z[d][0]), s, d, z[d][0], z[d][1], drv))
    rows.sort(reverse=True)
    covered = sum(1 for r in rows if r[5]);
    print(f"=== what's moving & why ===  US single names, {tmin}..{tmax}  |  {len(rows)} moves (|z|>{a.z})")
    print(f"    we attribute {covered}/{len(rows)} ({covered/max(len(rows),1):.0%}) to a news driver in the graph\n")
    for _, s, d, zz, pct, drv in rows[:a.top]:
        best = rank_drivers(drv)[0] if drv else None
        head = f"{s:6} {d}  {pct*100:+6.2f}% (z{zz:+.1f})"
        if best:
            agree = "✓" if (best["ndir"] > 0) == (zz > 0) else "✗"
            print(f"  {head}  {agree} [{best['mech']}] {best['cause']+' → ' if best['cause'] else ''}{best['dir']}  “{best['quote'][:70]}”")
        else:
            print(f"  {head}  — no graph driver")


if __name__ == "__main__":
    main()
