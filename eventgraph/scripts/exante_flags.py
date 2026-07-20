# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
EX-ANTE persistence flags (blind-spot #2 closed) — the product demo: "monitoring demo:
calibrated structure, untested activation layer."

Extends persistence_signal.build() (structure tiers + EX-POST confirmation) with three
EX-ANTE evidences, design hardened by external review:
  E1 CALENDAR (headline; genuinely ex-ante): cause slug -> keyword rules -> macro series;
     upcoming releases from LIVE BLS/BEA ICS + FOMC dates (Kalshi KXFED close dates).
     Slug->series mapping logged in provenance.
  E3 IMPLIED MOVE (second): live option chains (crumb-authed Yahoo); ATM straddle/spot at
     the expiry nearest +14d; reported as CROSS-SECTIONAL RANK of implied/realized across
     covered names (not raw threshold); MANDATORY earnings screen (earnings before expiry
     -> labeled earnings-contaminated). Ex-ante by construction; UNVALIDATED, accumulating.
  E2 DRIVER STATE (context metadata only, per review -- partially mechanical): cause ->
     ETF proxy; driver 20d move z; overlap warning when proxy is/correlates with a
     neutralization ETF.

NO calibrated percentages on flags: ordinal tier labels only; the 2010-12 MONTHLY
calibration appears once in a footnote, explicitly cross-horizon/cross-regime.

Usage: uv run scripts/exante_flags.py --graph-dir ../data/eg_runs/eg_live2 --tau 0.25
"""
import argparse, json, collections, math, re, sys, datetime as dt
from pathlib import Path
import numpy as np
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import persistence_signal as ps
from resolve_series import RULES
from formal_calendar import parse_ics, ics_to_utc, UA

NEUTRALIZERS = set(ps.PROXIES)
ICS_FEEDS = [("bls", "https://www.bls.gov/schedule/news_release/bls.ics"),
             ("bea", "https://www.bea.gov/news/schedule/ics")]
ORDINAL = {1: "TIER-C structure only", 2: "TIER-B structure+1-leg confirmed", 3: "TIER-A structure+both-confirmed"}
FOOTNOTE = ("[calibration footnote: on 2010-12 US data, MONTHLY horizon, elevated-corr pairs persisted "
            "20% unlinked / 41% dormant / 63% one-leg / 69% both-legs confirmed. Different horizon and "
            "regime; NOT this flag's probability. Ex-ante evidences E1/E3 are unvalidated, accumulating.]")


def upcoming_calendar(days=14):
    """series -> [dates] within the window, from live agency ICS + Kalshi FOMC closes."""
    today = dt.date.today(); lim = today + dt.timedelta(days=days)
    out = collections.defaultdict(list)
    rules = [(n, re.compile(rx, re.I), canon) for n, rx, canon in RULES]
    for agency, url in ICS_FEEDS:
        try:
            text = httpx.get(url, headers=UA, timeout=30, follow_redirects=True).text
        except Exception:
            continue
        for ev in parse_ics(text):
            t = ics_to_utc(ev.get("DTSTART", ""), ev.get("_TZID"))
            if not t: continue
            d = dt.date.fromisoformat(t[:10])
            if not (today <= d <= lim): continue
            summ = ev.get("SUMMARY", "").replace("\\,", ",")
            hit = next((c for n, rx, c in rules if rx.search(summ)), None)
            if hit: out[hit].append((str(d), summ[:48]))
    # FOMC from the archived Kalshi KXFED close dates
    mq = Path(__file__).resolve().parent.parent.parent / "data/eg_runs/formal/eg_ip/lake/market_quote.jsonl"
    if mq.exists():
        for l in open(mq):
            j = json.loads(l)
            if str(j.get("market_ref", "")).startswith("KXFED"):
                d = (j.get("close_time") or "")[:10]
                try:
                    dd = dt.date.fromisoformat(d)
                    if today <= dd <= lim and (d, "FOMC decision (Kalshi close)") not in out["US-FOMC"]:
                        out["US-FOMC"].append((d, "FOMC decision (Kalshi close)"))
                except ValueError: pass
    return {k: sorted(set(v)) for k, v in out.items()}


def slug_series(slug):
    text = slug.replace("_", " ")
    for n, rx, canon in [(n, re.compile(rx, re.I), c) for n, rx, c in RULES]:
        if rx.search(text): return canon
    return None


def crumb_session():
    s = httpx.Client(headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                     follow_redirects=True, timeout=25)
    try:
        s.get("https://fc.yahoo.com")
        crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb").text.strip()
        return s, crumb if crumb and "<" not in crumb else None
    except Exception:
        return s, None


def implied_move(sess, crumb, symn, horizon_days=14):
    """(implied_move, days_to_expiry, earnings_before_expiry, spot) from the live chain."""
    try:
        d = sess.get(f"https://query2.finance.yahoo.com/v7/finance/options/{symn}",
                     params={"crumb": crumb}).json()
        r = d["optionChain"]["result"][0]
        spot = r["quote"].get("regularMarketPrice"); exps = r.get("expirationDates", [])
        if not spot or not exps: return None
        target = dt.datetime.now(dt.UTC).timestamp() + horizon_days * 86400
        exp = min(exps, key=lambda e: abs(e - target))
        d2 = sess.get(f"https://query2.finance.yahoo.com/v7/finance/options/{symn}",
                      params={"crumb": crumb, "date": exp}).json()
        r2 = d2["optionChain"]["result"][0]; opt = r2["options"][0]
        def atm_mid(side):
            cands = [(abs(o["strike"] - spot), o) for o in opt.get(side, []) if o.get("bid") or o.get("ask")]
            if not cands: return None
            o = min(cands)[1]; b, a2 = o.get("bid") or 0, o.get("ask") or 0
            return (b + a2) / 2 if (b and a2) else (o.get("lastPrice") or None)
        c, p = atm_mid("calls"), atm_mid("puts")
        if c is None or p is None: return None
        dte = max(1, int((exp - dt.datetime.now(dt.UTC).timestamp()) / 86400))
        ets = r["quote"].get("earningsTimestamp")
        earn = bool(ets and ets < exp)
        return {"implied_move": (c + p) / spot, "dte": dte, "earnings_before_expiry": earn, "spot": spot}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="../data/eg_runs/eg_live2")
    ap.add_argument("--tau", type=float, default=0.25); ap.add_argument("--horizon", type=int, default=14)
    a = ap.parse_args(); gd = Path(a.graph_dir)
    rows, ar, trail_days, _ = ps.build(gd)
    flags = [r for r in rows if r["trailing_corr"] >= a.tau]
    flags.sort(key=lambda r: (-r["tier"], -r["trailing_corr"]))
    print(f"{len(flags)} structure flags (trailing >= {a.tau}); attaching ex-ante evidence ...")

    cal = upcoming_calendar(a.horizon)
    print(f"E1 calendar: upcoming mapped events: { {k: [d for d,_ in v] for k,v in cal.items()} }")

    # E2 metadata: cause -> etf proxy map from entity_symbol
    proxy = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "etf_proxy": proxy[j["entity_id"].split("__")[0]] = j["symbol"]

    # E3: fetch implied moves once per unique leg; realized move for the same horizon
    sess, crumb = crumb_session()
    legs = sorted({s for f in flags for s in f["pair"]})
    imp = {}
    if crumb:
        for s_ in legs:
            r = implied_move(sess, crumb, s_, a.horizon)
            if r:
                ret = ar.get(s_, {}); vals = list(ret.values())[-30:]
                if len(vals) >= 15:
                    rv = float(np.std(vals)) * math.sqrt(r["dte"])
                    r["realized_move_same_h"] = rv
                    r["ratio"] = r["implied_move"] / rv if rv > 0 else None
                imp[s_] = r
    ranked = sorted([s_ for s_ in imp if imp[s_].get("ratio")], key=lambda s_: -imp[s_]["ratio"])
    xrank = {s_: (i + 1, len(ranked)) for i, s_ in enumerate(ranked)}
    print(f"E3 implied: chains fetched for {len(imp)}/{len(legs)} legs (crumb {'ok' if crumb else 'FAILED'})")

    out = gd / "exante_flags.jsonl"; kept = []
    for f in flags:
        e1 = []
        for cslug in f["causes"]:
            ser = slug_series(cslug)
            if ser and ser in cal:
                e1.append({"cause": cslug, "series": ser, "events": cal[ser]})
        e3 = {}
        for s_ in f["pair"]:
            if s_ in imp:
                d_ = imp[s_]
                e3[s_] = {"implied_move": round(d_["implied_move"], 4), "dte": d_["dte"],
                          "xs_rank": f"{xrank[s_][0]}/{xrank[s_][1]}" if s_ in xrank else None,
                          "earnings_before_expiry": d_["earnings_before_expiry"]}
        e2 = {}
        for cslug in f["causes"]:
            if cslug in proxy:
                pxs = proxy[cslug]
                e2[cslug] = {"proxy": pxs, "overlap_warning": pxs in NEUTRALIZERS}
        f2 = dict(f); f2.pop("persist_calib", None)   # no rate numbers on flags
        f2["label"] = ORDINAL[f["tier"]]
        f2["exante"] = {"E1_calendar": e1, "E3_implied": e3, "E2_driver_context": e2}
        kept.append(f2)
    with open(out, "w") as fo:
        for f2 in kept: fo.write(json.dumps(f2) + "\n")

    print(f"\n=== EX-ANTE PERSISTENCE FLAGS (as of {trail_days[-1]}) ===")
    print(f"  {'pair':16} {'corr':>6} {'label':36} {'E1(next event)':20} {'E3 legs':14} causes")
    for f2 in kept[:20]:
        e1s = f2["exante"]["E1_calendar"]
        e1txt = e1s[0]["events"][0][0] + " " + e1s[0]["series"] if e1s else "-"
        e3n = sum(1 for v in f2["exante"]["E3_implied"].values() if v.get("xs_rank"))
        earn = any(v.get("earnings_before_expiry") for v in f2["exante"]["E3_implied"].values())
        print(f"  {'~'.join(f2['pair']):16} {f2['trailing_corr']:+6.2f} {f2['label']:36} {e1txt:20} "
              f"{e3n}{'(EARN)' if earn else '':8} {', '.join(f2['causes'][:2])}")
    print(f"\n  {len(kept)} flags -> {out}\n\n  {FOOTNOTE}")


if __name__ == "__main__":
    main()
