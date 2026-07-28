# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi"]
# ///
"""
DO IMF PROGRAMME EVENTS PREDICT FX STEP MOVES IN FRONTIER MARKETS?
A jump-hazard event-window profile — and the test of whether the "under-exploited
frontier calendar" thesis survives contact with data.

WHY A HAZARD MODEL AND NOT AN EVENT STUDY. The venues whose calendars are least
covered (Nigeria, Ghana, Egypt, Zambia, Pakistan) run administered or heavily
managed exchange rates. Measured over 5y: USDNGN has zero change on 9.6% of
sessions, USDGHS on 14.7%, USDKES on 15.0% — and then jumps 33%, 28%, 60% in a
single day. That is not a diffusion, so an abnormal-return study divided by
trailing vol is measuring a quantity that does not exist. std()>0 is not a
liquidity guard (the 267-sigma control in the stale-price work made that point
expensively). The right object is the HAZARD of a step move.

THE DESIGN AND WHY IT IS FALSIFIABLE. A jump is defined as the top 1% of a
currency's OWN |daily return| distribution. That makes the unconditional hazard
exactly 1% per session BY CONSTRUCTION, per currency, so any event-window offset
with hazard > 1% is excess by definition and no vol model is needed.

Events are IMF lending-arrangement approvals (authoritative, exactly dated, from
the Fund's own TSV export). Each is mapped to the FIRST TRADING SESSION ON OR
AFTER the board date, because board decisions land in the US afternoon and an
African or South Asian currency cannot react until its next local session — the
same alignment that moved the 8-K result 2.2x -> 3.2x.

READ THE ANSWER LIKE THIS. IMF approval usually REQUIRES exchange-rate action as a
prior action, so mass BEFORE the date is the mechanical expectation and is not
evidence of anything tradeable. The informative cell is AFTER:

    mass before  -> the move is a precondition, market had it -> efficient, no edge
    mass after   -> the market absorbed it slowly -> the under-exploitation claim survives
    no mass      -> IMF programme dates do not drive the FX step -> thesis dies here

NULL. Circular-shifted event dates within each currency's own series (the
technique that produced honest chance bands in the pair-regime work), so the null
preserves each currency's jump clustering instead of assuming independence.

Usage:
    uv run scripts/em_jump_hazard.py --graph-dir data/eg_runs/em_cal
    uv run scripts/em_jump_hazard.py --graph-dir data/eg_runs/em_cal --top-pct 2 --window 15
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import statistics
import time
from pathlib import Path

from curl_cffi import requests as cf

ARR1 = "https://www.imf.org/external/np/fin/tad/extarr1.aspx"
ARR2 = ("https://www.imf.org/external/np/fin/tad/extarr2.aspx"
        "?date1key=2026-06-30&memberkey1={key}&tsvflag=Y")
CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range=10y&interval=1d"

# country -> the USD cross that carries its sovereign stress.
#
# WHY THE WHOLE BORROWER SET AND NOT JUST THE HEADLINE FRONTIER NAMES. n was the
# binding constraint at 34 events. The obvious fix was IMF programme REVIEWS
# (~2/year/programme vs one approval), but MONA is decommissioned, imf.org/en/* is
# edge-blocked and no free syndicator carries board dates at scale. Widening the
# COUNTRY set is strictly better anyway: reviews inside one programme are highly
# correlated with each other and with the approval, whereas separate countries are
# closer to independent clusters -- which is what the permutation inference needs.
UNIVERSE = {
    "Egypt": "EGP", "Ghana": "GHS", "Nigeria": "NGN", "Zambia": "ZMW",
    "Pakistan": "PKR", "Kenya": "KES", "Sri Lanka": "LKR", "Argentina": "ARS",
    "Angola": "AOA", "Ethiopia": "ETB", "Malawi": "MWK", "Mozambique": "MZN",
    "Albania": "ALL", "Armenia": "AMD", "Bangladesh": "BDT", "Barbados": "BBD",
    "Bolivia": "BOB", "Colombia": "COP", "Costa Rica": "CRC", "Georgia": "GEL",
    "Gambia": "GMD", "Guinea": "GNF", "Honduras": "HNL", "Jamaica": "JMD",
    "Jordan": "JOD", "Kyrgyz": "KGS", "Liberia": "LRD", "Madagascar": "MGA",
    "Maldives": "MVR", "Mauritania": "MRU", "Mexico": "MXN", "Moldova": "MDL",
    "Mongolia": "MNT", "Morocco": "MAD", "Nepal": "NPR", "North Macedonia": "MKD",
    "Papua New Guinea": "PGK", "Paraguay": "PYG", "Peru": "PEN", "Rwanda": "RWF",
    "Serbia": "RSD", "Seychelles": "SCR", "Sierra Leone": "SLL", "Somalia": "SOS",
    "Sudan": "SDG", "Suriname": "SRD", "Tanzania": "TZS", "Tunisia": "TND",
    "Uganda": "UGX", "Ukraine": "UAH", "Uzbekistan": "UZS", "Azerbaijan": "AZN",
    "Kazakhstan": "KZT", "Tajikistan": "TJS", "Vietnam": "VND", "Cambodia": "KHR",
    "Myanmar": "MMK", "Botswana": "BWP", "Mauritius": "MUR", "Namibia": "NAD",
    "Eswatini": "SZL", "Fiji": "FJD", "Guyana": "GYD", "Trinidad": "TTD",
    "Dominican Republic": "DOP", "Guatemala": "GTQ", "Belarus": "BYN",
    "Iraq": "IQD", "Haiti": "HTG", "Lesotho": "LSL", "Turkiye": "TRY",
}
# Excluded by construction, not by a data filter: currencies whose moves are
# somebody else's monetary policy. XOF/XAF/KMF/CVE are euro pegs (USDXOF vol IS
# EURUSD vol), XCD and the dollarised/hard-pegged names cannot express local
# sovereign stress at all. Including them would add events and pure noise.
PEGGED = {
    # euro bloc / euro-linked
    "XOF", "XAF", "KMF", "CVE", "MKD", "BAM",
    # rand bloc -- LSL/SZL/NAD are 1:1 with ZAR, so their "jumps" are South
    # African monetary policy, not local sovereign stress
    "LSL", "SZL", "NAD",
    # rupee-linked
    "NPR", "BTN",
    # dollar pegs / dollarised
    "XCD", "USD", "PAB", "BSD", "BZD", "DJF", "BBD", "MVR", "BOB", "JOD",
}

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def cached(path: Path, fetch, ttl_days: int = 7):
    """Snapshot-and-reuse: these are slow, rate-limited endpoints and the whole
    project's doctrine is to keep the raw artifact rather than refetch."""
    if path.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
        if age.days < ttl_days:
            return path.read_text()
    txt = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    return txt


def member_keys(cache: Path) -> dict[str, str]:
    html = cached(cache / "imf_extarr1.html",
                  lambda: cf.get(ARR1, impersonate="chrome", timeout=40).text)
    opts = re.findall(r'<option[^>]*value="(\d+)"[^>]*>([^<]+)</option>', html)
    out = {}
    for key, name in opts:
        for want in UNIVERSE:
            if name.strip().lower().startswith(want.lower()):
                out[want] = key
    return out


def arrangements(cache: Path, country: str, key: str) -> list[dict]:
    """IMF lending-arrangement approvals from the Fund's TSV export."""
    txt = cached(cache / f"imf_arr_{key}.tsv",
                 lambda: cf.get(ARR2.format(key=key), impersonate="chrome", timeout=40).text)
    out = []
    for line in txt.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 4:
            continue
        m = re.match(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4})$", parts[1])
        if not m or not parts[0]:
            continue
        out.append({"country": country, "facility": parts[0],
                    "date": dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))})
    return out


def fx_series(cache: Path, sym: str) -> tuple[list[dt.date], list[float]]:
    def go():
        r = cf.get(CHART.format(sym=sym), impersonate="chrome", timeout=40)
        return r.text
    txt = cached(cache / f"fx_{sym.replace('=', '_')}.json", go)
    try:
        j = json.loads(txt)
        res = (j.get("chart") or {}).get("result")
        if not res:
            return [], []
        ts = res[0]["timestamp"]
        cl = res[0]["indicators"]["quote"][0]["close"]
    except Exception:
        return [], []
    days, px = [], []
    for t, c in zip(ts, cl):
        if c is None:
            continue
        days.append(dt.datetime.utcfromtimestamp(t).date())
        px.append(c)

    # DATA-INTEGRITY GUARD. Yahoo's frontier FX history carries scale glitches and
    # placeholder segments that masquerade as the largest moves in the sample:
    # USDGHS 5.67 -> 573.00, USDMWK 1.00 -> 710.00, USDNGN 3.59 -> 360.50. Those
    # land straight in a top-1% jump set and would manufacture the very result
    # this script is testing for. No floating currency moves 3x in a session --
    # Argentina's genuine +118% (Dec 2023) and Egypt's +71.7% float both survive
    # this threshold -- so treat any larger ratio as a broken series and keep only
    # the segment AFTER the last such break, since the pre-break level is wrong.
    last_break = 0
    for i in range(1, len(px)):
        if px[i - 1] <= 0 or px[i] / px[i - 1] > 3.0 or px[i] / px[i - 1] < 1 / 3.0:
            last_break = i
    if last_break:
        days, px = days[last_break:], px[last_break:]
    return days, px


def jumps(px: list[float], top_pct: float) -> tuple[list[int], float]:
    """Index positions of the largest |returns|. The unconditional hazard is
    top_pct BY CONSTRUCTION, which is what makes the excess readable without a
    volatility model."""
    rets = [abs((px[i] - px[i - 1]) / px[i - 1]) for i in range(1, len(px))]
    if not rets:
        return [], 0.0
    k = max(1, int(len(rets) * top_pct / 100))
    thresh = sorted(rets, reverse=True)[k - 1]
    idx = [i + 1 for i, r in enumerate(rets) if r >= thresh]
    return idx, thresh


def session_on_or_after(days: list[dt.date], d: dt.date) -> int | None:
    """First tradeable session at or after the board date. A board decision lands
    in the US afternoon; the local currency reacts on its next session."""
    lo, hi = 0, len(days)
    while lo < hi:
        mid = (lo + hi) // 2
        if days[mid] < d:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(days) else None


def profile(anchors: list[int], jump_set: set[int], n: int, w: int) -> dict[int, int]:
    """Count jumps at each offset from the event sessions."""
    prof = {k: 0 for k in range(-w, w + 1)}
    for a in anchors:
        for k in range(-w, w + 1):
            j = a + k
            if 0 <= j < n and j in jump_set:
                prof[k] += 1
    return prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--top-pct", type=float, default=1.0, help="jump = top X%% of |returns|")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--since", type=int, default=2016, help="events from this year on")
    ap.add_argument("--placebo", type=int, default=2000)
    ap.add_argument("--min-move", type=float, default=0.005,
                    help="skip currencies whose top-1%% move is below this (de-facto pegs)")
    ap.add_argument("--local", type=int, default=90,
                    help="local-control neighbourhood in sessions either side")
    a = ap.parse_args()
    cache = a.graph_dir / "raw" / "hazard"
    rng = random.Random(20260726)

    keys = member_keys(cache)
    print(f"IMF member keys resolved: {len(keys)}/{len(UNIVERSE)}\n")

    rows, total_anchor, total_jump, total_sessions = [], 0, 0, 0
    price_by_country, all_events = [], []
    agg = {k: 0 for k in range(-a.window, a.window + 1)}
    per_country_anchors = []

    for country, ccy in UNIVERSE.items():
        if ccy in PEGGED:
            continue
        sym = f"USD{ccy}=X"
        key = keys.get(country)
        if not key:
            print(f"  {country:12} no IMF member key — skipped")
            continue
        arr = [e for e in arrangements(cache, country, key) if e["date"].year >= a.since]
        days, px = fx_series(cache, sym)
        time.sleep(0.6)
        if len(px) < 250:
            print(f"  {country:12} {sym:10} no usable price history ({len(px)} bars) — skipped")
            continue
        jidx, thresh = jumps(px, a.top_pct)
        # De-facto peg detector. An explicit PEGGED list catches the euro/dollar
        # blocs by name, but plenty of currencies are crawling-peg in practice. If
        # a currency's 99th-percentile daily move is under 0.5% it has no step-move
        # behaviour to detect and its "top 1%" is quantisation noise.
        if thresh < a.min_move:
            print(f"  {country:12} {sym:10} de-facto peg (top-1% move only "
                  f"{thresh:.2%}) — skipped")
            continue
        jset = set(jidx)
        anchors = []
        for e in arr:
            if e["date"] < days[0] or e["date"] > days[-1]:
                continue
            i = session_on_or_after(days, e["date"])
            if i is not None:
                anchors.append(i)
        if not anchors:
            print(f"  {country:12} {sym:10} {len(arr):>2} arrangements, none inside price window")
            continue
        prof = profile(anchors, jset, len(px), a.window)
        for k, v in prof.items():
            agg[k] += v
        rows.append((country, sym, len(anchors), len(jidx), len(px), thresh, prof))
        per_country_anchors.append((anchors, jset, len(px)))
        price_by_country.append(px)
        for e in arr:
            if days[0] <= e['date'] <= days[-1]:
                i = session_on_or_after(days, e['date'])
                if i is not None:
                    all_events.append((e['date'], i, jset, len(px)))
        total_anchor += len(anchors)
        total_jump += len(jidx)
        total_sessions += len(px)
        print(f"  {country:12} {sym:10} {len(anchors):>2} events, {len(jidx):>3} jumps "
              f"(|r| >= {thresh:5.1%}) over {len(px)} sessions")

    if not rows:
        raise SystemExit("\nnothing testable — no country had both IMF events and price history")

    w = a.window
    span = 2 * w + 1
    exposure = total_anchor * span
    base = a.top_pct / 100
    print(f"\n=== JUMP-HAZARD PROFILE  ({total_anchor} events, {total_jump} jumps, "
          f"{total_sessions} sessions; unconditional hazard = {base:.1%} by construction)")
    print(f"  {'offset':>7} {'jumps':>6} {'hazard':>8}  {'lift':>6}")
    for k in range(-w, w + 1):
        h = agg[k] / total_anchor if total_anchor else 0
        mark = " <-- event session" if k == 0 else ""
        print(f"  {k:>7} {agg[k]:>6} {h:>7.1%} {h / base:>6.2f}x{mark}")

    pre = sum(agg[k] for k in range(-w, 0))
    post = sum(agg[k] for k in range(1, w + 1))
    at = agg[0]
    print(f"\n  pre  (-{w}..-1): {pre:>4} jumps  hazard {pre / (total_anchor * w):.1%}  "
          f"lift {pre / (total_anchor * w) / base:.2f}x")
    print(f"  at   (   0   ): {at:>4} jumps  hazard {at / total_anchor:.1%}  "
          f"lift {at / total_anchor / base:.2f}x")
    print(f"  post (+1..+{w}): {post:>4} jumps  hazard {post / (total_anchor * w):.1%}  "
          f"lift {post / (total_anchor * w) / base:.2f}x")

    # --- circular-shift placebo -------------------------------------------------
    # Shift every anchor by the same random lag within each currency's own series,
    # so the null keeps each currency's jump CLUSTERING (jumps arrive in bursts)
    # instead of assuming independence, which would understate the chance bands.
    obs = {"pre": pre, "at": at, "post": post, "total": pre + at + post}
    null = {k: [] for k in obs}
    for _ in range(a.placebo):
        p = q = r = 0
        for anchors, jset, n in per_country_anchors:
            shift = rng.randrange(n)
            for anc in anchors:
                b = (anc + shift) % n
                for k in range(-w, w + 1):
                    j = b + k
                    if 0 <= j < n and j in jset:
                        if k < 0:
                            p += 1
                        elif k == 0:
                            q += 1
                        else:
                            r += 1
        null["pre"].append(p); null["at"].append(q); null["post"].append(r)
        null["total"].append(p + q + r)

    print(f"\n=== CIRCULAR-SHIFT PLACEBO ({a.placebo} draws)")
    print(f"  {'cell':>6} {'observed':>9} {'null mean':>10} {'null p95':>9} {'p-value':>9}")
    for cell in ("pre", "at", "post", "total"):
        d = sorted(null[cell])
        mean = statistics.fmean(d)
        p95 = d[int(0.95 * len(d))]
        pval = (sum(1 for x in d if x >= obs[cell]) + 1) / (len(d) + 1)
        flag = "  *" if pval < 0.05 else ""
        print(f"  {cell:>6} {obs[cell]:>9} {mean:>10.1f} {p95:>9} {pval:>9.3f}{flag}")

    # --- LOCAL CONTROL: the test that actually decides this ---------------------
    # The global placebo above answers "do jumps cluster near IMF dates more than
    # near random dates" — and that is nearly guaranteed to say yes for an
    # uninformative reason: IMF arrangements are approved DURING currency crises,
    # and jumps happen during currency crises. Both are caused by the crisis, so a
    # global null tests a straw man.
    #
    # The honest null holds the crisis period FIXED and asks a harder question:
    # given that a country was in the kind of period where the Fund shows up, does
    # the jump land near the board date SPECIFICALLY? So the baseline is the
    # event's own +/-`local` day neighbourhood excluding the event window, and the
    # permutation relocates each event only WITHIN that neighbourhood.
    L = a.local
    in_j = in_n = out_j = out_n = 0
    for anchors, jset, n in per_country_anchors:
        for anc in anchors:
            lo, hi = max(0, anc - L), min(n, anc + L + 1)
            for j in range(lo, hi):
                inside = abs(j - anc) <= w
                if inside:
                    in_n += 1
                    in_j += j in jset
                else:
                    out_n += 1
                    out_j += j in jset
    h_in = in_j / in_n if in_n else 0
    h_out = out_j / out_n if out_n else 0
    print(f"\n=== LOCAL CONTROL (baseline = each event's own +/-{L}d neighbourhood, "
          f"event window excluded)")
    print(f"  in-window   {in_j:>4} jumps / {in_n:>5} sessions = {h_in:.1%}")
    print(f"  neighbourhood {out_j:>4} jumps / {out_n:>5} sessions = {h_out:.1%}")
    print(f"  lift vs local baseline: {(h_in / h_out) if h_out else float('nan'):.2f}x "
          f"(vs {h_in / base:.2f}x against the global 1% base)")

    hits = 0
    for _ in range(a.placebo):
        c = 0
        for anchors, jset, n in per_country_anchors:
            for anc in anchors:
                lo, hi = max(0, anc - L), min(n, anc + L + 1)
                fake = rng.randrange(lo, hi)
                for k in range(-w, w + 1):
                    j = fake + k
                    if lo <= j < hi and j in jset:
                        c += 1
        hits += c >= (in_j)
    p_local = (hits + 1) / (a.placebo + 1)
    print(f"  local permutation p-value: {p_local:.3f}"
          f"{'  *' if p_local < 0.05 else '   (not significant)'}")

    # --- DIRECTION: "jumps cluster" is far weaker than "DEVALUATIONS cluster" ----
    # |return| is direction-blind, and a direction-blind result is untradeable even
    # if real. Depreciation = USD/local rises.
    dep = app = 0
    for (anchors, jset, n), px_c in zip(per_country_anchors, price_by_country):
        for anc in anchors:
            for k in range(-w, w + 1):
                j = anc + k
                if 0 <= j < n and j in jset and j > 0:
                    if px_c[j] > px_c[j - 1]:
                        dep += 1
                    else:
                        app += 1
    tot = dep + app
    print(f"\n=== DIRECTION of in-window jumps")
    print(f"  depreciation {dep:>3}   appreciation {app:>3}   "
          f"= {dep / tot:.0%} depreciation" if tot else "  none")

    # --- STABILITY: split the events in half by date. The repo's graveyard is full
    # of results that were one regime wearing a trend coat.
    print("\n=== TEMPORAL STABILITY (events split at the median date)")
    dated = sorted(all_events, key=lambda e: e[0])
    if len(dated) >= 8:
        mid = len(dated) // 2
        for label, half in (("early", dated[:mid]), ("late", dated[mid:])):
            hj = hn = 0
            for _, anchors_one, jset, n in half:
                for k in range(-w, w + 1):
                    j = anchors_one + k
                    if 0 <= j < n:
                        hn += 1
                        hj += j in jset
            print(f"  {label:6} {len(half):>2} events ({dated[0 if label=='early' else mid][0]} .. "
                  f"{dated[mid-1 if label=='early' else -1][0]})  "
                  f"{hj:>2} jumps / {hn:>3} sessions = {hj/hn if hn else 0:.1%}  "
                  f"lift {(hj/hn)/base if hn else 0:.2f}x")

    print("\n=== PER-COUNTRY in-window lift (concentration check)")
    print(f"  {'country':18} {'events':>6} {'in-win':>6} {'expected':>8} {'lift':>6}")
    tot_l = []
    for (country, sym, ne, nj, ns, th, prof) in sorted(rows, key=lambda r: -sum(r[6].values())):
        inw = sum(prof[k] for k in range(-w, w + 1))
        exp = ne * (2 * w + 1) * base
        lift = inw / exp if exp else 0
        tot_l.append((country, ne, inw, lift))
        print(f"  {country:18} {ne:>6} {inw:>6} {exp:>8.1f} {lift:>5.2f}x")
    nz = sum(1 for _, _, i, _ in tot_l if i > 0)
    print(f"  -> {nz}/{len(tot_l)} countries contributed at least one in-window jump")

    out = a.graph_dir / "lake" / "jump_hazard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "params": {"top_pct": a.top_pct, "window": w, "since": a.since, "placebo": a.placebo, "local": a.local},
        "events": total_anchor, "jumps": total_jump, "sessions": total_sessions,
        "profile": {str(k): v for k, v in agg.items()},
        "cells": obs,
        "placebo_p": {c: (sum(1 for x in null[c] if x >= obs[c]) + 1) / (a.placebo + 1)
                      for c in obs},
        "per_country": [{"country": c, "sym": s, "events": ne, "jumps": nj,
                         "sessions": ns, "threshold": th}
                        for c, s, ne, nj, ns, th, _ in rows],
    }, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
