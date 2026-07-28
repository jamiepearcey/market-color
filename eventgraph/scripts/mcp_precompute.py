# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Precompute the cache the MCP server serves from.

The residual pass (250-day rolling betas over 547 names) takes ~40s. An MCP
server cannot pay that on every start, so everything the tools need is computed
once here and written to a single compact JSON:

    days           trading-day axis
    sectors        ticker -> {division, major_group, industry}   (SIC, from EDGAR)
    sigma          ticker -> {day_index: standardised idiosyncratic move}
    decomp         ticker -> {day_index: [total, market, sector, idio]}
    events         "TICKER|DATE" -> {types, docs}
    docs           doc_id -> {date, headline, chunks}
    neighbours     ticker -> [[other, corr, same_industry], ...]
    class_counts   ticker -> {event_class: n}    (the "news weight" breakdown)

Everything is derived with the shared liquidity guard and the shared SIC map, so
the server cannot drift from the research scripts the way the per-page copies did.

Usage:
    uv run scripts/mcp_precompute.py
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liquidity import tradeable, usable_window  # noqa: E402
from sic import division, subsector, industry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
OUT = G / "mcp_cache.json"
BETA_WIN, TRAIL = 250, 60
TOP_NEIGHBOURS = 12
MIN_OVERLAP = 400

# MARKETS. The two graphs resolve entities and sectors by DIFFERENT means, and the
# difference matters enough to keep explicit rather than hide behind a flag:
#
#   us     entities carry resolution_status == resolved_security from the extractor;
#          sectors are SIC, ASSIGNED BY THE SEC per filer — authoritative.
#   india  the extractor resolved nothing (0 resolved_security; the legacy
#          nifty50_resolved.json is partly hallucinated — it maps SBI to SBILIFE and
#          Tata Power to POWERGRID). Resolution comes from india_ticker_map.json,
#          built by exact-normalised or full-token-subset matching against the real
#          NSE listing, which is conservative but verifiable. Sectors, however, are
#          TEXT-INFERRED (sector_text/name_text) — the same class of inference that
#          produced a spurious sector effect in the US work. India sector-level
#          figures are therefore weaker evidence than the US ones and are labelled
#          as such throughout.
MARKETS = {
    "us": {"graph": "eg100k_graph", "label": "United States (Bloomberg)",
           "sector_source": "SIC — assigned by the SEC per filer",
           "sector_confidence": "authoritative"},
    # the live capture — same extraction shape as eg100k, US names, SIC sectors
    "live": {"graph": "eg_live2", "label": "United States (live capture, 2026)",
             "sector_source": "SIC — assigned by the SEC per filer",
             "sector_confidence": "authoritative"},
    "india": {"graph": "india2021", "label": "India (NSE)",
              "sector_source": "text-inferred from article and company names",
              "sector_confidence": "weak — inferred, not assigned"},
}


def load_entities(g: Path, market: str) -> tuple[dict, dict]:
    ent, name_of = {}, {}
    if market == "india":
        m = json.loads((g / "classification" / "india_ticker_map.json").read_text())
        for eid, v in m.items():
            tk = v["ticker"].upper()
            ent[eid] = tk
            name_of.setdefault(tk, v.get("company_name") or tk)
        return ent, name_of
    for l in open(g / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = j.get("resolved_ticker")
        if j.get("resolution_status") == "resolved_security" and tk:
            ent[j["entity_id"]] = tk.upper()
            name_of.setdefault(tk.upper(), j.get("name") or tk)
    return ent, name_of


def load_sectors(g: Path, market: str, ent: dict) -> dict:
    """ticker -> {division, major_group, industry}. India has no assigned taxonomy,
    so the single text-inferred label is repeated at every level rather than
    fabricating a hierarchy that does not exist."""
    if market != "india":
        return {}
    sec_of = {}
    for l in open(g / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        eid = j.get("entity_id")
        if eid in ent:
            s = j.get("std_sector") or "UNK"
            sec_of.setdefault(ent[eid], s)
    return {tk: {"division": s, "major_group": s, "industry": s,
                 "inferred": True} for tk, s in sec_of.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=sorted(MARKETS), default="us")
    a = ap.parse_args()
    global G, OUT
    market = a.market
    G = ROOT / MARKETS[market]["graph"]
    OUT = G / "mcp_cache.json"
    print(f"market={market}  graph={G.name}")

    ent, name_of = load_entities(G, market)
    india_sec = load_sectors(G, market, ent)
    print(f"{len(ent)} resolved entities -> {len(set(ent.values()))} tickers")

    docs = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docs[j["doc_id"]] = {"date": d, "headline": j.get("headline") or "", "chunks": []}
    chunkbuf = collections.defaultdict(list)
    for l in open(G / "lake" / "chunk.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") in docs and j.get("text"):
            chunkbuf[j["doc_id"]].append((j.get("seq", 0), j["text"]))
    for did, parts in chunkbuf.items():
        docs[did]["chunks"] = [t for _, t in sorted(parts)][:4]

    doctypes = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("std_event_type"):
            doctypes[j["doc_id"]].add(j["std_event_type"])

    cells = collections.defaultdict(lambda: {"types": set(), "docs": []})
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        if e not in ent or doc not in docs:
            continue
        c = cells[(ent[e], docs[doc]["date"])]
        c["types"] |= (doctypes.get(doc) or {"other"})
        if doc not in c["docs"]:
            c["docs"].append(doc)
    print(f"{len(cells)} (ticker, day) event cells")

    # ---- prices, with the shared liquidity guard ----
    rets, dropped = {}, 0
    for tk in sorted({t for t, _ in cells}):
        p = G / "prices" / f"{tk.replace('/', '-')}.json"
        if not p.exists():
            continue
        try:
            r = json.loads(p.read_text())["chart"]["result"][0]
            ts = r["timestamp"]; ind = r["indicators"]
            cl = (ind.get("adjclose", [{}])[0].get("adjclose")
                  if "adjclose" in ind else None) or ind["quote"][0]["close"]
        except Exception:
            continue
        dd, px = [], []
        for t, c in zip(ts, cl):
            if c and c > 0:
                dd.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
                px.append(float(c))
        rr = {dd[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}
        if len(rr) > 600 and tradeable(rr):
            rets[tk] = rr
        elif len(rr) > 600:
            dropped += 1
    print(f"{len(rets)} tradeable tickers ({dropped} dropped as stale)")

    dc = collections.Counter()
    for r in rets.values():
        dc.update(r.keys())
    # A trading day is kept when enough of the universe traded on it. The floor must
    # SCALE with the universe: a hardcoded minimum of 30 names silently excludes every
    # day when the universe is smaller than 30, which yields an empty day axis and
    # zero residuals rather than an error. Capped at the universe size for that reason.
    min_names = min(max(8, int(.3 * len(rets))), len(rets))
    days = sorted([d for d, c in dc.items() if c >= min_names])
    print(f"day axis: {len(days)} days (a day needs >= {min_names} of {len(rets)} names)")
    di = {d: i for i, d in enumerate(days)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in days])

    # sector factor = equal-weighted mean residual-to-market within each SIC major group
    # SECTOR FACTOR GRANULARITY — hierarchical, finest level that can support a factor.
    #
    # This used the SIC MAJOR GROUP (2-digit) throughout. Measuring idiosyncratic
    # co-movement by pair type showed that was leaving structure on the table:
    # same-4-digit-industry pairs average |r| 0.174 with 35% above 0.2, against 0.145
    # / 26% for same-major-group. But 4-digit alone is too sparse — only 49 of 180
    # industries have the >=3 members a factor needs, covering 60% of names.
    #
    # So: use the FINEST SIC level that has enough members, per name. 60% of names get
    # a 4-digit industry factor, 27% fall back to major group, 4% to division, 8% get
    # none (their sector component is zero and lands in the idiosyncratic residual,
    # which is the honest treatment — better than assigning them a factor built from
    # firms they have nothing to do with).
    MIN_SECTOR = 3
    if india_sec:
        sect_of = {tk: india_sec.get(tk, {}).get("major_group", "UNK") for tk in rets}
    else:
        levels = [("ind", industry), ("mg", subsector), ("div", division)]
        counts = {tag: collections.Counter(fn(tk) for tk in rets) for tag, fn in levels}
        sect_of = {}
        for tk in rets:
            lab = "UNK"
            for tag, fn in levels:
                g = fn(tk)
                if g != "UNK" and counts[tag][g] >= MIN_SECTOR:
                    lab = f"{tag}:{g}"
                    break
            sect_of[tk] = lab
        used = collections.Counter(s.split(":")[0] for s in sect_of.values())
        print(f"sector factor level: {dict(used)}")
    Y = {tk: np.array([r.get(d, np.nan) for d in days]) for tk, r in rets.items()}

    beta_m, resid_m = {}, {}
    for tk, y in Y.items():
        ok = np.isfinite(y)
        if ok.sum() < 600:
            continue
        e = np.full(len(days), np.nan); b = np.full(len(days), np.nan)
        for i in range(BETA_WIN, len(days)):
            sl = slice(i - BETA_WIN, i); m = ok[sl]
            if m.sum() < 150 or not ok[i]:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean(); den = float(fc @ fc)
            if den > 0:
                bb = float((yy - yy.mean()) @ fc / den)
                b[i] = bb; e[i] = y[i] - bb * F[i]
        beta_m[tk] = b; resid_m[tk] = e
    print(f"{len(resid_m)} names with residuals")

    sect_mean = {}
    for s in set(sect_of.values()):
        mem = [tk for tk in resid_m if sect_of.get(tk) == s]
        if len(mem) >= 3:
            M = np.vstack([resid_m[tk] for tk in mem])
            with np.errstate(invalid="ignore"):
                sect_mean[s] = np.nanmean(M, axis=0)

    # CONSISTENCY FIX. `sigma` used to be built from the MARKET-ONLY residual while
    # `decomp[3]` removed market AND sector — so the gate was testing a different
    # quantity from the "idiosyncratic" shown beside it in the same UI, and every
    # improvement to the sector factor bypassed the gate entirely. Both now use the
    # market-and-sector residual, which is what "idiosyncratic" should mean.
    sigma, decomp = {}, {}
    for tk, e_mkt in resid_m.items():
        s = sect_of.get(tk, "UNK")
        sm = sect_mean.get(s)
        # subtract the sector factor before anything is standardised
        e = e_mkt - sm if sm is not None else e_mkt
        sg, dc2 = {}, {}
        for i in range(BETA_WIN + TRAIL + 5, len(days)):
            if not np.isfinite(e[i]):
                continue
            w = e[i - TRAIL:i]
            if not usable_window(w):
                continue
            sd = float(w[np.isfinite(w)].std())
            sg[i] = round(abs(e[i]) / sd, 3)
            tot = float(Y[tk][i]) if np.isfinite(Y[tk][i]) else None
            if tot is None:
                continue
            mk = float(beta_m[tk][i] * F[i]) if np.isfinite(beta_m[tk][i]) else 0.0
            sc = float(sm[i]) if sm is not None and np.isfinite(sm[i]) else 0.0
            dc2[i] = [round(tot, 6), round(mk, 6), round(sc, 6), round(tot - mk - sc, 6)]
        sigma[tk] = sg; decomp[tk] = dc2

    # ---- correlation neighbours on the idiosyncratic residual ----
    tks = sorted(resid_m)
    R = np.vstack([resid_m[t] for t in tks])
    neigh = {}
    for a in range(len(tks)):
        ra = R[a]
        cors = []
        for b in range(len(tks)):
            if a == b:
                continue
            m = np.isfinite(ra) & np.isfinite(R[b])
            if m.sum() < MIN_OVERLAP:
                continue
            x, y = ra[m], R[b][m]
            sx, sy = x.std(), y.std()
            if sx <= 0 or sy <= 0:
                continue
            c = float(((x - x.mean()) @ (y - y.mean())) / (len(x) * sx * sy))
            cors.append((abs(c), c, tks[b]))
        cors.sort(reverse=True)
        def _ind(x):
            return (india_sec.get(x, {}).get("industry", "UNK") if india_sec
                    else industry(x))
        neigh[tks[a]] = [[t, round(c, 3), _ind(tks[a]) == _ind(t)]
                         for _, c, t in cors[:TOP_NEIGHBOURS]]
    print(f"neighbours for {len(neigh)} names")

    class_counts = collections.defaultdict(collections.Counter)
    for (tk, d), c in cells.items():
        if tk in sigma:
            class_counts[tk].update(c["types"])

    ev = {f"{tk}|{d}": {"types": sorted(c["types"]), "docs": c["docs"][:6]}
          for (tk, d), c in cells.items() if tk in sigma}

    # AUTHORITATIVE EARNINGS DATES from SEC 8-K Item 2.02, aligned to the session
    # the release could FIRST have affected. 49% of releases are accepted at or
    # after 16:00 ET, so dating them by publication (as the news graph does) puts
    # half of them one session early — which splits the reaction across two days
    # and understates the event-day effect by ~45% (2.2x measured, 3.2x aligned).
    # Stored separately rather than merged into `events`: this is a different and
    # better-identified event set, not more of the same.
    cal_path = ROOT / "earnings_calendar.json"
    calendar = {}
    if cal_path.exists() and market == "us":
        cal = json.loads(cal_path.read_text())
        n_shift = 0
        for tk, recs in cal.items():
            if tk not in sigma:
                continue
            out_r = []
            for r in recs:
                i = di.get(r["filed"])
                if i is None:
                    continue
                acc = r.get("accepted") or ""
                after = False
                if len(acc) >= 13:
                    try:
                        after = int(acc[11:13]) >= 16
                    except ValueError:
                        after = False
                j = i + 1 if (after and i + 1 < len(days)) else i
                n_shift += after
                out_r.append({"filed": r["filed"], "session": days[j],
                              "after_close": after, "accepted": acc[:16]})
            if out_r:
                calendar[tk] = out_r
        print(f"calendar: {len(calendar)} tickers, "
              f"{sum(len(v) for v in calendar.values())} earnings releases "
              f"({n_shift} shifted to the next session)")
    used = {doc for v in ev.values() for doc in v["docs"]}

    payload = {
        "days": days,
        "names": {tk: name_of.get(tk, tk) for tk in sigma},
        "market": market, "market_meta": MARKETS[market],
        "sectors": {tk: (india_sec.get(tk, {"division": "UNK", "major_group": "UNK",
                                            "industry": "UNK", "inferred": True})
                         if india_sec else
                         {"division": division(tk), "major_group": subsector(tk),
                          "industry": industry(tk)}) for tk in sigma},
        "sigma": {tk: {str(k): v for k, v in s.items()} for tk, s in sigma.items()},
        "decomp": {tk: {str(k): v for k, v in s.items()} for tk, s in decomp.items()},
        "events": ev,
        "earnings_calendar": calendar,
        "docs": {d: docs[d] for d in used},
        "neighbours": neigh,
        "class_counts": {tk: dict(c) for tk, c in class_counts.items()},
    }
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB): "
          f"{len(sigma)} names, {len(ev)} event cells, {len(used)} documents")


if __name__ == "__main__":
    main()
