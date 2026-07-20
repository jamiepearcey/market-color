# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Full-panel attribution COVERAGE — the question cross_sectional_ic.py can't answer
(it only iterates graph-covered name-days, so it sees precision-on-covered but not
the denominator). Here we compute abnormal-z for EVERY (resolved US name, day) in
the window and ask: of ALL big idiosyncratic movers (|abnormal z|>2), what fraction
does the graph explain (has an edge that day)?  This is the number that was 1.1% on
the 6k corpus and hypothesized to be CORPUS-DENSITY-bound.

Reuses cross_sectional_ic's price cache + factor CSV + the same EWMA-WLS abnormal
return. US listings only ('.'-free symbols), macro/ETF proxies excluded.

Usage: uv run eventgraph/scripts/coverage_panel.py --graph-dir /tmp/eg100k_graph --years 2010,2011,2012
"""
import argparse, json, collections, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012")
    ap.add_argument("--price-from", type=int, default=2008); ap.add_argument("--price-to", type=int, default=2014)
    ap.add_argument("--zcut", type=float, default=2.0)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(","))
    import datetime as dt
    p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; ds = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        Fmap[ds] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])

    # resolved US securities: entity_id -> symbol (US listings only)
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    universe = sorted(set(sym.values()))
    print(f"resolved US universe: {len(universe)} symbols")

    # graph edges -> {(symbol, day): net_dir} over the window (the DIRECTIONAL-COVERED set)
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}
    edge = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        edge[(sym[e], d[:10])] += DIR[j["effect_dir"]]
    print(f"graph edges in window: {len(edge)} (symbol,day) cells over {len(set(k[0] for k in edge))} names")

    # "had a story that day" (the HONEST denominator): the name appears in ANY
    # extracted fact from a doc published that day -- even if no directional edge
    # survived. Gap edge->story = extraction cost (non-directional / gate-dropped /
    # named only as cause). Gap story->raw-text is unmeasured (would need a text scan).
    story = set()
    def add_story(fname, *fields):
        for l in open(lake / fname):
            j = json.loads(l); d = docdate.get(j.get("doc_id"))
            if not d or d[:4] not in years: continue
            for fld in fields:
                e = j.get(fld)
                if e in sym: story.add((sym[e], d[:10]))
    add_story("causal_event_edge.jsonl", "effect_entity", "cause_entity")
    add_story("event.jsonl", "issuer_entity")
    add_story("sensitivity_edge.jsonl", "asset_entity")
    add_story("sentiment_annotation.jsonl", "target_entity")
    print(f"name-days with ANY story: {len(story)} over {len(set(k[0] for k in story))} names")

    # prices for the WHOLE universe (cached + fetch the rest)
    px = {}
    for i, s in enumerate(universe):
        r = logret(yahoo(s, cache, p1, p2))
        if len(r) > 250: px[s] = r
        if (i + 1) % 200 == 0: print(f"  priced {i+1}/{len(universe)} ...", flush=True)
    print(f"{len(px)} names with usable history")

    # abnormal z for every (name, day-in-window): rolling EWMA-WLS on the 9 factors
    big = 0; big_cov = 0; cov_hit = 0; total_obs = 0
    big_story = 0; big_story_cov = 0; big_story_hit = 0
    for s, r in px.items():
        ds = sorted(r)
        for i in range(len(ds)):
            day = ds[i]
            if day[:4] not in years or i < 170 or day not in Fmap: continue
            est = [d for d in ds[i-165:i-11] if d in Fmap]
            if len(est) < 90: continue
            Fc = np.array([Fmap[d] for d in est])
            colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]
            if len(colok) == 0: continue
            rows = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
            if len(rows) < 90: continue
            y = np.array([r[est[k]] for k in rows]); Xf = Fc[np.ix_(rows, colok)]
            X = np.column_stack([np.ones(len(rows)), Xf]); age = np.arange(len(rows))[::-1]; w = 0.5 ** (age / 252)
            beta = wls(y, X, w); resid = y - X @ beta; rstd = resid.std()
            if rstd == 0: continue
            f = Fmap[day][colok]
            if not np.all(np.isfinite(f)): continue
            z = (r[day] - (beta[0] + beta[1:] @ f)) / rstd
            total_obs += 1
            if abs(z) > a.zcut:
                big += 1
                nd = edge.get((s, day)); has_edge = nd is not None and nd != 0
                had_story = (s, day) in story
                if has_edge:
                    big_cov += 1; cov_hit += (np.sign(nd) == np.sign(z))
                if had_story:  # the honest, user-experienced denominator
                    big_story += 1
                    if has_edge:
                        big_story_cov += 1; big_story_hit += (np.sign(nd) == np.sign(z))

    print(f"\n=== attribution coverage over the resolved US panel ({sorted(years)}) ===")
    print(f"  {total_obs} name-day obs; {big} big idiosyncratic movers (|z|>{a.zcut})")
    if big:
        print(f"  RAW coverage (all big movers):        {big_cov:5}/{big} = {big_cov/big:.1%}   [most movers have NO story]")
        print(f"  ...of which had a story that day:     {big_story:5}/{big} = {big_story/big:.1%}   [our news lands on {big_story/big:.0%} of movers]")
    if big_cov:
        print(f"  sign-precision on covered big movers:  {cov_hit/big_cov:.0%}")
    print(f"\n=== CONDITIONAL: big movers that HAD a story (the number a user sees) ===")
    if big_story:
        print(f"  directional recall  (edge | story):   {big_story_cov:5}/{big_story} = {big_story_cov/big_story:.0%}   [had story -> gave a direction]")
    if big_story_cov:
        print(f"  precision on those:                    {big_story_hit/big_story_cov:.0%}")
        print(f"  END-TO-END (correct dir | had story): {big_story_hit:5}/{big_story} = {big_story_hit/big_story:.0%}   [big move + story -> right call]")


if __name__ == "__main__":
    main()
