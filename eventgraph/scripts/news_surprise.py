# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
CLASS-ADJUSTED NEWS SURPRISE RANKER.

Existing tooling (news_volume.py) surfaces news by RAW abnormality: "this name moved
3sd, here's the news." That ranks an expected earnings-day move the same as a genuinely
strange one. news_volume.py already showed per-std_event_type mean next-day abnormal
vol-z is highly structured (earnings/guidance high, m_and_a/rating_action low) -- i.e.
event classes carry different EXPECTED impact. So rank instead by

    surprise = observed_abnormality(T)  -  expected_for_that_event_class

computed OUT OF SAMPLE: class priors (mean/sd of observed vol_z on the news day itself,
by std_event_type) are fit on an EARLY training window and applied only to a LATER
scoring window, so a class's prior can't be built from the very rows it scores.

TIMING (established in activation_split.py / news_volume.py's contemporaneous check):
the volume spike lands ON the news day T and is >half decayed by T+1, so we score day T
observed abnormality directly (NOT T+1 as news_volume.py's predictive panel does --
that script is answering a different question, incremental forecast power for T+1).

Multi-event-type days: when several std_event_types co-occur on the same (ticker,day),
the CONSERVATIVE choice is used for "expected" -- the MAX prior among the types present,
so we never overstate surprise merely because a low-prior class rode along with a
high-prior one. The type achieving that max is reported as std_event_type/event_group
for the row. Rows with edges but no event_type get their expected value from
std_mechanism if it maps into the same class vocab, else the global fallback prior.

Usage: uv run scripts/news_surprise.py --graph-dir ../data/eg_runs/eg100k_graph
"""
import argparse, json, collections, sys, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import taxonomy as tax

SCHEDULED = {"earnings", "guidance", "monetary_policy", "inflation", "employment",
             "growth", "econ_indicator"}

TRAIN_START, TRAIN_END = "2010-01-01", "2011-06-30"   # class priors fit here
SCORE_START, SCORE_END = "2011-07-01", "2012-12-31"   # scored here (out of sample)


def load_jsonl(path):
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def volume_z_series(symn, cache):
    """day -> log-volume z vs trailing 60 obs (inlined from activation_split.py /
    news_volume.py -- disk-only cached Yahoo chart JSON, no network)."""
    p = cache / f"{symn}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]
        vol = (res["indicators"]["quote"][0].get("volume") or [])
    except Exception:
        return {}
    import datetime as dt
    days, lv = [], []
    for t, v in zip(ts, vol):
        if v:
            days.append(dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"))
            lv.append(math.log(v))
    out = {}
    for i in range(60, len(days)):
        w = np.array(lv[i - 60:i])
        sd = w.std()
        if sd > 0:
            out[days[i]] = (lv[i] - w.mean()) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="../data/eg_runs/eg100k_graph")
    ap.add_argument("--topn", type=int, default=300)
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    lake = gd / "lake"
    cls = gd / "classification"
    cache = gd / "prices"

    # ---- 1. universe: resolved_security entities with usable vol-z series ----
    ent_rows = list(load_jsonl(cls / "entity_class.jsonl"))
    resolved = [r for r in ent_rows if r["resolution_status"] == "resolved_security"]
    ticker_of_entity, sector_of_entity, name_of_entity = {}, {}, {}
    for r in resolved:
        ticker_of_entity[r["entity_id"]] = r["resolved_ticker"]
        sector_of_entity[r["entity_id"]] = r["std_sector"]
        name_of_entity[r["entity_id"]] = r["name"]

    vz_by_ticker = {}
    seen = set()
    for r in resolved:
        t = r["resolved_ticker"]
        if t in seen:
            continue
        seen.add(t)
        symn = t.replace("/", "-")
        vz = volume_z_series(symn, cache)
        vz = {d: z for d, z in vz.items() if d[:4] in ("2010", "2011", "2012")}
        if len(vz) >= 300:
            vz_by_ticker[t] = vz
    print(f"tickers with usable vol-z series: {len(vz_by_ticker)}", file=sys.stderr)

    # ---- 2. docs: date, headline, published_at ----
    docdate, headline_of, pub_of = {}, {}, {}
    for j in load_jsonl(lake / "document.jsonl"):
        pa = j.get("published_at")
        if pa:
            docdate[j["doc_id"]] = pa[:10]
            headline_of[j["doc_id"]] = j.get("headline") or ""
            pub_of[j["doc_id"]] = pa

    # ---- 3. edges (mechanism, quote) and events (event_type) per (ticker,day) ----
    edge_class_rows = list(load_jsonl(cls / "edge_class.jsonl"))
    edge_rows = list(load_jsonl(lake / "causal_event_edge.jsonl"))
    quote_by_key = {}  # (doc_id, effect_entity) -> quote text
    for j in edge_rows:
        q = (j.get("quote") or j.get("effect_verbatim") or "").strip()
        if q:
            quote_by_key[(j["doc_id"], j.get("effect_entity"))] = q

    day_edges = collections.defaultdict(lambda: {"n": 0, "mechs": collections.Counter(),
                                                   "docs": set()})
    for j in edge_class_rows:
        eff = j["effect_entity"]
        t = ticker_of_entity.get(eff)
        if not t or t not in vz_by_ticker:
            continue
        d = docdate.get(j["doc_id"])
        if not d:
            continue
        key = (t, d)
        rec = day_edges[key]
        rec["n"] += 1
        rec["mechs"][j["std_mechanism"]] += 1
        rec["docs"].add((j["doc_id"], eff))

    event_rows = list(load_jsonl(lake / "event.jsonl"))
    event_class_by_id = {j["event_id"]: j for j in load_jsonl(cls / "event_class.jsonl")}
    day_events = collections.defaultdict(lambda: collections.Counter())  # (t,d) -> {std_event_type: n}
    day_event_docs = collections.defaultdict(set)  # (t,d) -> {doc_id}
    for ev in event_rows:
        t = ticker_of_entity.get(ev.get("issuer_entity"))
        if not t or t not in vz_by_ticker:
            continue
        d = docdate.get(ev.get("doc_id"))
        if not d:
            continue
        ec = event_class_by_id.get(ev["event_id"])
        if not ec:
            continue
        day_events[(t, d)][ec["std_event_type"]] += 1
        day_event_docs[(t, d)].add(ev.get("doc_id"))

    all_keys = set(day_edges) | set(day_events)
    print(f"distinct (ticker,day) news keys: {len(all_keys)}", file=sys.stderr)

    # ---- 4. build per-key row: obs_vol_z on day T + candidate event types ----
    def event_types_for(t, d):
        types = set(day_events.get((t, d), {}))
        if types:
            return types
        # fallback: mechanisms from edges that land in the same taxonomy vocab
        mechs = day_edges.get((t, d), {}).get("mechs", {})
        return {m for m in mechs if m in tax.STD_EVENT_TYPES}

    keyed_rows = {}
    for t, d in all_keys:
        vz = vz_by_ticker[t]
        if d not in vz:
            continue
        types = event_types_for(t, d)
        keyed_rows[(t, d)] = dict(obs_vol_z=vz[d], types=types)

    # ---- 5. class priors on TRAIN window (mean/sd of obs_vol_z on the news day,
    # keyed by std_event_type; any type present on that day contributes its
    # obs_vol_z to EACH of its own types -- days with a single type dominate) ----
    train_by_type = collections.defaultdict(list)
    for (t, d), rec in keyed_rows.items():
        if not (TRAIN_START <= d <= TRAIN_END):
            continue
        for et in rec["types"]:
            train_by_type[et].append(rec["obs_vol_z"])

    global_train_vals = [v for et in train_by_type.values() for v in et]
    global_mean = float(np.mean(global_train_vals)) if global_train_vals else 0.0
    global_sd = float(np.std(global_train_vals)) if global_train_vals else 1.0

    priors = {}
    for et, vals in train_by_type.items():
        n = len(vals)
        if n >= 20:
            priors[et] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)) or None,
                               n=n, prior_source="class")
        else:
            priors[et] = dict(mean=global_mean, sd=global_sd or None, n=n,
                               prior_source="global")
    # any type never seen in train at all
    for et in tax.STD_EVENT_TYPES:
        if et not in priors:
            priors[et] = dict(mean=global_mean, sd=global_sd or None, n=0,
                               prior_source="global")

    n_train_days = sum(1 for (t, d) in keyed_rows if TRAIN_START <= d <= TRAIN_END)

    # ---- 6. score the LATER window ----
    scored = []
    n_score_days = 0
    for (t, d), rec in sorted(keyed_rows.items(), key=lambda kv: kv[0][1]):
        if not (SCORE_START <= d <= SCORE_END):
            continue
        n_score_days += 1
        types = rec["types"]
        obs = rec["obs_vol_z"]
        if types:
            # conservative: expected = MAX prior mean among types present
            best_et = max(types, key=lambda et: priors[et]["mean"])
        else:
            best_et = "other"
        pr = priors[best_et]
        expected = pr["mean"]
        surprise = obs - expected
        surprise_sd = surprise / pr["sd"] if pr.get("sd") else None
        group = tax.EVENT_GROUP.get(best_et, "other")

        ne = day_edges.get((t, d))
        n_edges = ne["n"] if ne else 0
        top_mech = ne["mechs"].most_common(1)[0][0] if ne and ne["mechs"] else None

        docs = set()
        if ne:
            docs |= {doc for doc, eff in ne["docs"]}
        docs |= day_event_docs.get((t, d), set())
        docs = sorted(docs, key=lambda doc: pub_of.get(doc, ""))
        doc_id = docs[0] if docs else None
        headline = headline_of.get(doc_id, "") if doc_id else ""
        published_at = pub_of.get(doc_id) if doc_id else None
        quote = ""
        if ne:
            for doc, eff in ne["docs"]:
                q = quote_by_key.get((doc, eff))
                if q:
                    quote = q[:200]
                    break

        eid = next((e for e, tk in ticker_of_entity.items() if tk == t), None)

        scored.append(dict(
            date=d, ticker=t, entity_id=eid, name=name_of_entity.get(eid, t),
            std_sector=sector_of_entity.get(eid, "UNK"), std_event_type=best_et,
            event_group=group, std_mechanism=top_mech,
            obs_vol_z=round(obs, 4), expected_vol_z=round(expected, 4),
            surprise=round(surprise, 4),
            surprise_sd=round(surprise_sd, 4) if surprise_sd is not None else None,
            tape_confirmed=bool(abs(obs) >= 2.0),
            unscheduled=bool(group != "macro" and best_et not in SCHEDULED),
            n_edges=n_edges, headline=headline, quote=quote, doc_id=doc_id,
            published_at=published_at,
        ))

    scored.sort(key=lambda r: -r["surprise"])

    out_dir = gd
    (out_dir / "news_surprise.json").write_text(json.dumps(dict(
        train_window=[TRAIN_START, TRAIN_END], score_window=[SCORE_START, SCORE_END],
        n_train_days=n_train_days, n_score_rows=len(scored),
        n_tickers=len(vz_by_ticker), priors=priors, rows=scored,
    ), indent=None))
    top = scored[:a.topn]
    (out_dir / "news_surprise_top.json").write_text(json.dumps(dict(
        train_window=[TRAIN_START, TRAIN_END], score_window=[SCORE_START, SCORE_END],
        n_train_days=n_train_days, n_score_rows=len(scored),
        n_tickers=len(vz_by_ticker), priors=priors, rows=top,
    )))

    # ---- 7. summary ----
    n_unsched = sum(1 for r in scored if r["unscheduled"])
    n_tape = sum(1 for r in scored if r["tape_confirmed"])
    print(f"\nN scored rows: {len(scored)}  (train days={n_train_days}, "
          f"score-window candidate days={n_score_days})")
    print(f"tickers: {len(vz_by_ticker)}  unscheduled: {n_unsched} "
          f"({100*n_unsched/len(scored):.1f}%)  tape_confirmed: {n_tape} "
          f"({100*n_tape/len(scored):.1f}%)")
    print("\nout-of-sample class priors (train window, n>=20 else global fallback):")
    for et in sorted(priors, key=lambda e: -priors[e]["mean"]):
        p = priors[et]
        if p["n"] == 0 and p["prior_source"] == "global":
            continue
        print(f"  {et:18s} mean={p['mean']:+.3f}  sd={p['sd'] and round(p['sd'],3)}  "
              f"n={p['n']:4d}  source={p['prior_source']}")
    print("\ntop 10 by surprise:")
    for r in scored[:10]:
        print(f"  {r['date']} {r['ticker']:8s} {r['std_event_type']:15s} "
              f"obs={r['obs_vol_z']:+.2f} exp={r['expected_vol_z']:+.2f} "
              f"surprise={r['surprise']:+.2f}  {r['headline'][:70]}")


if __name__ == "__main__":
    main()
