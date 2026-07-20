# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
ACTIVATION-FIRST (inverted pipeline). The project's recurring lesson: the graph knows
lots of narratives; the market tells you which matter. So invert the workflow —
don't build a graph and ask "what predicts?"; observe an ACTIVATION and retrieve.

  1. OBSERVE activations: |abnormal z| >= zcut (proxy-neutralized) or volume z >= vcut
     across the resolved US universe over the last N trading days.
  2. FOCUSED RETRIEVAL per activation: every fact mentioning the name in the lookback
     (causal edges as effect AND as cause, events, sentiments, sensitivities),
     first-hop entities, shared-driver names, related companies (relation edges).
  3. LOCAL CAUSAL GRAPH: which causes recur; which mechanisms connect multiple firms;
     where explanations DISAGREE (opposing directions on the same cause).
  4. DOMINANT NARRATIVE (deterministic template — no LLM in the loop):
     primary driver + consensus, secondary channels, disagreements, spillover
     candidates (names sharing the driver, with their own recent z), verbatim
     provenance. Honest empty state: "UNEXPLAINED — no narrative in corpus."

The inverted coverage metric: fraction of ACTIVATIONS with a narrative (news clusters
around big moves, so retrieval-on-demand should beat the broad 2.2% story-coverage).

Usage: uv run scripts/activation_first.py --graph-dir ../data/eg_runs/eg_live2 --days 7
"""
import argparse, json, collections, math, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import persistence_signal as ps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="../data/eg_runs/eg_live2")
    ap.add_argument("--days", type=int, default=7, help="activation scan window (trading days)")
    ap.add_argument("--lookback", type=int, default=7, help="news retrieval window before activation (calendar days)")
    ap.add_argument("--zcut", type=float, default=2.5); ap.add_argument("--vcut", type=float, default=3.0)
    ap.add_argument("--max-names", type=int, default=300); ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)

    # resolved US universe (whole graph, not just news-dense names — that's the inversion)
    sym, ent_of = {}, collections.defaultdict(set)
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in ps.SKIP:
            sym[j["entity_id"]] = j["symbol"]; ent_of[j["symbol"]].add(j["entity_id"])
    docs = {}
    for l in open(lake / "document.jsonl"):
        j = json.loads(l); docs[j["doc_id"]] = {"day": (j.get("published_at") or "")[:10],
                                               "headline": j.get("headline") or "", "source": j.get("source") or ""}
    universe = sorted(set(sym.values()))[:a.max_names]
    print(f"universe: {len(universe)} resolved US names; fetching prices ...")
    p1 = int(dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime.now(dt.UTC).timestamp())
    prox = ps.fetch_returns(ps.PROXIES, cache, p1, p2)
    px = ps.fetch_returns(universe, cache, p1, p2)
    ar = ps.neutralize(px, {p: prox[p] for p in ps.PROXIES if p in prox})
    print(f"{len(ar)} names with neutralized series")

    # 1. OBSERVE — activations over the last N trading days
    acts = []
    for s_, series in ar.items():
        ds = sorted(series); vals = [series[d] for d in ds]
        for k in range(max(20, len(ds) - a.days), len(ds)):
            w = np.array(vals[max(0, k-60):k]); sd = w.std()
            if sd == 0: continue
            z = vals[k] / sd
            vz = ps.vol_z(s_, cache, ds[k])
            if abs(z) >= a.zcut or (vz is not None and vz >= a.vcut):
                acts.append({"sym": s_, "day": ds[k], "abn_z": round(float(z), 1),
                             "vol_z": None if vz is None else round(vz, 1)})
    acts.sort(key=lambda x: -abs(x["abn_z"]))
    print(f"\n{len(acts)} activations (|z|>={a.zcut} or vol_z>={a.vcut}) in last {a.days} trading days\n")

    # preload fact tables once
    causal, events, sents, sens, rels = [], [], [], [], []
    for l in open(lake / "causal_event_edge.jsonl"): causal.append(json.loads(l))
    for l in open(lake / "event.jsonl"): events.append(json.loads(l))
    for l in open(lake / "sentiment_annotation.jsonl"): sents.append(json.loads(l))
    for l in open(lake / "sensitivity_edge.jsonl"): sens.append(json.loads(l))
    if (lake / "relation_edge.jsonl").exists():
        for l in open(lake / "relation_edge.jsonl"): rels.append(json.loads(l))

    def in_window(doc_id, day):
        d = docs.get(doc_id, {}).get("day")
        if not d: return False
        try:
            dd = (dt.date.fromisoformat(day) - dt.date.fromisoformat(d)).days
        except ValueError:
            return False
        return 0 <= dd <= a.lookback

    explained = 0; out = []
    for act in acts[:a.top]:
        s_, day = act["sym"], act["day"]; eids = ent_of[s_]
        # 2. FOCUSED RETRIEVAL
        mine = [e for e in causal if (e.get("effect_entity") in eids or e.get("cause_entity") in eids) and in_window(e.get("doc_id"), day)]
        my_events = [e for e in events if e.get("issuer_entity") in eids and in_window(e.get("doc_id"), day)]
        my_sents = [e for e in sents if e.get("target_entity") in eids and in_window(e.get("doc_id"), day)]
        my_sens = [e for e in sens if e.get("asset_entity") in eids and in_window(e.get("doc_id"), day)]
        my_rels = [e for e in rels if (e.get("source_entity") in eids or e.get("target_entity") in eids) and in_window(e.get("doc_id"), day)]
        # 3. LOCAL CAUSAL GRAPH
        DIRV = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
        by_cause = collections.defaultdict(lambda: {"n": 0, "net": 0, "up": 0, "dn": 0, "mechs": collections.Counter(), "quote": None, "doc": None})
        for e in mine:
            if e.get("effect_entity") not in eids: continue
            c = (e.get("cause_entity") or "?").split("__")[0]
            g = by_cause[c]; g["n"] += 1; g["mechs"][e.get("mechanism") or "?"] += 1
            d_ = DIRV.get(e.get("effect_dir"))
            if d_ is not None:
                g["net"] += d_; g["up"] += d_ > 0; g["dn"] += d_ < 0
            if g["quote"] is None and e.get("quote"): g["quote"], g["doc"] = e["quote"][:110], e.get("doc_id")
        # spillover: other names driven by my top causes in the window
        spill = collections.Counter()
        topc = {c for c, _ in sorted(by_cause.items(), key=lambda kv: -kv[1]["n"])[:3]}
        if topc:
            for e in causal:
                c = (e.get("cause_entity") or "").split("__")[0]
                if c in topc and in_window(e.get("doc_id"), day):
                    t = e.get("effect_entity")
                    if t in sym and sym[t] != s_: spill[sym[t]] += 1
        rel_names = sorted({sym[x] for e in my_rels for x in (e.get("source_entity"), e.get("target_entity"))
                            if x in sym and sym[x] != s_})
        n_facts = len(mine) + len(my_events) + len(my_sents) + len(my_sens)
        has_narr = bool(by_cause) or n_facts >= 2
        explained += bool(has_narr)
        # 4. DOMINANT NARRATIVE (template)
        rep = {"activation": act, "n_facts": n_facts, "explained": bool(has_narr)}
        print(f"── ACTIVATION {s_} @ {day}   abn_z {act['abn_z']:+}   vol_z {act['vol_z']}")
        if not has_narr:
            print("   UNEXPLAINED — no narrative in corpus lookback window\n"); out.append(rep); continue
        ranked = sorted(by_cause.items(), key=lambda kv: -kv[1]["n"])
        if ranked:
            c, g = ranked[0]
            cons = f"consensus {'down' if g['net']<0 else 'up'} ({max(g['up'],g['dn'])}:{min(g['up'],g['dn'])})" if (g["up"] or g["dn"]) else "no direction"
            print(f"   dominant driver : {c}  ({g['n']} edges; mechanisms {'/'.join(m for m,_ in g['mechs'].most_common(2))}; {cons})")
            if g["quote"]:
                dd = docs.get(g["doc"], {})
                print(f"      \"{g['quote']}\" — {dd.get('headline','')[:60]} [{dd.get('source','')}]")
            for c2, g2 in ranked[1:3]:
                print(f"   secondary       : {c2}  ({g2['n']} edges, {'/'.join(m for m,_ in g2['mechs'].most_common(1))})")
            dis = [(c3, g3) for c3, g3 in ranked if g3["up"] and g3["dn"]]
            if dis:
                print(f"   DISAGREEMENT    : " + "; ".join(f"{c3} ({g3['up']}up/{g3['dn']}dn)" for c3, g3 in dis[:2]))
        if my_events: print(f"   events          : " + "; ".join((e.get('event_type') or '?') for e in my_events[:4]))
        if spill:
            sp = [f"{n}({k})" + (f" z={round(list(ar[n].values())[-1]/ (np.std(list(ar[n].values())[-60:]) or 1),1)}" if n in ar else "")
                  for n, k in spill.most_common(5)]
            print(f"   spillover cands : {', '.join(sp)}")
        if rel_names: print(f"   related (graph) : {', '.join(rel_names[:5])}")
        print()
        rep.update({"drivers": [(c, g["n"]) for c, g in ranked[:3]], "spillover": spill.most_common(5)})
        out.append(rep)

    print(f"=== INVERTED COVERAGE: {explained}/{min(a.top, len(acts))} top activations have a narrative "
          f"(vs 2.2% broad story-coverage on the 2010-12 panel) ===")
    with open(gd / "activation_first_report.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"report -> {gd / 'activation_first_report.jsonl'}")


if __name__ == "__main__":
    main()
