# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
LATENT NEWS FACTORS via matrix factorization — the central exhibit. Factor the asset x
cause tf-idf matrix (NMF): each latent factor is a weighted set of CAUSES (name it) with
asset LOADINGS. This turns the cause_sim embedding into named, legible factors and, by
printing each factor's top assets WITH SIC sector, makes the CROSS-SECTOR composition
visible -- an 'oil/energy' factor spanning energy + transport, a 'sovereign/credit' factor
spanning banks + insurers + sovereigns -- economic linkages a sector/GICS scheme can't
express in one dimension.

Usage: uv run eventgraph/scripts/news_embedding.py --graph-dir /tmp/eg100k_graph --k 14
"""
import argparse, json, collections, math, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import NMF

SIC = {"60":"banks","61":"credit-inst","62":"brokers","63":"insurance","64":"ins-agents","65":"real-estate",
       "67":"holding/invest","28":"pharma/chem","29":"petroleum-refine","13":"oil&gas-extract","10":"metal-mining",
       "12":"coal","20":"food","21":"tobacco","26":"paper","33":"primary-metal","35":"machinery/computers",
       "36":"electronics","37":"transport-equip","38":"instruments","48":"communications","49":"utilities",
       "50":"wholesale","53":"retail-general","54":"food-retail","57":"furniture-ret","58":"eating/drinking",
       "59":"retail-misc","70":"hotels","73":"business-services/software","78":"motion-pictures","80":"health",
       "45":"air-transport","44":"water-transport","40":"railroads","47":"transport-svcs","51":"wholesale-nondur",
       "16":"heavy-construction","15":"building-construction","32":"stone/glass","34":"fabricated-metal","24":"lumber",
       "27":"printing","30":"rubber/plastic","39":"misc-mfg","23":"apparel","25":"furniture","22":"textile"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--min-asset-edges", type=int, default=8); ap.add_argument("--min-cause-spread", type=int, default=3)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; years = set(a.years.split(","))

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}
    sec = json.loads((gd / "sector.json").read_text()) if (gd / "sector.json").exists() else {}

    # asset x cause counts (over the window); cause names kept for labelling
    ac = collections.defaultdict(collections.Counter); cause_assets = collections.defaultdict(set); aedges = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or not c: continue
        s = sym[e]; ac[s][c] += 1; cause_assets[c].add(s); aedges[s] += 1
    assets = sorted(s for s in ac if aedges[s] >= a.min_asset_edges)
    causes = sorted(c for c, ss in cause_assets.items() if len(ss) >= a.min_cause_spread)
    ci = {c: i for i, c in enumerate(causes)}
    N = len(assets)
    idf = {c: math.log(1 + N / len(cause_assets[c])) for c in causes}
    M = np.zeros((len(assets), len(causes)))
    for ai, s in enumerate(assets):
        for c, n in ac[s].items():
            if c in ci: M[ai, ci[c]] = n * idf[c]
    print(f"asset x cause matrix: {M.shape[0]} assets x {M.shape[1]} causes; factoring into {a.k} latent factors\n")

    nmf = NMF(n_components=a.k, init="nndsvda", max_iter=500, random_state=0)
    W = nmf.fit_transform(M); H = nmf.components_  # W: assets x k ; H: k x causes

    def sec_of(s): return SIC.get(sec.get(s, ""), sec.get(s, "?"))
    def cause_name(c): return c.split("__")[0].replace("_", " ")[:22]

    print("=== latent NEWS FACTORS (name = top causes; assets shown with SIC sector) ===")
    for f in range(a.k):
        top_c = np.argsort(H[f])[::-1][:5]
        label = " / ".join(cause_name(causes[c]) for c in top_c if H[f][c] > 0)
        top_a = np.argsort(W[:, f])[::-1][:8]
        # cross-sector diversity of the top assets (Shannon entropy over sectors)
        secs = [sec_of(assets[i]) for i in top_a if W[i, f] > 0]
        known = [x for x in secs if x not in ("?", "")]
        cnt = collections.Counter(known); H_ent = -sum((n/len(known))*math.log(n/len(known)) for n in cnt.values()) if known else 0
        xs = "CROSS-SECTOR" if len(set(known)) >= 4 else ("mixed" if len(set(known)) >= 2 else "single-sector")
        alist = ", ".join(f"{assets[i]}[{sec_of(assets[i])}]" for i in top_a if W[i, f] > 0)
        print(f"\n  F{f:02d} [{xs}, {len(set(known))} sectors] :: {label}")
        print(f"       {alist}")

    # summary: how many factors span >=4 sectors (the cross-sector economic linkages)?
    nx = 0
    for f in range(a.k):
        top_a = np.argsort(W[:, f])[::-1][:8]
        known = [sec_of(assets[i]) for i in top_a if W[i, f] > 0 and sec_of(assets[i]) not in ("?", "")]
        if len(set(known)) >= 4: nx += 1
    print(f"\n  => {nx}/{a.k} latent factors span >=4 SIC sectors = cross-sector economic linkages "
          f"the causal graph names but a 1-D sector scheme cannot.")


if __name__ == "__main__":
    main()
