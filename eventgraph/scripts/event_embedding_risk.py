# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx", "sentence-transformers", "torch"]
# ///
"""
EMBEDDING-DEFINED RISK BASKETS x EVENT BEFORE/AFTERS.

The creative extension of event_regime_correlation.py. There, the "rate cluster"
was hand-picked, so its FOMC de-diversification was probably just a GICS sector
factor. Here the basket is defined by an EMBEDDING axis instead of by hand:

  1. RISK ANCHORS   short risk phrases ("interest-rate hikes, monetary
                    tightening, bond yields") -> embedding vectors.
  2. COMPANY VECTOR mean embedding of a name's news quotes (embed_exposure.py's
                    recipe) -> a NARRATIVE risk fingerprint, orthogonal to GICS.
  3. RELATED        nearest companies to an anchor (cosine) -> a risk-themed
                    basket that CUTS ACROSS sectors.
  4. BEFORE/AFTERS  the pre/impact/post event-regime residual correlation on the
                    basket, matched anchor<->event (rate<->FOMC), computed on
                    SECTOR-NEUTRALISED idio returns (9 macro factors + 9 SPDR
                    sectors, leave-one-month-out — attribution_data.py's recipe).

The test that makes this worth doing: since we already strip the 9 sector
factors, a pure-GICS basket's co-movement is mostly gone. If the CROSS-SECTOR
embedding basket STILL de-diversifies on FOMC in the idio residuals — and a
GICS-sector basket and a random basket do not — the embedding captured a risk
theme industry classification misses. If it just recovers a sector, honest null.

Controls kept from the parent spike: GICS-basket + random-basket comparison,
placebo-null date distribution, split-half OOS.

Embedding backend: hand-rolled tf-idf by default (numpy only, always runs);
--st switches to all-MiniLM-L6-v2 (needs sentence-transformers+torch). tf-idf is
a crude proxy — treat --st as the real test.

Usage:
  uv run scripts/event_embedding_risk.py --anchor rate
  uv run scripts/event_embedding_risk.py --anchor energy --st --topn 20
"""
import argparse, json, sys, re, math, collections, datetime as dt
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import FN, SKIP, yahoo, logret
from event_regime_correlation import regime_daysets, mean_offdiag, pair_corr
# SIC (SEC EDGAR), not GICS. GICS is proprietary so gics.py was ~140 tickers
# assigned BY HAND — 12% of the universe, unverifiable, and flat. SIC is public
# domain, assigned per filer by the SEC itself (74% coverage here), and genuinely
# hierarchical: division -> major group -> 4-digit industry, which is what the
# drill-down needs. SIC is an older and coarser taxonomy than GICS; it is used
# because it is real and checkable, which beats a better-shaped invented one.
from sic import sector as gics_sector, subsector, industry  # noqa: E402,F401

SPDR = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
SN = {"XLF": "Financials", "XLK": "Tech", "XLE": "Energy", "XLV": "Health", "XLI": "Industrials",
      "XLY": "ConsDisc", "XLP": "Staples", "XLU": "Utilities", "XLB": "Materials"}
DIRV = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
YEARS = {"2010", "2011", "2012"}

RISK_ANCHORS = {
    "rate": "interest rate hikes, monetary policy tightening, Federal Reserve, rising bond yields, higher borrowing costs",
    "credit": "credit spreads widening, default risk, corporate bond stress, rating downgrade, refinancing risk, leverage",
    "energy": "oil price shock, crude oil, energy costs, gasoline prices, commodity price spike",
    "regulatory": "regulation, antitrust, government investigation, lawsuit, litigation, fines, subpoena",
    "demand": "weak consumer demand, recession, slowing sales, economic slowdown, falling revenue",
}

# Real event days 2010-2012 (public, scheduled ahead). Verify vs the formal
# calendar for production. FOMC decision days (24) and OPEC ordinary/extra
# meeting decision days (~7). OPEC is the sparse OIL catalyst that fits the
# before/after regime — but N=7 in this window is UNDERPOWERED for the placebo /
# OOS gates (treat OPEC results as illustrative, not conclusive). EIA weekly
# petroleum fires ~52x/yr — too dense for a sparse regime (no clean baseline),
# so it is deliberately not used here.
DEMO_FOMC = [
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
]
DEMO_OPEC = [
    "2010-03-17", "2010-10-14", "2010-12-11",
    "2011-06-08", "2011-12-14",
    "2012-06-14", "2012-12-12",
]
EVENTS = {"fomc": DEMO_FOMC, "opec": DEMO_OPEC}
# Sensible anchor -> matching-event default (override with --event).
ANCHOR_EVENT = {"rate": "fomc", "credit": "fomc", "demand": "fomc", "energy": "opec", "regulatory": "fomc"}

STOP = set("the a an of to in on for and or with at by from as is are was were be been being this that "
           "it its their his her they we you i he she will would could should may might can has have had "
           "not no s t re said say says after into over under up down out about than then more most".split())


def load_quotes(G, n_univ):
    """company symbol -> concatenated news-quote text (2010-2012), top names by
    causal-edge frequency (the embedding corpus, per embed_exposure.py)."""
    sym = {}
    for l in open(G / "entity_symbol.jsonl"):
        j = json.loads(l)
        s = j.get("symbol", "")
        if j["kind"] == "security" and "." not in s and not s.startswith("^") and s not in SKIP:
            sym[j["entity_id"]] = s
    dmeta = {json.loads(l)["doc_id"]: ((json.loads(l).get("published_at") or "")[:7], json.loads(l).get("headline") or "")
             for l in open(G / "lake" / "document.jsonl")}
    freq = collections.Counter(); quotes = collections.defaultdict(list)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l); mh = dmeta.get(j.get("doc_id")); e = j.get("effect_entity"); d = j.get("effect_dir")
        if not mh or mh[0][:4] not in YEARS or e not in sym or d not in DIRV:
            continue
        freq[sym[e]] += 1
        q = (j.get("quote") or "").strip()
        if q and len(quotes[sym[e]]) < 60:
            quotes[sym[e]].append((q + ". " + mh[1])[:200])
    top = [s for s, _ in freq.most_common() if len(quotes[s]) >= 4][:n_univ]
    return {s: " ".join(quotes[s]) for s in top}, freq


def tokenize(t):
    return [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP]


def tfidf(docs):
    """dict name->text  ->  (names, dense l2-normalised tf-idf matrix, vocab idf).
    Anchors are projected into the same space by project()."""
    names = list(docs)
    df = collections.Counter()
    toks = {}
    for s in names:
        tk = tokenize(docs[s]); toks[s] = tk
        for w in set(tk):
            df[w] += 1
    N = len(names)
    vocab = [w for w, c in df.most_common() if 3 <= c <= 0.5 * N][:4000]
    vi = {w: i for i, w in enumerate(vocab)}
    idf = np.array([math.log(N / df[w]) for w in vocab])
    M = np.zeros((N, len(vocab)))
    for r, s in enumerate(names):
        c = collections.Counter(w for w in toks[s] if w in vi)
        for w, k in c.items():
            M[r, vi[w]] = k
        M[r] *= idf
        nrm = np.linalg.norm(M[r])
        if nrm > 0:
            M[r] /= nrm
    return names, M, (vi, idf)


def project_tfidf(text, space):
    vi, idf = space
    v = np.zeros(len(vi))
    for w, k in collections.Counter(w for w in tokenize(text) if w in vi).items():
        v[vi[w]] = k
    v *= idf
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_st(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    v = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # EMB_CENTER=1 -> mean-centre the embedding space (F43). Transformer spaces are
    # anisotropic: uncentred cosine is dominated by proximity to the corpus centroid,
    # which tracks coverage volume and therefore firm size. Two results died here.
    if __import__("os").environ.get("EMB_CENTER"):
        import numpy as _np
        v = v - v.mean(0)
        _n = _np.linalg.norm(v, axis=1, keepdims=True); v = v / _np.where(_n > 0, _n, 1)
        print("[EMB_CENTER] embedding space mean-centred", flush=True)
    return np.asarray(v)


def idio_residuals(names, G, cache, p1, p2):
    """macro (9 FN factors) + sector (9 SPDR) leave-one-month-out residuals ->
    ar[name] = {day: idio}. Ported from attribution_data.py."""
    Fmap = {}
    for row in __import__("csv").DictReader(open(G / "factor_snapshot_factor_returns.csv")):
        d = row["date"]
        Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array(
            [float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])
    ret = lambda t: logret(yahoo(t, cache, p1, p2))
    spdr = {e: ret(e) for e in SPDR}
    have = [s for s in names if sum(1 for d in ret(s) if d[:4] in YEARS) > 300]
    alld = sorted({d for s in have for d in ret(s) if d[:4] in YEARS and d in Fmap and all(d in spdr[e] for e in SPDR)})
    mcol = [i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld) >= len(alld) * 0.5]
    ar = {}
    for s in have:
        r = ret(s); days = [d for d in alld if d in r]
        if len(days) < 200:
            continue
        Y = np.array([r[d] for d in days])
        if np.max(np.abs(Y)) > 0.35 or np.std(Y) > 0.06:
            continue
        Xm = np.nan_to_num(np.column_stack([np.ones(len(days))] + [[Fmap[d][i] for d in days] for i in mcol]))
        Xs = np.column_stack([np.ones(len(days))] + [[spdr[e][d] for d in days] for e in SPDR])
        dmon = np.array([d[:7] for d in days])
        macfit = np.zeros(len(days)); secfit = np.zeros(len(days))
        for m_ in sorted(set(dmon)):
            te = dmon == m_; tr = ~te
            if tr.sum() < 120:
                continue
            bm, *_ = np.linalg.lstsq(Xm[tr], Y[tr], rcond=None); macfit[te] = Xm[te][:, 1:] @ bm[1:]
        res1 = Y - macfit
        for m_ in sorted(set(dmon)):
            te = dmon == m_; tr = ~te
            if tr.sum() < 120:
                continue
            bs, *_ = np.linalg.lstsq(Xs[tr], res1[tr], rcond=None); secfit[te] = Xs[te][:, 1:] @ bs[1:]
        idio = res1 - secfit
        ar[s] = {d: idio[i] for i, d in enumerate(days)}
    return ar, spdr


def sector_of(s, ar, spdr):
    best = ("?", -1)
    for e in SPDR:
        cm = [d for d in ar[s] if d in spdr[e]]
        if len(cm) < 40:
            continue
        x = np.array([ar[s][d] for d in cm]); y = np.array([spdr[e][d] for d in cm])
        # note: ar is sector-neutralised, so this is a weak tag from residual overlap
        if x.std() and y.std():
            cc = abs(np.corrcoef(x, y)[0, 1])
            if cc > best[1]:
                best = (SN[e], cc)
    return best[0]


def regime_report(tag, ar, names, trading_days, anchors, args, base_hint=None):
    by_off, pre_s, imp_s, post_s, base_s, n_ev = regime_daysets(trading_days, anchors, args.pre, args.impact, args.post)
    base_c, _ = mean_offdiag(ar, names, base_s)
    imp_c, _ = mean_offdiag(ar, names, imp_s)
    pre_c, _ = mean_offdiag(ar, names, pre_s)
    post_c, _ = mean_offdiag(ar, names, post_s)
    print(f"  {tag:22} base {base_c:+.3f}  pre {pre_c-base_c:+.3f}  IMPACT {imp_c-base_c:+.3f}  post {post_c-base_c:+.3f}   (|basket|={len(names)})")
    return {"base": base_c, "impact_delta": imp_c - base_c, "n_ev": n_ev}


def placebo_percentile(ar, names, trading_days, real_delta, base_c, args, k, B=200):
    rng = np.random.default_rng(args.seed + 7)
    lo = dt.date(2010, 1, 15).toordinal(); hi = dt.date(2012, 12, 15).toordinal()
    null = []
    for _ in range(B):
        ds = sorted({dt.date.fromordinal(int(x)).isoformat() for x in rng.integers(lo, hi, k * 3)})[:k]
        _, _, imp2, _, _, _ = regime_daysets(trading_days, ds, args.pre, args.impact, args.post)
        c2, _ = mean_offdiag(ar, names, imp2)
        if np.isfinite(c2):
            null.append(c2 - base_c)
    null = np.array(null)
    return float(np.mean(null < real_delta)), float(null.mean()), float(np.percentile(null, 95))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", default="../data/eg_runs/eg100k_graph")
    ap.add_argument("--anchor", default="rate", choices=list(RISK_ANCHORS))
    ap.add_argument("--event", default=None, choices=list(EVENTS),
                    help="event set for the before/afters (default: anchor's matching event)")
    ap.add_argument("--topn", type=int, default=18, help="basket size")
    ap.add_argument("--n-univ", type=int, default=140, help="embedding universe (top by edge frequency)")
    ap.add_argument("--st", action="store_true", help="use all-MiniLM embeddings (needs sentence-transformers+torch)")
    ap.add_argument("--pre", type=int, default=3)
    ap.add_argument("--impact", type=int, default=1)
    ap.add_argument("--post", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260725)
    a = ap.parse_args()
    G = Path(a.graph_dir); cache = G / "prices"
    p1 = int(dt.datetime(2008, 1, 1, tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime(2014, 12, 31, tzinfo=dt.UTC).timestamp())

    # ---- embeddings ----
    docs, freq = load_quotes(G, a.n_univ)
    print(f"corpus: {len(docs)} companies with >=4 news quotes (2010-2012)")
    anchor_text = RISK_ANCHORS[a.anchor]
    if a.st:
        names = list(docs)
        V = embed_st([docs[s] for s in names]); av = embed_st([anchor_text])[0]
        backend = "all-MiniLM-L6-v2"
    else:
        names, V, space = tfidf(docs); av = project_tfidf(anchor_text, space)
        backend = "tf-idf"
    sims = V @ av
    order = np.argsort(-sims)
    basket = [names[i] for i in order[:a.topn]]
    print(f"\n[{backend}] anchor '{a.anchor}': {anchor_text}")
    print(f"  nearest {a.topn}: {', '.join(f'{names[i]}({sims[i]:.2f})' for i in order[:a.topn])}")

    # ---- neutralised idio residuals for the embedding basket + controls ----
    # Universe to residualise: the basket + a random control + the rest (for GICS control).
    rng = np.random.default_rng(a.seed)
    rest = [s for s in names if s not in basket]
    rand_basket = list(rng.choice(rest, min(a.topn, len(rest)), replace=False))
    need = sorted(set(basket) | set(rand_basket) | set(names))  # residualise the whole universe once
    print(f"\nneutralising {len(need)} names on 9 macro + 9 sector factors (leave-one-month-out) ...")
    ar, spdr = idio_residuals(need, G, cache, p1, p2)
    basket = [s for s in basket if s in ar]
    rand_basket = [s for s in rand_basket if s in ar]
    trading_days = sorted(set().union(*[set(v) for v in ar.values()]))

    # GICS control (REAL classification, gics.py): the dominant KNOWN sector in
    # the embedding basket, then the top-N OTHER same-sector names as a
    # same-sector, non-embedding basket.
    sec_tags = collections.Counter(gics_sector(s) for s in basket)
    known_tags = collections.Counter({k: v for k, v in sec_tags.items() if k != "UNK"})
    dom_sec = known_tags.most_common(1)[0][0] if known_tags else "UNK"
    gics_basket = [s for s in names if s in ar and s not in basket and gics_sector(s) == dom_sec][:a.topn]
    cross = len(known_tags)
    print(f"  basket GICS spread (real): {dict(sec_tags)}  -> spans {cross} known sectors; dominant={dom_sec}")

    event_key = a.event or ANCHOR_EVENT.get(a.anchor, "fomc")
    anchors = [d for d in EVENTS[event_key] if trading_days and trading_days[0] <= d <= trading_days[-1]]
    powerwarn = " [UNDERPOWERED: N<12]" if len(anchors) < 12 else ""
    print(f"\n=== {event_key.upper()} BEFORE/AFTERS on idio residuals  [{a.anchor} anchor, {len(anchors)} events{powerwarn}, "
          f"windows pre[-{a.pre},-1] impact[0,{a.impact}] post] ===")
    emb = regime_report(f"embedding[{a.anchor}]", ar, basket, trading_days, anchors, a)
    regime_report(f"GICS[{dom_sec}]", ar, gics_basket, trading_days, anchors, a) if gics_basket else None
    regime_report("random", ar, rand_basket, trading_days, anchors, a)

    # ---- placebo null + OOS for the embedding basket ----
    pct, nmean, n95 = placebo_percentile(ar, basket, trading_days, emb["impact_delta"], emb["base"], a, len(anchors))
    print(f"\n  placebo null (embedding basket): impact-Δ null mean {nmean:+.3f}, 95th {n95:+.3f}; "
          f"real {emb['impact_delta']:+.3f} at {pct:.0%} pct (p={1-pct:.3f})")

    mid = anchors[len(anchors) // 2]
    a1 = [d for d in anchors if d < mid]; a2 = [d for d in anchors if d >= mid]
    _, _, i1, _, base1, _ = regime_daysets(trading_days, a1, a.pre, a.impact, a.post)
    _, _, i2, _, base2, _ = regime_daysets(trading_days, a2, a.pre, a.impact, a.post)
    d1 = mean_offdiag(ar, basket, i1)[0] - mean_offdiag(ar, basket, base1)[0]
    d2 = mean_offdiag(ar, basket, i2)[0] - mean_offdiag(ar, basket, base2)[0]
    print(f"  OOS split-half impact-Δ: H1 {d1:+.3f}  H2 {d2:+.3f}  ({'holds' if d1>0 and d2>0 else 'does NOT hold'})")

    out = {"anchor": a.anchor, "backend": backend, "basket": basket, "sectors_spanned": cross,
           "dominant_sector": dom_sec, "embedding_impact_delta": emb["impact_delta"],
           "placebo_percentile": pct, "oos": [d1, d2]}
    (G / f"event_embedding_risk_{a.anchor}.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {G / f'event_embedding_risk_{a.anchor}.json'}")
    print("  READ: embedding basket impact-Δ > 0, spans multiple sectors, beats placebo (p<0.05) AND "
          "beats the GICS same-sector basket AND holds OOS  =>  the embedding captured cross-sector risk "
          "co-exposure that sector-neutralisation missed. Otherwise it just recovered a sector (null).")


if __name__ == "__main__":
    main()
