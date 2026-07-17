# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy", "fastembed>=0.3"]
# ///
"""
Feed the receptor W with realised-CONFIRMED causal edges.

Tests the receptor post-mortem hypothesis: an asymmetric transport operator W
(ridge: W = (AᵀA+λI)⁻¹AᵀB, A=cause rows, B=effect rows) recovers cause→effect
DIRECTION better when trained on realised-CONFIRMED edges (effect actually moved in
the narrative direction, |z|≥τ) than on ALL (noisy) edges.

Representation: each entity -> MiniLM embedding of its canonical name (L2-normed).
Direction accuracy = share of held-out {cause,effect} where cos(cause·W,effect) >
cos(effect·W,cause). Split leak-free by effect-entity hash. Same test set for every W.

Usage: uv run eventgraph/scripts/receptor_feed.py --graph-dir /tmp/eg_6k
"""
import argparse, json, time, math, collections, hashlib, datetime as dt, re
from pathlib import Path
import httpx, numpy as np

FORDER = ["market", "rates", "credit", "oil", "usd", "gold", "em"]
FP = {"market": "SPY", "rates": "IEF", "credit": "HYG", "oil": "USO", "usd": "UUP", "gold": "GLD", "em": "EEM"}
PROXIES = set(FP.values()); DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def yahoo(sym, cache, p1=1230768000, p2=1420070400):
    p = cache / f"{sym}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25); txt = r.text if r.status_code == 200 else ""
        except Exception: txt = ""
        p.write_text(txt); time.sleep(0.12)
    out = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], cl):
            if c is not None: out[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception: pass
    return out

def logret(cl):
    ds = sorted(cl); return {ds[i]: math.log(cl[ds[i]] / cl[ds[i - 1]]) for i in range(1, len(ds)) if cl[ds[i - 1]] > 0 and cl[ds[i]] > 0}

def ridge(A, B, w, lam):
    Aw = A * w[:, None]
    return np.linalg.solve(A.T @ Aw + lam * np.eye(A.shape[1]), Aw.T @ B)

def dir_acc(W, pairs):  # pairs: list of (cause_vec, effect_vec)
    c = sum(1 for a, b in pairs if (a @ W) @ b > (b @ W) @ a)
    return c / max(len(pairs), 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)

    epat = re.compile(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','((?:[^']|'')*)','([^']*)'")
    name = {m.group(1): m.group(2).replace("''", "'") for m in (epat.search(l) for l in open(gd / "pg_upsert.sql")) if m}
    syms = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            syms[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    edges = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); c, e = j.get("cause_entity"), j.get("effect_entity")
        if not c or not e or c == e or c not in name or e not in name or j.get("effect_dir") not in DIR: continue
        edges.append({"c": c, "e": e, "date": (docdate.get(j.get("doc_id")) or "")[:10], "dir": j["effect_dir"],
                      "modality": j.get("modality") or "happened"})

    # realised confirmation for edges whose effect is priced
    need = collections.Counter(syms[x["e"]] for x in edges if x["e"] in syms)
    fetch = list(dict.fromkeys(list(FP.values()) + [s for s, _ in need.most_common(400)]))
    print(f"prices: {len(fetch)} symbols (cached) ...")
    px = {s: logret(yahoo(s, cache)) for s in fetch}; ok = {s for s, r in px.items() if len(r) > 300}
    fdates = sorted(set.intersection(*[set(px[FP[f]]) for f in FORDER]))
    F = np.array([[px[FP[f]][d] for f in FORDER] for d in fdates]); G = np.zeros_like(F)
    for j in range(len(FORDER)):
        if j == 0: G[:, 0] = F[:, 0]
        else:
            X = np.column_stack([np.ones(len(fdates)), G[:, :j]]); b, *_ = np.linalg.lstsq(X, F[:, j], rcond=None); G[:, j] = F[:, j] - X @ b
    Gmap = {d: G[i] for i, d in enumerate(fdates)}
    def zscore(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if date not in r:
            nx = [d for d in ds if d >= date]; date = nx[0] if nx else None
        if not date or date not in r: return None
        i = ds.index(date)
        if i < 170: return None
        est = [d for d in ds[i - 165:i - 11] if d in Gmap]
        if len(est) < 90 or date not in Gmap: return None
        y = np.array([r[d] for d in est]); Gc = np.array([Gmap[d] for d in est])
        Xm = np.column_stack([np.ones(len(est)), Gc]); beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        resid = y - Xm @ beta; rstd = resid.std()
        pre = [ds[k] for k in range(i - 5, i + 1) if ds[k] in Gmap]
        if rstd == 0 or not pre: return None
        return sum(r[d] - (beta[0] + beta[1:] @ Gmap[d]) for d in pre) / (rstd * math.sqrt(len(pre)))
    nconf = 0
    for x in edges:
        z = zscore(syms[x["e"]], x["date"]) if (x["e"] in syms and syms[x["e"]] in ok and syms[x["e"]] not in PROXIES) else None
        x["z"] = z
        x["conf"] = (z is not None and abs(z) >= a.tau and (z > 0) == (DIR[x["dir"]] > 0))
        x["priced"] = z is not None; nconf += x["conf"]
    print(f"edges {len(edges)}  ·  priced {sum(x['priced'] for x in edges)}  ·  realised-CONFIRMED {nconf}\n")

    # embed distinct entity names once
    ents = sorted({x["c"] for x in edges} | {x["e"] for x in edges})
    print(f"embedding {len(ents)} entity names (MiniLM) ...")
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    V = np.array(list(model.embed([name[e] for e in ents])), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9); emb = {e: V[i] for i, e in enumerate(ents)}

    # leak-free split by effect hash
    def test_of(x): return int(hashlib.md5(x["e"].encode()).hexdigest(), 16) % 100 < 30
    tr = [x for x in edges if not test_of(x)]; te = [x for x in edges if test_of(x)]
    pairs = lambda xs: [(emb[x["c"]], emb[x["e"]]) for x in xs]
    A_all = np.array([emb[x["c"]] for x in tr]); B_all = np.array([emb[x["e"]] for x in tr])
    conf_tr = [x for x in tr if x["conf"]]
    test_pairs = pairs(te); test_conf = pairs([x for x in te if x["conf"]])

    W_all = ridge(A_all, B_all, np.ones(len(tr)), a.lam)
    W_conf = ridge(np.array([emb[x["c"]] for x in conf_tr]), np.array([emb[x["e"]] for x in conf_tr]), np.ones(len(conf_tr)), a.lam) if len(conf_tr) > 20 else None
    wz = np.array([min(abs(x["z"]), 4) if x["conf"] else (0.15 if x["priced"] else 0.4) for x in tr])
    W_wt = ridge(A_all, B_all, wz, a.lam)

    print("=== RECEPTOR W · DIRECTION ACCURACY (held-out) ===")
    print(f"  train pairs: all {len(tr)}  ·  confirmed {len(conf_tr)}   |   test pairs {len(test_pairs)} (confirmed {len(test_conf)})")
    print(f"  {'training set':28} {'acc @ all-test':>15} {'acc @ conf-test':>16}")
    print(f"  {'cosine (symmetric floor)':28} {0.5:>15.3f} {0.5:>16.3f}")
    print(f"  {'W · ALL edges':28} {dir_acc(W_all, test_pairs):>15.3f} {dir_acc(W_all, test_conf):>16.3f}")
    if W_conf is not None:
        print(f"  {'W · CONFIRMED only':28} {dir_acc(W_conf, test_pairs):>15.3f} {dir_acc(W_conf, test_conf):>16.3f}")
    print(f"  {'W · |z|-weighted':28} {dir_acc(W_wt, test_pairs):>15.3f} {dir_acc(W_wt, test_conf):>16.3f}")

if __name__ == "__main__":
    main()
