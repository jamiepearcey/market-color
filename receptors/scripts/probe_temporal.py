# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
The decisive follow-up: does the nonlinear HEADROOM (direction) and the impact
signal SURVIVE a temporal split, or is it random-split memorization?

If GB's advantage over linear persists train-early / test-late, a better
representation is genuinely worth building. If it collapses (like impact's random
0.35 -> temporal ~0), the "headroom" is non-stationary and won't generalize.

Runs direction (pair ordering) and impact under RANDOM vs TEMPORAL split,
linear vs gradient-boosting.
"""
import json, math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
RNG = np.random.default_rng(0)

def norml(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

def direction(split):
    emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
    docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
    from collections import defaultdict
    n = len(docs); df = defaultdict(int)
    for d in docs:
        for e in set(d["entities"]): df[e] += 1
    cap = math.ceil(0.03 * n); post = defaultdict(list)
    for i, d in enumerate(docs):
        for e in d["entities"]: post[e].append(i)
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0: continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    if a != b and docs[a]["published_epoch"] > 0 and 0 < tb - docs[a]["published_epoch"]:
                        pos.append((a, b))
    pos = list({p for p in pos}); RNG.shuffle(pos); pos = pos[:9000]
    eff_ep = np.array([docs[b]["published_epoch"] for _, b in pos])
    if split == "random":
        te = np.random.default_rng(1).random(len(pos)) < 0.3
    else:
        cut = np.quantile(eff_ep, 0.7); te = eff_ep >= cut
    def feat(a, b):
        ea, eb = emb[a], emb[b]
        return np.concatenate([ea, eb, ea - eb, ea * eb])
    Xp = np.array([feat(a, b) for a, b in pos], np.float32)
    Xr = np.array([feat(b, a) for a, b in pos], np.float32)
    X = np.vstack([Xp, Xr]); y = np.r_[np.ones(len(pos)), np.zeros(len(pos))]
    m = np.r_[te, te]
    lin = LogisticRegression(max_iter=2000).fit(X[~m], y[~m])
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[~m], y[~m])
    return {
        "split": split, "n_pairs": len(pos), "n_test": int(te.sum()),
        "linear_acc": round(accuracy_score(y[m], lin.predict(X[m])), 3),
        "gb_acc": round(accuracy_score(y[m], gb.predict(X[m])), 3),
        "linear_auc": round(roc_auc_score(y[m], lin.decision_function(X[m])), 3),
        "gb_auc": round(roc_auc_score(y[m], gb.predict_proba(X[m])[:, 1]), 3),
    }

def impact(split):
    emb = norml(np.load(DATA / "facts.npy").astype(np.float32))
    facts = [json.loads(l) for l in (DATA / "facts.jsonl").read_text().splitlines() if l.strip()]
    pp = {}
    for l in (DATA / "price_pairs.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); pp[r["doc_id"]] = max(pp.get(r["doc_id"], 0.0), abs(r["zscore"]))
    idx = [i for i, f in enumerate(facts) if f["doc_id"] in pp and f["published_epoch"] > 0]
    X = emb[idx]; y = np.array([pp[facts[i]["doc_id"]] for i in idx], float)
    ep = np.array([facts[i]["published_epoch"] for i in idx])
    if split == "random":
        te = np.random.default_rng(2).random(len(idx)) < 0.3
    else:
        te = ep >= np.quantile(ep, 0.7)
    lin = Ridge(alpha=1.0).fit(X[~te], y[~te])
    gb = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(X[~te], y[~te])
    return {
        "split": split, "n": len(idx), "n_test": int(te.sum()),
        "linear_rho": round(float(spearmanr(lin.predict(X[te]), y[te]).correlation), 3),
        "gb_rho": round(float(spearmanr(gb.predict(X[te]), y[te]).correlation), 3),
    }

if __name__ == "__main__":
    print("=== DIRECTION: random vs temporal (does nonlinear headroom generalize forward?) ===")
    dres = [direction("random"), direction("temporal")]
    for r in dres:
        print(f"  {r['split']:<9} n_test={r['n_test']:<6} linear_acc {r['linear_acc']} / AUC {r['linear_auc']}"
              f"   GB_acc {r['gb_acc']} / AUC {r['gb_auc']}   headroom(AUC) {r['gb_auc']-r['linear_auc']:+.3f}")
    print("\n=== IMPACT: random vs temporal (is the signal stationary?) ===")
    ires = [impact("random"), impact("temporal")]
    for r in ires:
        print(f"  {r['split']:<9} n_test={r['n_test']:<6} linear_rho {r['linear_rho']}   GB_rho {r['gb_rho']}")
    (DATA / "metrics" / "probe_temporal.json").write_text(json.dumps({"direction": dres, "impact": ires}, indent=2))
    print("\nwrote data/metrics/probe_temporal.json")
