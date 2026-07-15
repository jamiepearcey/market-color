# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Probing-ladder + mutual-information diagnostic.

The receptors so far are LINEAR maps over a FROZEN MiniLM space. This asks, per
task: how much of the label is (a) linearly decodable, (b) NON-linearly decodable
(MLP / gradient boosting), and (c) present at all model-free (mutual information)?

Reading:
  * MLP >> linear            -> nonlinear HEADROOM: a better representation / model
                                would help (invest in fine-tuning / GNN / SAE).
  * MLP ~= linear >> chance  -> linear-SATURATED: signal is there and already
                                captured; more model won't help, need more data/signal.
  * all ~= chance & MI ~= 0  -> NO SIGNAL in this embedding: the negative is real,
                                the signal isn't in the text representation at all.

Tasks: direction (pair ordering), polarity, predicate (11-way), impact (|zscore|
regression), corroboration. Everything held-out; deterministic.
"""
import json, math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MET = DATA / "metrics"; MET.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

def load_facts():
    emb = np.load(DATA / "facts.npy").astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    facts = [json.loads(l) for l in (DATA / "facts.jsonl").read_text().splitlines() if l.strip()]
    return emb, facts

def load_docs():
    emb = np.load(DATA / "embeddings.npy").astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
    return emb, docs

def hash_split(n, frac=0.3, seed=1):
    r = np.random.default_rng(seed).random(n)
    return r >= frac, r < frac  # train, test masks

def mi_class(X, y, k=50):
    Xp = PCA(n_components=min(k, X.shape[1]), random_state=0).fit_transform(X)
    return float(mutual_info_classif(Xp, y, random_state=0).sum())

def mi_reg(X, y, k=50):
    Xp = PCA(n_components=min(k, X.shape[1]), random_state=0).fit_transform(X)
    return float(mutual_info_regression(Xp, y, random_state=0).sum())

RESULTS = []

def verdict(linear, best_nl, chance, tol=0.02):
    if best_nl < chance + 0.02 and linear < chance + 0.02:
        return "NO-SIGNAL"
    if best_nl > linear + tol:
        return "NONLINEAR-HEADROOM"
    return "LINEAR-SATURATED"

# ---------- classification task runner ----------
def run_class(name, X, y, chance, multiclass=False):
    tr, te = hash_split(len(y))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    out = {"task": name, "n": int(len(y)), "n_test": int(te.sum()), "chance": round(chance, 3)}
    # linear
    lin = LogisticRegression(max_iter=2000, C=1.0)
    lin.fit(Xtr, ytr); pl = lin.predict(Xte)
    out["linear_acc"] = round(accuracy_score(yte, pl), 3)
    out["linear_bal"] = round(balanced_accuracy_score(yte, pl), 3)
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True,
                        max_iter=200, random_state=0)
    mlp.fit(Xtr, ytr); pm = mlp.predict(Xte)
    out["mlp_acc"] = round(accuracy_score(yte, pm), 3)
    out["mlp_bal"] = round(balanced_accuracy_score(yte, pm), 3)
    # gradient boosting
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    gb.fit(Xtr, ytr); pg = gb.predict(Xte)
    out["gb_acc"] = round(accuracy_score(yte, pg), 3)
    # AUC for binary
    if not multiclass:
        try:
            out["linear_auc"] = round(roc_auc_score(yte, lin.decision_function(Xte)), 3)
            out["mlp_auc"] = round(roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1]), 3)
        except Exception:
            pass
    out["mi_nats"] = round(mi_class(X, y), 3)
    best_nl = max(out["mlp_acc"], out["gb_acc"])
    out["verdict"] = verdict(out["linear_acc"], best_nl, chance)
    RESULTS.append(out)
    print(f"\n[{name}]  n={out['n']} chance={chance:.3f}")
    print(f"  linear acc {out['linear_acc']}  MLP acc {out['mlp_acc']}  GB acc {out['gb_acc']}  MI~{out['mi_nats']}  => {out['verdict']}")
    return out

def run_reg(name, X, y, baseline=None):
    tr, te = hash_split(len(y))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    out = {"task": name, "n": int(len(y)), "n_test": int(te.sum())}
    lin = Ridge(alpha=1.0).fit(Xtr, ytr)
    out["linear_spearman"] = round(float(spearmanr(lin.predict(Xte), yte).correlation), 3)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 64), early_stopping=True,
                       max_iter=300, random_state=0).fit(Xtr, ytr)
    out["mlp_spearman"] = round(float(spearmanr(mlp.predict(Xte), yte).correlation), 3)
    gb = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(Xtr, ytr)
    out["gb_spearman"] = round(float(spearmanr(gb.predict(Xte), yte).correlation), 3)
    if baseline is not None:
        out["baseline_spearman"] = round(float(spearmanr(baseline[te], yte).correlation), 3)
    out["mi_nats"] = round(mi_reg(X, y), 3)
    best = max(out["mlp_spearman"], out["gb_spearman"], out["linear_spearman"])
    out["verdict"] = ("NO-SIGNAL" if best < 0.05
                      else "NONLINEAR-HEADROOM" if best > out["linear_spearman"] + 0.03
                      else "LINEAR-SATURATED")
    RESULTS.append(out)
    print(f"\n[{name}]  n={out['n']}")
    print(f"  linear ρ {out['linear_spearman']}  MLP ρ {out['mlp_spearman']}  GB ρ {out['gb_spearman']}"
          + (f"  baseline ρ {out['baseline_spearman']}" if baseline is not None else "")
          + f"  MI~{out['mi_nats']}  => {out['verdict']}")
    return out

# ================= build tasks =================
def task_polarity(emb, facts):
    idx = [i for i, f in enumerate(facts) if abs(f.get("direction", 0)) > 0.5]
    X = emb[idx]; y = np.array([1 if facts[i]["direction"] > 0 else 0 for i in idx])
    chance = max(y.mean(), 1 - y.mean())
    run_class("polarity (bull/bear)", X, y, chance)

def task_predicate(emb, facts):
    labs = sorted({f["predicate"] for f in facts})
    keep = [p for p in labs if sum(f["predicate"] == p for f in facts) >= 150]
    idx = [i for i, f in enumerate(facts) if f["predicate"] in keep]
    lab2i = {p: k for k, p in enumerate(keep)}
    X = emb[idx]; y = np.array([lab2i[facts[i]["predicate"]] for i in idx])
    counts = np.bincount(y); chance = counts.max() / counts.sum()
    run_class(f"predicate ({len(keep)}-way)", X, y, chance, multiclass=True)

def task_impact(emb, facts):
    pp = {}
    for l in (DATA / "price_pairs.jsonl").read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l); m = abs(r["zscore"])
        pp[r["doc_id"]] = max(pp.get(r["doc_id"], 0.0), m)
    idx = [i for i, f in enumerate(facts) if f["doc_id"] in pp]
    X = emb[idx]; y = np.array([pp[facts[i]["doc_id"]] for i in idx], float)
    conf = np.array([facts[i].get("confidence", 0.0) for i in idx], float)
    run_reg("impact (|zscore|)", X, y, baseline=conf)

def task_corroboration(emb, facts):
    # entity inverted index, corrob = # same-sign facts within 3d sharing an entity
    from collections import defaultdict
    inv = defaultdict(list)
    for i, f in enumerate(facts):
        for e in f.get("entities", []):
            inv[e].append(i)
    ep = np.array([f["published_epoch"] for f in facts])
    sgn = np.array([np.sign(f.get("direction", 0)) for f in facts])
    corrob = np.zeros(len(facts), int)
    W = 3 * 86400
    for i, f in enumerate(facts):
        if ep[i] == 0: continue
        cand = set()
        for e in f.get("entities", []):
            cand.update(inv[e])
        c = 0
        for j in cand:
            if j != i and ep[j] > 0 and abs(ep[i] - ep[j]) <= W and sgn[j] == sgn[i]:
                c += 1
        corrob[i] = c
    idx = [i for i in range(len(facts)) if ep[i] > 0]
    X = emb[idx]; y = np.array([1 if corrob[i] >= 2 else 0 for i in idx])
    chance = max(y.mean(), 1 - y.mean())
    run_class("corroboration (>=2)", X, y, chance)

def task_direction(demb, docs):
    # mine directed pairs a->b (a.entities ∩ b.cause_entities, epoch_a<epoch_b, specific)
    from collections import defaultdict
    n = len(docs)
    df = defaultdict(int)
    for d in docs:
        for e in set(d["entities"]): df[e] += 1
    cap = math.ceil(0.03 * n)
    post = defaultdict(list)
    for i, d in enumerate(docs):
        for e in d["entities"]: post[e].append(i)
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0: continue
        for c in d.get("cause_entities", []):
            if df.get(c, 0) == 0 or df[c] > cap: continue
            for a in post.get(c, []):
                if a != b and 0 < tb - docs[a]["published_epoch"]:
                    pos.append((a, b))
    # dedup + sample
    pos = list({p for p in pos})
    RNG.shuffle(pos)
    pos = pos[:8000]
    # build ordering-classification set: (a,b)->1, (b,a)->0
    def feat(a, b):
        ea, eb = demb[a], demb[b]
        return np.concatenate([ea, eb, ea - eb, ea * eb])
    X = np.array([feat(a, b) for a, b in pos] + [feat(b, a) for a, b in pos], np.float32)
    y = np.array([1] * len(pos) + [0] * len(pos))
    print(f"\n  (direction: mined {len(pos)} positive pairs)")
    run_class("direction (pair ordering)", X, y, 0.5)

if __name__ == "__main__":
    print("=== probing-ladder + MI diagnostic ===")
    emb, facts = load_facts()
    task_polarity(emb, facts)
    task_predicate(emb, facts)
    task_impact(emb, facts)
    task_corroboration(emb, facts)
    demb, docs = load_docs()
    task_direction(demb, docs)
    (MET / "probe_ladder.json").write_text(json.dumps(RESULTS, indent=2))
    print("\n=== SUMMARY ===")
    print(f"  {'task':<26} {'linear':>8} {'nonlin':>8} {'MI':>6}  verdict")
    for r in RESULTS:
        lin = r.get("linear_acc", r.get("linear_spearman"))
        nl = max([r[k] for k in ("mlp_acc","gb_acc","mlp_spearman","gb_spearman") if k in r])
        print(f"  {r['task']:<26} {lin:>8} {nl:>8} {r['mi_nats']:>6}  {r['verdict']}")
    print("\nwrote data/metrics/probe_ladder.json")
