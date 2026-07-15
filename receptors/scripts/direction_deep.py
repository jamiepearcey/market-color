# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
direction_deep.py — stress-test the "direction has nonlinear headroom" finding
and ask WHAT the asymmetric signal actually IS.

Reuses probe_temporal.py's EXACT cause->effect pair miner (a->b if
a.entities ∩ b.cause_entities, both epochs>0, epoch_a<epoch_b, shared entity
specific i.e. df<=ceil(0.03*n)). Temporal split by effect-doc epoch.

1. MODELS on full [E_a,E_b,D,P]: LogReg vs MLP(256,64) vs HistGB (acc+AUC).
2. FEATURE ABLATION: linear + GB on {E_a},{E_b},{D},{P},{D,P},{all} -> AUC.
   Does the asymmetric signal live in the interaction (D,P) or is it a
   marginal topic-prior (E_a/E_b = "which topics tend to be causes")?
3. RECENCY/ARTIFACT CONTROL: co-topical pairs sharing >=1 specific entity but
   with NO causal link; label by which doc is EARLIER; predict time-order from
   [e_a,e_b,D,P]. High control AUC => "direction" partly = recency/topical drift,
   not causal semantics. Subtract to estimate genuine causal component.
"""
import json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(0)


def norml(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


# ---- shared load + entity index (mirrors probe_temporal.py) -----------------
def load():
    emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
    docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
    n = len(docs)
    df = defaultdict(int)
    for d in docs:
        for e in set(d["entities"]):
            df[e] += 1
    cap = math.ceil(0.03 * n)
    post = defaultdict(list)
    for i, d in enumerate(docs):
        for e in d["entities"]:
            post[e].append(i)
    return emb, docs, df, cap, post, n


def mine_causal_pairs(docs, df, cap, post):
    """EXACT miner from probe_temporal.py."""
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0:
            continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    if a != b and docs[a]["published_epoch"] > 0 and 0 < tb - docs[a]["published_epoch"]:
                        pos.append((a, b))
    pos = list({p for p in pos})
    RNG.shuffle(pos)
    return pos[:9000]


def mine_control_pairs(docs, df, cap, post, causal_set, n_target):
    """
    Co-topical CONTROL: pairs sharing >=1 SPECIFIC entity (df<=cap) but with NO
    causal (cause_entities) link between them, both epochs>0, distinct times.
    Ordered so first element is EARLIER (label = time-order, not causality).
    """
    ctrl = set()
    causal = set(causal_set) | {(b, a) for a, b in causal_set}
    specific_ents = [e for e, c in df.items() if 0 < c <= cap]
    rng = np.random.default_rng(7)
    rng.shuffle(specific_ents)
    for e in specific_ents:
        members = [i for i in post.get(e, []) if docs[i]["published_epoch"] > 0]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        # sample a bounded number of pairs per entity to keep it diverse
        for k in range(min(len(members), 30)):
            i, j = members[k], members[(k + 1) % len(members)]
            if i == j:
                continue
            ti, tj = docs[i]["published_epoch"], docs[j]["published_epoch"]
            if ti == tj:
                continue
            a, b = (i, j) if ti < tj else (j, i)  # a earlier than b
            if (a, b) in causal or (b, a) in causal:
                continue
            # exclude any latent causal link either direction (cause_entities overlap)
            ea, eb = set(docs[a]["entities"]), set(docs[b]["entities"])
            ca, cb = set(docs[a].get("cause_entities", [])), set(docs[b].get("cause_entities", []))
            if (ea & cb) or (eb & ca):
                continue
            ctrl.add((a, b))
            if len(ctrl) >= n_target:
                break
        if len(ctrl) >= n_target:
            break
    return list(ctrl)


# ---- feature blocks ---------------------------------------------------------
def blocks(emb, a, b):
    ea, eb = emb[a], emb[b]
    return {"E_a": ea, "E_b": eb, "D": ea - eb, "P": ea * eb}


def build_matrix(emb, pairs, keys):
    rows = []
    for a, b in pairs:
        bl = blocks(emb, a, b)
        rows.append(np.concatenate([bl[k] for k in keys]))
    return np.asarray(rows, np.float32)


ALL_KEYS = ["E_a", "E_b", "D", "P"]


def ordering_dataset(emb, pairs, keys):
    """(a,b)->1, (b,a)->0. Returns X, y and per-row is-test mask maker later."""
    Xp = build_matrix(emb, pairs, keys)
    Xr = build_matrix(emb, [(b, a) for a, b in pairs], keys)
    X = np.vstack([Xp, Xr])
    y = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    return X, y


def temporal_mask(docs, pairs, order_key="effect"):
    """Test = latest 30% by effect-doc (b) epoch. Duplicated for the flipped rows."""
    ep = np.array([docs[b]["published_epoch"] for _, b in pairs])
    cut = np.quantile(ep, 0.7)
    te = ep >= cut
    return np.r_[te, te]


def fit_eval(Xtr, ytr, Xte, yte, model):
    if model == "linear":
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        score = clf.decision_function(Xte)
        pred = clf.predict(Xte)
    elif model == "gb":
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
    elif model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                            early_stopping=True, random_state=0)
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
    else:
        raise ValueError(model)
    return round(float(accuracy_score(yte, pred)), 3), round(float(roc_auc_score(yte, score)), 3)


def main():
    emb, docs, df, cap, post, n = load()
    causal = mine_causal_pairs(docs, df, cap, post)
    out = {"n_docs": n, "specific_cap_df": cap, "n_causal_pairs": len(causal)}

    # ---- 1. MODELS on full features, temporal split -------------------------
    Xfull, y = ordering_dataset(emb, causal, ALL_KEYS)
    m = temporal_mask(docs, causal)
    Xtr, ytr, Xte, yte = Xfull[~m], y[~m], Xfull[m], y[m]
    models = {}
    for mo in ["linear", "mlp", "gb"]:
        acc, auc = fit_eval(Xtr, ytr, Xte, yte, mo)
        models[mo] = {"acc": acc, "auc": auc}
    out["n_test_rows"] = int(m.sum())
    out["models_temporal_full"] = models
    lin_auc = models["linear"]["auc"]
    gb_auc = models["gb"]["auc"]
    mlp_auc = models["mlp"]["auc"]
    out["headroom_gb_minus_linear_auc"] = round(gb_auc - lin_auc, 3)
    out["headroom_mlp_minus_linear_auc"] = round(mlp_auc - lin_auc, 3)

    # ---- 2. FEATURE ABLATION (linear + gb), temporal split ------------------
    subsets = {
        "E_a": ["E_a"], "E_b": ["E_b"], "D": ["D"], "P": ["P"],
        "D,P": ["D", "P"], "all": ALL_KEYS,
    }
    ablation = {}
    for name, keys in subsets.items():
        Xs, ys = ordering_dataset(emb, causal, keys)
        la = fit_eval(Xs[~m], ys[~m], Xs[m], ys[m], "linear")[1]
        ga = fit_eval(Xs[~m], ys[~m], Xs[m], ys[m], "gb")[1]
        ablation[name] = {"linear_auc": la, "gb_auc": ga}
    out["feature_ablation"] = ablation

    # ---- 3. RECENCY/ARTIFACT CONTROL ---------------------------------------
    control = mine_control_pairs(docs, df, cap, post, causal, len(causal))
    out["n_control_pairs"] = len(control)
    Xc, yc = ordering_dataset(emb, control, ALL_KEYS)
    mc = temporal_mask(docs, control)
    ctrl_res = {}
    for mo in ["linear", "gb"]:
        acc, auc = fit_eval(Xc[~mc], yc[~mc], Xc[mc], yc[mc], mo)
        ctrl_res[mo] = {"acc": acc, "auc": auc}
    out["control_temporal_full"] = ctrl_res
    ctrl_gb_auc = ctrl_res["gb"]["auc"]
    ctrl_lin_auc = ctrl_res["linear"]["auc"]
    # genuine causal component = direction AUC over 0.5, minus recency-explainable part
    out["direction_gb_auc"] = gb_auc
    out["recency_control_gb_auc"] = ctrl_gb_auc
    out["genuine_causal_component_gb"] = round((gb_auc - 0.5) - (ctrl_gb_auc - 0.5), 3)
    out["genuine_causal_component_linear"] = round((lin_auc - 0.5) - (ctrl_lin_auc - 0.5), 3)

    # ---- interpretation of ablation ----------------------------------------
    interp_auc = {k: v["gb_auc"] for k, v in ablation.items()}
    dp = interp_auc["D,P"]
    marg = max(interp_auc["E_a"], interp_auc["E_b"])
    signal_is = "interaction (D,P)" if (dp - 0.5) >= (marg - 0.5) else "marginal topic-prior (E_a/E_b)"
    out["ablation_verdict"] = {
        "DP_gb_auc": dp, "best_marginal_gb_auc": round(marg, 3), "signal_is": signal_is,
    }

    # ---- VERDICT ------------------------------------------------------------
    genuine = out["genuine_causal_component_gb"]
    # non-stationary check: temporal full AUC must be meaningfully > 0.5
    if gb_auc < 0.55 and lin_auc < 0.55:
        verdict = "NON-STATIONARY"
    elif genuine >= 0.10 and gb_auc >= 0.65:
        verdict = "SIGNAL"
    elif genuine >= 0.03 and gb_auc >= 0.55:
        verdict = "WEAK"
    else:
        verdict = "NO-SIGNAL"
    out["verdict"] = verdict

    # ---- persist ------------------------------------------------------------
    (DATA / "metrics").mkdir(parents=True, exist_ok=True)
    (DATA / "figures").mkdir(parents=True, exist_ok=True)
    (DATA / "metrics" / "direction_deep.json").write_text(json.dumps(out, indent=2))

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    mkeys = ["linear", "mlp", "gb"]
    ax[0].bar(mkeys, [models[k]["auc"] for k in mkeys],
              color=["#888", "#4c78a8", "#e45756"])
    ax[0].axhline(0.5, ls="--", c="k", lw=0.8)
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Model test AUC (temporal, full feats)")
    for i, k in enumerate(mkeys):
        ax[0].text(i, models[k]["auc"] + 0.005, f'{models[k]["auc"]:.3f}', ha="center")

    anames = list(subsets.keys())
    xpos = np.arange(len(anames))
    w = 0.4
    ax[1].bar(xpos - w / 2, [ablation[a]["linear_auc"] for a in anames], w, label="linear", color="#888")
    ax[1].bar(xpos + w / 2, [ablation[a]["gb_auc"] for a in anames], w, label="GB", color="#e45756")
    ax[1].axhline(0.5, ls="--", c="k", lw=0.8)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(anames, rotation=30, ha="right")
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("Feature-ablation AUC")
    ax[1].legend()

    ax[2].bar(["direction\n(causal)", "recency\ncontrol"],
              [gb_auc, ctrl_gb_auc], color=["#e45756", "#54a24b"])
    ax[2].axhline(0.5, ls="--", c="k", lw=0.8)
    ax[2].set_ylim(0.4, 1.0)
    ax[2].set_title(f"Direction vs recency-control (GB)\ngenuine causal Δ={genuine:+.3f}")
    for i, v in enumerate([gb_auc, ctrl_gb_auc]):
        ax[2].text(i, v + 0.005, f"{v:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(DATA / "figures" / "direction_deep.png", dpi=110)

    # ---- summary (<=4 lines) -----------------------------------------------
    print(f"Direction (temporal, full): linear AUC {lin_auc} | MLP {mlp_auc} | GB {gb_auc} "
          f"(headroom GB-lin {out['headroom_gb_minus_linear_auc']:+.3f})")
    print(f"Ablation: signal lives in {signal_is} "
          f"(D,P GB AUC {dp} vs best marginal {round(marg,3)}; D {interp_auc['D']} P {interp_auc['P']})")
    print(f"Recency control GB AUC {ctrl_gb_auc} -> genuine causal component {genuine:+.3f} "
          f"(n_causal {len(causal)}, n_control {len(control)})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
