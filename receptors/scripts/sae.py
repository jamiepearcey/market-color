# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","torch","scikit-learn","matplotlib"]
# ///
"""
SAE concept-atlas experiment for the receptors fact corpus.

Question: is a LEARNED sparse dictionary a better "concept atlas" than the
6 hand-picked receptor axes? Train a sparse autoencoder on facts.npy and test
(a) linear decodability of polarity/predicate from sparse codes vs raw dims and
(b) whether interpretable monosemantic concept-neurons emerge per free label.

CPU-only, small, <3 min. Emits data/metrics/sae.json + data/figures/sae.png.
"""
import json, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
torch.manual_seed(0); np.random.seed(0)

# ---------------------------------------------------------------- load + norm
X = np.load(os.path.join(DATA, "facts.npy")).astype(np.float32)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)   # L2-normalize rows
facts = [json.loads(l) for l in open(os.path.join(DATA, "facts.jsonl"))]
N, D = X.shape
assert len(facts) == N, (len(facts), N)

claims = [f.get("claim", "") for f in facts]
polarity = np.array([int(np.sign(f.get("direction", 0.0))) for f in facts])  # -1/0/+1
predicate = np.array([f.get("predicate", "other") for f in facts])
desk = np.array([f.get("desk", "") for f in facts])

# temporal split if epochs present, else random 80/20
epochs = np.array([f.get("published_epoch", 0) or 0 for f in facts])
if np.unique(epochs).size > 5:
    order = np.argsort(epochs); cut = int(0.8 * N)
    tr_idx, te_idx = order[:cut], order[cut:]
    split = "temporal"
else:
    perm = np.random.permutation(N); cut = int(0.8 * N)
    tr_idx, te_idx = perm[:cut], perm[cut:]
    split = "random"

Xtr = torch.from_numpy(X[tr_idx]); Xte = torch.from_numpy(X[te_idx])

# ---------------------------------------------------------------- SAE model
M = 1024
class SAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.enc = nn.Linear(d, m, bias=True)
        self.dec = nn.Linear(m, d, bias=False)  # free decoder
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.t())  # init tied
    def forward(self, x):
        c = torch.relu(self.enc(x - self.b_pre))
        xh = self.dec(c) + self.b_pre
        return xh, c
    def unit_dec(self):
        with torch.no_grad():
            w = self.dec.weight  # (d, m)
            self.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))

def train_sae(lam, epochs_n=120):
    torch.manual_seed(0)
    model = SAE(D, M)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = Xtr.shape[0]; bs = 512
    for ep in range(epochs_n):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = Xtr[perm[i:i+bs]]
            xh, c = model(b)
            loss = ((xh - b) ** 2).sum(1).mean() + lam * c.abs().sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            model.unit_dec()
    return model

def eval_recon(model):
    model.eval()
    with torch.no_grad():
        xh, c = model(Xte)
        resid = ((xh - Xte) ** 2).sum().item()
        tot = ((Xte - Xte.mean(0)) ** 2).sum().item()
        r2 = 1.0 - resid / tot
        l0 = (c > 1e-6).float().sum(1).mean().item()
    return r2, l0

# ---------------------------------------------------------------- lambda sweep
t0 = time.time()
lams = [1e-1, 3e-1, 1e0]
sweep = []
models = {}
for lam in lams:
    m = train_sae(lam)
    r2, l0 = eval_recon(m)
    sweep.append({"lambda": lam, "r2": round(r2, 4), "l0": round(l0, 2)})
    models[lam] = m
    print(f"  lam={lam:.0e}  R2={r2:.3f}  L0={l0:.1f}")

# pick lambda with L0 in [10,40] closest to 25, else closest to that band
def score(s):
    l0 = s["l0"]
    if 10 <= l0 <= 40: return abs(l0 - 25)
    return 100 + min(abs(l0 - 10), abs(l0 - 40))
best = min(sweep, key=score)
lam = best["lambda"]; model = models[lam]
print(f"  chosen lambda={lam:.0e} (L0={best['l0']})")

# full codes at chosen lambda
model.eval()
with torch.no_grad():
    _, C_all = model(torch.from_numpy(X))
    C_all = C_all.numpy()
Ctr, Cte = C_all[tr_idx], C_all[te_idx]

# ---------------------------------------------------------------- decodability
def decode(feat_tr, feat_te, y, labtr=tr_idx, labte=te_idx, mask_val=None):
    ytr, yte = y[tr_idx], y[te_idx]
    if mask_val is not None:  # drop rows equal to mask (e.g. neutral polarity 0)
        mtr = ytr != mask_val; mte = yte != mask_val
        feat_tr, ytr = feat_tr[mtr], ytr[mtr]
        feat_te, yte = feat_te[mte], yte[mte]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feat_tr, ytr)
    return accuracy_score(yte, clf.predict(feat_te))

decod = {}
# polarity: sign, drop neutral 0 -> binary +/-
decod["polarity"] = {
    "raw": round(decode(X[tr_idx], X[te_idx], polarity, mask_val=0), 4),
    "sae": round(decode(Ctr, Cte, polarity, mask_val=0), 4),
}
# predicate: top classes (all 12, multiclass)
decod["predicate"] = {
    "raw": round(decode(X[tr_idx], X[te_idx], predicate), 4),
    "sae": round(decode(Ctr, Cte, predicate), 4),
}
# desk multiclass
decod["desk"] = {
    "raw": round(decode(X[tr_idx], X[te_idx], desk), 4),
    "sae": round(decode(Ctr, Cte, desk), 4),
}
for k, v in decod.items():
    print(f"  decode {k}: raw={v['raw']:.3f} sae={v['sae']:.3f} (dl={v['sae']-v['raw']:+.3f})")

# ---------------------------------------------------------------- concept discovery
# For each label group, find SAE feature with highest mean-activation difference
# (in-group mean minus out-group mean). Report alignment + coverage + examples.
def top_pred_classes(n=6):
    return [p for p, _ in Counter(predicate).most_common(n)]
def top_desks(n=4):
    return [d for d, _ in Counter(desk).most_common(n)]

groups = {}
groups["polarity+"] = polarity == 1
groups["polarity-"] = polarity == -1
for p in top_pred_classes(6):
    groups[f"pred:{p}"] = predicate == p
for d in top_desks(4):
    groups[f"desk:{d}"] = desk == d

active = C_all > 1e-6
concepts = []
for name, mask in groups.items():
    inm = C_all[mask].mean(0)
    outm = C_all[~mask].mean(0)
    diff = inm - outm
    fi = int(np.argmax(diff))
    act_idx = np.where(active[:, fi])[0]
    # fraction of activating facts that belong to the group = "purity"
    if act_idx.size:
        purity = float(mask[act_idx].mean())
    else:
        purity = 0.0
    n_act = int(active[:, fi].sum())
    # 2 example claims maximally activating this feature
    top_ex = act_idx[np.argsort(-C_all[act_idx, fi])[:2]] if act_idx.size else []
    examples = [claims[i][:110] for i in top_ex]
    concepts.append({
        "label": name, "feature": fi,
        "act_diff": round(float(diff[fi]), 4),
        "n_facts_active": n_act,
        "purity_to_label": round(purity, 3),
        "examples": examples,
    })

concepts.sort(key=lambda c: -c["act_diff"])
top_concepts = concepts[:10]
mean_purity = float(np.mean([c["purity_to_label"] for c in top_concepts]))
base_rates = {c["label"]: round(float(groups[c["label"]].mean()), 3) for c in top_concepts}
# interpretable if discovered neurons are enriched well above base rate
lift = np.mean([top_concepts[i]["purity_to_label"] / max(base_rates[top_concepts[i]["label"]], 1e-6)
                for i in range(len(top_concepts))])

print(f"  discovered {len(top_concepts)} concept-neurons; mean purity={mean_purity:.2f}, mean lift x{lift:.1f}")

# ---------------------------------------------------------------- verdict
# decodability retained if SAE within ~2pts of raw on both polarity & predicate
dl_pol = decod["polarity"]["sae"] - decod["polarity"]["raw"]
dl_pred = decod["predicate"]["sae"] - decod["predicate"]["raw"]
retained = (dl_pol >= -0.02) and (dl_pred >= -0.02)
interpretable = (mean_purity >= 0.5) and (lift >= 1.5)
recon_ok = best["r2"] >= 0.5 and 10 <= best["l0"] <= 40

if retained and interpretable and recon_ok:
    verdict = "SIGNAL"
elif (retained or interpretable) and recon_ok:
    verdict = "WEAK"
else:
    verdict = "NO-SIGNAL"

elapsed = round(time.time() - t0, 1)

# ---------------------------------------------------------------- emit json
os.makedirs(os.path.join(DATA, "metrics"), exist_ok=True)
os.makedirs(os.path.join(DATA, "figures"), exist_ok=True)
out = {
    "n_facts": N, "dim": D, "M": M, "split": split,
    "lambda_sweep": sweep, "chosen_lambda": lam,
    "recon_r2": best["r2"], "mean_l0": best["l0"],
    "decodability": decod,
    "discovered_concepts": top_concepts,
    "concept_base_rates": base_rates,
    "mean_concept_purity": round(mean_purity, 3),
    "mean_concept_lift": round(float(lift), 2),
    "verdict": verdict, "elapsed_s": elapsed,
}
with open(os.path.join(DATA, "metrics", "sae.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
l0s = [s["l0"] for s in sweep]; r2s = [s["r2"] for s in sweep]
ax1.plot(l0s, r2s, "o-", color="steelblue")
for s in sweep:
    mk = "*" if s["lambda"] == lam else "o"
    ax1.scatter([s["l0"]], [s["r2"]], s=180 if mk == "*" else 40,
                color="crimson" if mk == "*" else "steelblue", zorder=5)
    ax1.annotate(f"λ={s['lambda']:.0e}", (s["l0"], s["r2"]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8)
ax1.axvspan(10, 40, color="green", alpha=0.08)
ax1.set_xlabel("mean L0 (active features / fact)")
ax1.set_ylabel("held-out reconstruction R²")
ax1.set_title(f"Recon–sparsity tradeoff (chosen λ={lam:.0e}, R²={best['r2']:.2f})")

labels = [c["label"] for c in top_concepts]
purities = [c["purity_to_label"] for c in top_concepts]
brs = [base_rates[c["label"]] for c in top_concepts]
yp = np.arange(len(labels))
ax2.barh(yp, purities, color="darkorange", label="concept-neuron purity")
ax2.barh(yp, brs, color="gray", alpha=0.5, height=0.45, label="label base rate")
ax2.set_yticks(yp); ax2.set_yticklabels(labels, fontsize=8); ax2.invert_yaxis()
ax2.set_xlabel("fraction of activating facts in label")
ax2.set_title("Best concept-feature alignment per label")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DATA, "figures", "sae.png"), dpi=110)

# ---------------------------------------------------------------- summary
print(f"SAE atlas | split={split} λ={lam:.0e} recon R²={best['r2']:.2f} mean L0={best['l0']:.0f} ({elapsed}s)")
print(f"Decodability SAE vs raw | polarity {decod['polarity']['sae']:.2f}/{decod['polarity']['raw']:.2f} "
      f"predicate {decod['predicate']['sae']:.2f}/{decod['predicate']['raw']:.2f} desk "
      f"{decod['desk']['sae']:.2f}/{decod['desk']['raw']:.2f}")
print(f"Concept-neurons | {len(top_concepts)} found, mean purity={mean_purity:.2f} lift x{lift:.1f} "
      f"({'interpretable' if interpretable else 'weak/mixed'})")
print(f"VERDICT: {verdict}")
