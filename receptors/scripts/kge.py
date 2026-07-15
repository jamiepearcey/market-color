# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","torch","matplotlib"]
# ///
"""
KGE directionality probe.

Question: is the entity-level causal graph genuinely DIRECTIONAL/relational?
We build a causal knowledge graph of triples (cause_entity -> predicate -> entity)
from docs.jsonl and train three 128-d KGE models under a shared NSSA loss:
  - RotatE   (complex rotation, ASYMMETRIC)
  - DistMult (bilinear diagonal, SYMMETRIC baseline)
  - TransE   (translation, asymmetric-ish)
If RotatE (asymmetric) beats DistMult (symmetric) on filtered link prediction,
the causal graph carries genuine direction, not mere co-occurrence.
"""
import json, math, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.join(DATA, "metrics"), exist_ok=True)
os.makedirs(os.path.join(DATA, "figures"), exist_ok=True)

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

DIM = 128
MIN_ENT = 3      # keep entities appearing >= 3 times (as node in any triple)
MIN_REL = 30     # keep relations with >= 30 triples
EPOCHS = 120
BATCH = 1024
NNEG = 16
MARGIN = 9.0     # NSSA gamma
LR = 0.01
ADV_TEMP = 1.0

# ---------------------------------------------------------------- build triples
# For each doc, emit entity->entity causal edges: h = a cause_entity, t = an
# entity, r = doc.predicate. This is directional (cause precedes effect).
raw = []
for line in open(os.path.join(DATA, "docs.jsonl")):
    d = json.loads(line)
    causes = d.get("cause_entities") or []
    ents = d.get("entities") or []
    pred = d.get("predicate")
    if not causes or not ents or not pred:
        continue
    for c in causes:
        for e in ents:
            if c == e:
                continue
            raw.append((c, pred, e))

# dedup
raw = list(set(raw))

# prune: entities appearing >= MIN_ENT (as head or tail), relations >= MIN_REL
def prune(trips):
    while True:
        ecount = Counter()
        rcount = Counter()
        for h, r, t in trips:
            ecount[h] += 1; ecount[t] += 1; rcount[r] += 1
        keepE = {e for e, c in ecount.items() if c >= MIN_ENT}
        keepR = {r for r, c in rcount.items() if c >= MIN_REL}
        nt = [(h, r, t) for h, r, t in trips
              if h in keepE and t in keepE and r in keepR]
        if len(nt) == len(trips):
            return nt
        trips = nt

trips = prune(raw)

ents_sorted = sorted({h for h, _, _ in trips} | {t for _, _, t in trips})
rels_sorted = sorted({r for _, r, _ in trips})
E2I = {e: i for i, e in enumerate(ents_sorted)}
R2I = {r: i for i, r in enumerate(rels_sorted)}
nE, nR = len(ents_sorted), len(rels_sorted)

T = np.array([[E2I[h], R2I[r], E2I[t]] for h, r, t in trips], dtype=np.int64)
perm = rng.permutation(len(T))
T = T[perm]
split = int(0.8 * len(T))
train, test = T[:split], T[split:]

# filtered eval: known (h,r)->set(t) and (r,t)->set(h) from ALL triples
all_hr_t = {}
all_rt_h = {}
for h, r, t in T:
    all_hr_t.setdefault((h, r), set()).add(t)
    all_rt_h.setdefault((r, t), set()).add(h)

print(f"[graph] |E|={nE} |R|={nR} |triples|={len(T)} "
      f"(train={len(train)} test={len(test)})")

device = "cpu"
train_t = torch.from_numpy(train).to(device)


# ---------------------------------------------------------------- models
class KGE(nn.Module):
    """Shared skeleton; score() sign convention: higher = more plausible."""
    def __init__(self, kind, nE, nR, dim):
        super().__init__()
        self.kind = kind
        self.dim = dim
        self.margin = MARGIN
        if kind == "rotate":
            self.ent = nn.Embedding(nE, 2 * dim)   # complex: [re|im]
            self.rel = nn.Embedding(nR, dim)       # phase
        elif kind == "distmult":
            self.ent = nn.Embedding(nE, dim)
            self.rel = nn.Embedding(nR, dim)
        elif kind == "transe":
            self.ent = nn.Embedding(nE, dim)
            self.rel = nn.Embedding(nR, dim)
        nn.init.uniform_(self.ent.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel.weight, -0.1, 0.1)

    def _score_hrt(self, h, r, t):
        # h,r,t: (..., ) index tensors, broadcastable
        eh = self.ent(h); et = self.ent(t)
        if self.kind == "rotate":
            d = self.dim
            hr, hi = eh[..., :d], eh[..., d:]
            tr, ti = et[..., :d], et[..., d:]
            phase = self.rel(r) * math.pi          # phase in [-pi,pi]-ish
            rr, ri = torch.cos(phase), torch.sin(phase)
            # rotate head: (hr+i hi)*(rr+i ri)
            pr = hr * rr - hi * ri
            pi = hr * ri + hi * rr
            dr = pr - tr; di = pi - ti
            dist = torch.sqrt(dr * dr + di * di + 1e-9).sum(-1)
            return self.margin - dist
        elif self.kind == "distmult":
            r_ = self.rel(r)
            return (eh * r_ * et).sum(-1)          # symmetric in h,t
        else:  # transe
            r_ = self.rel(r)
            dist = torch.norm(eh + r_ - et, p=2, dim=-1)
            return self.margin - dist

    def forward(self, pos, neg_t):
        h, r, t = pos[:, 0], pos[:, 1], pos[:, 2]
        pos_s = self._score_hrt(h, r, t)                       # (B,)
        # neg_t: (B, NNEG) corrupted tails
        h_e = h.unsqueeze(1).expand_as(neg_t)
        r_e = r.unsqueeze(1).expand_as(neg_t)
        neg_s = self._score_hrt(h_e, r_e, neg_t)               # (B,NNEG)
        return pos_s, neg_s


def nssa_loss(pos_s, neg_s):
    # Negative-sampling self-adversarial loss (RotatE paper).
    weights = torch.softmax(neg_s * ADV_TEMP, dim=1).detach()
    pos_l = -torch.log(torch.sigmoid(pos_s) + 1e-9)
    neg_l = -(weights * torch.log(torch.sigmoid(-neg_s) + 1e-9)).sum(1)
    return (pos_l + neg_l).mean()


def train_model(kind):
    torch.manual_seed(SEED)
    m = KGE(kind, nE, nR, DIM).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    n = len(train_t)
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            b = train_t[idx[s:s + BATCH]]
            neg_t = torch.randint(0, nE, (b.shape[0], NNEG))
            pos_s, neg_s = m(b, neg_t)
            loss = nssa_loss(pos_s, neg_s)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b.shape[0]
        if ep % 30 == 0 or ep == EPOCHS - 1:
            print(f"  [{kind}] ep{ep} loss={tot/n:.4f}")
    return m


@torch.no_grad()
def evaluate(m, mode="tail"):
    """Filtered MRR + Hits@10. mode: 'tail' corrupts t, 'head' corrupts h."""
    m.eval()
    all_e = torch.arange(nE)
    ranks = []
    for h, r, t in test:
        h = int(h); r = int(r); t = int(t)
        if mode == "tail":
            hh = torch.full((nE,), h); rr = torch.full((nE,), r)
            scores = m._score_hrt(hh, rr, all_e)          # (nE,)
            true = t
            filt = all_hr_t.get((h, r), set())
        else:
            tt = torch.full((nE,), t); rr = torch.full((nE,), r)
            scores = m._score_hrt(all_e, rr, tt)
            true = h
            filt = all_rt_h.get((r, t), set())
        true_score = scores[true].item()
        # filtered: mask out other known-true entities
        mask = torch.ones(nE, dtype=torch.bool)
        for x in filt:
            if x != true:
                mask[x] = False
        higher = ((scores > true_score) & mask).sum().item()
        rank = higher + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@10": float(np.mean(ranks <= 10)),
    }


# ---------------------------------------------------------------- run
t0 = time.time()
results = {}
for kind in ["rotate", "distmult", "transe"]:
    print(f"[train] {kind}")
    m = train_model(kind)
    tail = evaluate(m, "tail")
    head = evaluate(m, "head")
    results[kind] = {"tail": tail, "head": head}
    print(f"  [{kind}] tail MRR={tail['mrr']:.4f} H@10={tail['hits@10']:.4f} "
          f"| head MRR={head['mrr']:.4f}")

# ---------------------------------------------------------------- verdict
rot = results["rotate"]["tail"]["mrr"]
dm = results["distmult"]["tail"]["mrr"]
gap = rot - dm
rel_gap = gap / dm if dm > 0 else 0.0
if gap > 0 and rel_gap >= 0.15:
    verdict = "SIGNAL"
elif gap > 0 and rel_gap >= 0.05:
    verdict = "WEAK"
else:
    verdict = "NO-SIGNAL"

metrics = {
    "graph": {"n_entities": nE, "n_relations": nR, "n_triples": int(len(T)),
              "n_train": int(len(train)), "n_test": int(len(test)),
              "relations": rels_sorted, "split": "random 80/20 seed=0"},
    "config": {"dim": DIM, "epochs": EPOCHS, "loss": "NSSA", "nneg": NNEG,
               "margin": MARGIN, "lr": LR, "min_ent": MIN_ENT, "min_rel": MIN_REL},
    "models": results,
    "rotate_vs_distmult": {"rotate_mrr": rot, "distmult_mrr": dm,
                           "abs_gap": gap, "rel_gap": rel_gap},
    "verdict": verdict,
    "runtime_sec": round(time.time() - t0, 1),
}
with open(os.path.join(DATA, "metrics", "kge.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------- figure
models = ["rotate", "distmult", "transe"]
labels = ["RotatE\n(asym)", "DistMult\n(sym)", "TransE"]
mrrs = [results[m]["tail"]["mrr"] for m in models]
hits = [results[m]["tail"]["hits@10"] for m in models]
x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w / 2, mrrs, w, label="MRR (tail, filtered)", color="#2b6cb0")
b2 = ax.bar(x + w / 2, hits, w, label="Hits@10 (tail)", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("score"); ax.set_ylim(0, max(mrrs + hits) * 1.25 + 1e-3)
ax.set_title(f"Causal-KG link prediction  |  RotatE vs DistMult -> {verdict}\n"
             f"|E|={nE} |R|={nR} |T|={len(T)}")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DATA, "figures", "kge.png"), dpi=110)

# ---------------------------------------------------------------- summary
print("---")
print(f"GRAPH: |E|={nE} |R|={nR} |triples|={len(T)} (random 80/20)")
print(f"TAIL MRR: RotatE={rot:.4f}  DistMult={dm:.4f}  "
      f"gap={gap:+.4f} ({rel_gap*100:+.1f}% rel)  H@10 Rot={results['rotate']['tail']['hits@10']:.3f}")
print(f"DIRECTIONALITY: asymmetric {'beats' if gap>0 else 'does not beat'} symmetric "
      f"-> graph is {'directional/relational' if gap>0 else 'co-occurrence-like'}")
print(f"VERDICT: {verdict}")
