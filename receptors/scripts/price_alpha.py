# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Textual-alpha test: after removing expressed sentiment from the embedding, does
the RESIDUAL predict the ACTUAL price-move sign (exogenous) beyond sentiment?

Q1: does expressed sentiment predict the real move at all?
Q2: does the embedding predict the real move BEYOND sentiment (residualized)?

Honest guards: articles STRICTLY before the move day (no reporting-after leakage);
temporal split by move-date; metrics reported at EVENT level (articles within one
(symbol,date) move share a label, so pair-level counts overstate sample size).
"""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"

# --- expressed-sentiment direction w (ridge on claim embeddings) ---
C = np.load(D / "claims.npy")
cy = np.array([json.loads(l)["y"] for l in (D / "claims_labels.jsonl").read_text().splitlines() if l.strip()], float)
lam = 1.0
w = np.linalg.solve(C.T @ C + lam * np.eye(C.shape[1]), C.T @ cy)
w /= np.linalg.norm(w) + 1e-9

# --- article embeddings + dates ---
E = np.load(D / "embeddings.npy")
docs = [json.loads(l) for l in (D / "docs.jsonl").read_text().splitlines() if l.strip()]
id2row = {d["doc_id"]: i for i, d in enumerate(docs)}
id2date = {d["doc_id"]: d.get("date", "") for d in docs}

# --- price pairs: keep only articles strictly BEFORE the move (lag>=1) ---
pairs = [json.loads(l) for l in (D / "price_pairs.jsonl").read_text().splitlines() if l.strip()]
rows, y, sent, evt, mdate = [], [], [], [], []
for p in pairs:
    r = id2row.get(p["doc_id"])
    if r is None:
        continue
    ad = id2date.get(p["doc_id"], "")
    if not ad or ad >= p["move_date"]:   # strictly before the move day
        continue
    x = E[r]
    rows.append(x); y.append(p["sign"]); sent.append(float(x @ w))
    evt.append((p["symbol"], p["move_date"])); mdate.append(p["move_date"])
X = np.array(rows); y = np.array(y, float); sent = np.array(sent); mdate = np.array(mdate)
print(f"usable pairs (article strictly before move): {len(y)}")
print(f"distinct move-events: {len(set(evt))}   (this is the real sample size)")

# --- temporal split by move-date ---
dts = sorted(set(mdate)); cut = dts[int(len(dts) * 0.6)]
tr, te = mdate < cut, mdate >= cut
print(f"split at {cut}: {tr.sum()} train / {te.sum()} test pairs; "
      f"{len(set(e for e,m in zip(evt,mdate) if m>=cut))} test events\n")

def acc(pred): return float(np.mean(np.sign(pred[te]) == y[te]))
def event_acc(pred):
    # one vote per test event (mean prediction), vs its label
    d = {}
    for i in np.where(te)[0]:
        d.setdefault(evt[i], [[], y[i]])[0].append(pred[i])
    ok = [ (np.sign(np.mean(v[0])) == v[1]) for v in d.values() ]
    return float(np.mean(ok)), len(ok)

maj = max(y[te].mean() > 0, y[te].mean() < 0)  # majority baseline (unsigned share)
base = max((y[te] == 1).mean(), (y[te] == -1).mean())

# Q1: expressed sentiment -> actual move
# Q2a: full-embedding probe -> actual move
lamp = 5.0
d = X.shape[1]
wf = np.linalg.solve(X[tr].T @ X[tr] + lamp * np.eye(d), X[tr].T @ y[tr])
# Q2b: residualize (remove sentiment direction) then probe
Xr = X - np.outer(X @ w, w)
wr = np.linalg.solve(Xr[tr].T @ Xr[tr] + lamp * np.eye(d), Xr[tr].T @ y[tr])

print("prediction of ACTUAL price-move sign (test):")
print(f"  {'predictor':<34}{'pair-acc':>9}{'event-acc':>11}")
print(f"  {'majority baseline':<34}{base:>9.3f}{'—':>11}")
ea,ne = event_acc(sent);      print(f"  {'expressed sentiment (Q1)':<34}{acc(sent):>9.3f}{ea:>11.3f}")
ea,_  = event_acc(X @ wf);    print(f"  {'full-embedding probe (Q2a)':<34}{acc(X @ wf):>9.3f}{ea:>11.3f}")
ea,_  = event_acc(Xr @ wr);   print(f"  {'sentiment-REMOVED residual (Q2b)':<34}{acc(Xr @ wr):>9.3f}{ea:>11.3f}")
print(f"\n  (event-level n = {ne} — interpret accordingly)")
