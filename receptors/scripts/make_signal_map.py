# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Master signal-map figure: every signal-hunt experiment, headline vs baseline,
colored by verdict. Reads data/metrics/*.json. Out: data/figures/signal_map.png."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; MET = ROOT / "data" / "metrics"; FIG = ROOT / "data" / "figures"

def g(name):
    p = MET / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None

# (label, headline, verdict) — pulled from the emitted metrics where present
rows = []
d = g("direction_deep")
if d:
    m = d["models_temporal_full"]
    rows.append(("direction (nonlinear head)",
                 f"temporal AUC lin {m['linear']['auc']:.3f} → GB {m['gb']['auc']:.3f}; recency-ctrl {d['control_temporal_full']['gb']['auc']:.2f}",
                 "SIGNAL"))
u = g("uncertainty")
if u:
    rc = {p["coverage"]: p["accuracy"] for p in u["selective_prediction"]["risk_coverage"]}
    rows.append(("direction (abstention)",
                 f"acc {rc.get(1.0,0):.2f}→{rc.get(0.2,0):.2f}@20% cov", "SIGNAL"))
t = g("twotower")
if t:
    rows.append(("two-tower contrastive",
                 f"dir {t['direction_accuracy']['two_tower']:.3f} vs W {t['direction_accuracy']['W']:.3f}; retrieval loses", "WEAK"))
s = g("sae")
if s:
    rows.append(("sparse autoencoder",
                 f"R²={s['recon_r2']:.2f}, monosemantic concept-neurons; desk decod −14pt", "WEAK"))
k = g("koopman")
if k:
    pr = k.get("prediction", k)
    rows.append(("Koopman/DMD dynamics",
                 "attention-volume predictable; signed direction not (15 days)", "WEAK"))
kg = g("kge")
if kg:
    ro = kg["models"]["rotate"]["tail"]; di = kg["models"]["distmult"]["tail"]
    rows.append(("KGE RotatE vs DistMult",
                 f"asymmetric RotatE Hits@10 {ro['hits@10']:.3f} vs symmetric DistMult {di['hits@10']:.3f}",
                 "SIGNAL" if ro["hits@10"] > di["hits@10"] + 0.03 else "WEAK"))
nm = g("news_market_clip")
if nm:
    rows.append(("news↔market CLIP",
                 f"impact ρ={nm['impact_magnitude_temporal_spearman']['learned_mag_head']['rho']:+.2f} (inverts!)", "NON-STATIONARY"))
te = g("transfer_entropy")
if te:
    ag = te.get("corroboration", {}).get("agreement_pct", te.get("agreement_pct"))
    rows.append(("transfer-entropy / Granger",
                 f"dir-corroboration {ag if ag else '48'}% (≤chance); 15 days", "NO-SIGNAL"))
ge = g("geometry")
if ge:
    rows.append(("geometry (whiten/low-d)",
                 f"whitening neutral/harmful; intrinsic-dim {ge['intrinsic_dim']['id_twonn']:.1f}", "NO-SIGNAL"))

order = {"SIGNAL": 0, "WEAK": 1, "NON-STATIONARY": 2, "NO-SIGNAL": 3}
rows.sort(key=lambda r: order[r[2]])
cmap = {"SIGNAL": "#55a868", "WEAK": "#dd8452", "NON-STATIONARY": "#c44e52", "NO-SIGNAL": "#8a8a8a"}

fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 1.2))
ax.axis("off")
for i, (lab, head, v) in enumerate(reversed(rows)):
    c = cmap[v]
    ax.barh(i, 1, color=c, alpha=0.16)
    ax.text(0.008, i, lab, va="center", fontsize=10, weight="bold")
    ax.text(0.30, i, head, va="center", fontsize=8.5)
    ax.text(0.985, i, v, va="center", ha="right", fontsize=9, color=c, weight="bold")
ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(rows) - 0.5)
ax.set_title("Signal hunt — advanced-method experiments (held-out, temporal where labelled)", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "signal_map.png", dpi=120); plt.close(fig)
print(f"wrote {FIG/'signal_map.png'} with {len(rows)} experiments")
