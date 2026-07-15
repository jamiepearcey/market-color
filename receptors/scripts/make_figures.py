# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Render one quantitative figure per receptor from the JSON bundles the Rust side
wrote to data/metrics/*.json, plus a master scoreboard. Bootstrap CIs are drawn
either from the CIs Rust already computed or recomputed here from the emitted raw
arrays (same numbers, so figures and stats agree).

Run:  uv run scripts/make_figures.py
Out:  data/figures/*.png  and  RECEPTOR_METRICS.md (master table)
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "data" / "metrics"
FIG = ROOT / "data" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def load(name):
    p = MET / f"{name}.json"
    if not p.exists():
        print(f"  [skip] {name}: no metrics file")
        return None
    return json.loads(p.read_text())

RNG = np.random.default_rng(12345)

def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    bs = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    # rank-based Mann-Whitney
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order)+1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def roc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order] == 1); fp = np.cumsum(labels[order] == 0)
    P = max((labels == 1).sum(), 1); N = max((labels == 0).sum(), 1)
    return np.concatenate([[0], fp/N]), np.concatenate([[0], tp/P])

SCORE = {}  # name -> (headline str, verdict)

def bars_ci(ax, labels, points, los, his, colors=None, ylabel=""):
    x = np.arange(len(labels))
    yerr = [np.array(points)-np.array(los), np.array(his)-np.array(points)]
    ax.bar(x, points, yerr=yerr, capsize=4, color=colors or "#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)

# ---------------- A1 mechanism ----------------
def fig_mechanism():
    d = load("mechanism")
    if not d: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pp = sorted(d["per_predicate"], key=lambda r: -r["acc"])
    names = [r["predicate"] for r in pp]; accs = [r["acc"] for r in pp]
    ax1.barh(range(len(names)), accs, color="#4c72b0")
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
    ax1.axvline(0.5, color="k", ls="--", lw=1, label="cosine floor")
    ax1.axvline(d["single_acc"], color="crimson", ls="-", lw=1.5, label=f"single W {d['single_acc']:.3f}")
    ax1.axvline(d["typed_acc"], color="green", ls="-", lw=1.5, label=f"typed {{W_r}} {d['typed_acc']:.3f}")
    ax1.set_xlabel("direction accuracy"); ax1.set_title("A1 mechanism tensor: per-predicate operator")
    ax1.legend(fontsize=8, loc="lower right"); ax1.invert_yaxis()
    cd = d["cross_desk"]; M = np.array(cd["matrix"], float); desks = cd["desks"]
    im = ax2.imshow(M, cmap="viridis", vmin=0.45, vmax=0.85)
    ax2.set_xticks(range(len(desks))); ax2.set_xticklabels(desks, rotation=30, ha="right", fontsize=8)
    ax2.set_yticks(range(len(desks))); ax2.set_yticklabels(desks, fontsize=8)
    for i in range(len(desks)):
        for j in range(len(desks)):
            ax2.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i,j] < 0.7 else "black", fontsize=8)
    ax2.set_title("cross-desk transfer (train row → test col)")
    ax2.set_xlabel("test desk"); ax2.set_ylabel("train desk")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG/"mechanism.png", dpi=120); plt.close(fig)
    SCORE["A1 mechanism"] = (f"typed {d['typed_acc']:.3f} vs single {d['single_acc']:.3f} (McNemar χ²={d['mcnemar_chi2']:.0f})", "WIN")
    print("  [ok] mechanism.png")

# ---------------- A2 atlas ----------------
def fig_atlas():
    d = load("atlas")
    if not d: return
    ax_rows = d["axes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = [a["axis"] for a in ax_rows]; aucs = [a["auc"] for a in ax_rows]
    los = [a["ci"][0] for a in ax_rows]; his = [a["ci"][1] for a in ax_rows]
    bars_ci(ax1, names, aucs, los, his, ylabel="held-out AUC")
    ax1.axhline(0.5, color="k", ls="--", lw=1); ax1.set_ylim(0.4, 1.02)
    ax1.set_title("A2 atlas: axis separation (95% CI)")
    fp = d["fingerprint"]; M = np.array(fp["matrix"], float)
    im = ax2.imshow(M, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax2.set_xticks(range(len(fp["axis_names"]))); ax2.set_xticklabels(fp["axis_names"], rotation=30, ha="right", fontsize=7)
    ax2.set_yticks(range(len(fp["desks"]))); ax2.set_yticklabels(fp["desks"], fontsize=8)
    ax2.set_title("per-desk fingerprint (mean coordinate)")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG/"atlas.png", dpi=120); plt.close(fig)
    best = min(aucs);
    SCORE["A2 atlas"] = (f"all {len(names)} axes AUC {min(aucs):.2f}–{max(aucs):.2f}", "WIN")
    print("  [ok] atlas.png")

# ---------------- spectrum ----------------
def fig_spectrum():
    d = load("spectrum")
    if not d: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    cv = d["cumulative_variance_pct"]
    ax1.plot(range(1, len(cv)+1), cv, marker="o", ms=3)
    ax1.set_xlabel("# leading directions"); ax1.set_ylabel("cumulative variance %")
    ax1.set_title("anisotropy of the embedding cloud"); ax1.grid(alpha=0.3)
    gs = d["gist_sweep"]; ks = [g["gist_k"] for g in gs]; accs = [g["acc"] for g in gs]
    ax2.plot(ks, accs, marker="s", color="crimson")
    ax2.set_xlabel("gist directions stripped (k)"); ax2.set_ylabel("direction accuracy")
    ax2.set_title("de-gisting sweep"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"spectrum.png", dpi=120); plt.close(fig)
    print("  [ok] spectrum.png")

# ---------------- B5 metric ----------------
def fig_metric():
    d = load("metric")
    if not d: return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ks = d["ks"]
    ax.plot(ks, d["cosine_curve"], marker="o", label="raw cosine (384-d)")
    ax.plot(ks, d["metric_curve"], marker="s", label=f"learned metric (r={d['r']})")
    ax.set_xlabel("k"); ax.set_ylabel("Recall@k"); ax.set_xscale("log")
    ax.set_title("B5 metric learning vs cosine (first-stage recall)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"metric.png", dpi=120); plt.close(fig)
    SCORE["B5 metric"] = (f"R@10 metric {d['metric_r10']:.3f} vs cosine {d['cosine_r10']:.3f}", "NEGATIVE")
    print("  [ok] metric.png")

# ---------------- A3 impact ----------------
def fig_impact():
    d = load("impact")
    if not d: return
    pred = np.array(d["pred"], float); actual = np.array(d["actual"], float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(pred, actual, s=8, alpha=0.3)
    ax1.set_xlabel("predicted impact"); ax1.set_ylabel("actual |zscore|")
    ax1.set_title(f"A3 impact: ρ={d['spearman_embed']:+.3f} (conf baseline {d['spearman_conf']:+.3f})")
    # decile calibration
    order = np.argsort(pred); dec = np.array_split(order, 10)
    means = [actual[idx].mean() for idx in dec]
    ax2.bar(range(1, 11), means, color="#55a868")
    ax2.axhline(actual.mean(), color="k", ls="--", label="overall mean")
    ax2.set_xlabel("predicted-impact decile"); ax2.set_ylabel("mean actual |zscore|")
    ax2.set_title("calibration by predicted decile"); ax2.legend()
    fig.tight_layout(); fig.savefig(FIG/"impact.png", dpi=120); plt.close(fig)
    SCORE["A3 impact"] = (f"Spearman {d['spearman_embed']:+.3f} (conf {d['spearman_conf']:+.3f})", "NEGATIVE")
    print("  [ok] impact.png")

# ---------------- B1 latency ----------------
def fig_latency():
    d = load("latency")
    if not d: return
    pred = np.array(d["pred"], float); actual = np.array(d["actual"], float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hexbin(actual, pred, gridsize=25, cmap="Blues", mincnt=1)
    lim = [0, max(actual.max(), pred.max())]
    ax1.plot(lim, lim, "r--", lw=1); ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("actual lag (days)"); ax1.set_ylabel("predicted lag (days)")
    ax1.set_title(f"B1 latency: ρ={d['spearman']:+.3f}")
    labels = ["mean baseline", "content probe"]
    ax2.bar(labels, [d["mae_baseline"], d["mae_probe"]], color=["#999999", "#4c72b0"])
    ax2.set_ylabel("MAE (days, lower=better)")
    ax2.set_title(f"MAE: {d['mae_probe']:.2f} vs {d['mae_baseline']:.2f}d")
    fig.tight_layout(); fig.savefig(FIG/"latency.png", dpi=120); plt.close(fig)
    SCORE["B1 latency"] = (f"MAE {d['mae_probe']:.2f}d vs {d['mae_baseline']:.2f}d, ρ={d['spearman']:+.3f}", "WIN")
    print("  [ok] latency.png")

# ---------------- B2 spillover ----------------
def fig_spillover():
    d = load("spillover")
    if not d: return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cohorts = ["primary", "secondary", "all"]
    naive = [d[c]["naive"] for c in cohorts]; recp = [d[c]["receptor"] for c in cohorts]
    def ci(c, key):
        arr = d[c].get(key)
        return boot_mean(arr)[1:] if arr else (None, None)
    x = np.arange(len(cohorts)); w = 0.38
    nerr = _err([ci(c, "naive_hits") for c in cohorts], naive)
    rerr = _err([ci(c, "recp_hits") for c in cohorts], recp)
    ax.bar(x-w/2, naive, w, yerr=nerr, capsize=4, label="naive cosine", color="#999999")
    ax.bar(x+w/2, recp, w, yerr=rerr, capsize=4, label="receptor W_spill", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(cohorts); ax.set_ylabel("Recall@3")
    ax.set_title("B2 spillover: moved-symbol recall"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"spillover.png", dpi=120); plt.close(fig)
    SCORE["B2 spillover"] = (f"secondary R@3 receptor {d['secondary']['receptor']:.3f} vs cosine {d['secondary']['naive']:.3f}", "NEGATIVE")
    print("  [ok] spillover.png")

def _err(cis, points):
    lo = []; hi = []
    for (c, p) in zip(cis, points):
        if c[0] is None: lo.append(0); hi.append(0)
        else: lo.append(p-c[0]); hi.append(c[1]-p)
    return [lo, hi]

# ---------------- B3 confound ----------------
def fig_confound():
    d = load("confound")
    if not d: return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fpr_p, tpr_p = roc(d["probe_scores"], d["labels"])
    fpr_c, tpr_c = roc(d["cosine_scores"], d["labels"])
    ax.plot(fpr_p, tpr_p, label=f"learned probe (AUC {d['probe_auc']:.3f})", lw=2)
    ax.plot(fpr_c, tpr_c, label=f"cosine only (AUC {d['cosine_auc']:.3f})", lw=2, ls="--")
    ax.plot([0,1],[0,1],"k:",lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("B3 confounder: direct edge vs common-cause sibling"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG/"confound.png", dpi=120); plt.close(fig)
    SCORE["B3 confound"] = (f"probe AUC {d['probe_auc']:.3f} vs cosine {d['cosine_auc']:.3f}", "WIN")
    print("  [ok] confound.png")

# ---------------- B4 source ----------------
def fig_source():
    d = load("source")
    if not d: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr = roc(d["probe_scores"], d["labels"])
    ax1.plot(fpr, tpr, lw=2, label=f"embedding (AUC {d['embed_auc']:.3f})")
    ax1.plot([0,1],[0,1],"k:",lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title(f"B4 source: predict corroboration\n(embedding {d['embed_auc']:.3f} vs conf {d['conf_auc']:.3f})")
    ax1.legend(loc="lower right")
    prov = d["provenance"][:12]
    srcs = [p["source"][:16] for p in prov]; pct = [p["pct_corrob"] for p in prov]
    ax2.barh(range(len(srcs)), pct, color="#c44e52")
    ax2.set_yticks(range(len(srcs))); ax2.set_yticklabels(srcs, fontsize=8); ax2.invert_yaxis()
    ax2.set_xlabel("% corroborated"); ax2.set_title("provenance: corroboration rate by source")
    fig.tight_layout(); fig.savefig(FIG/"source.png", dpi=120); plt.close(fig)
    SCORE["B4 source"] = (f"embedding AUC {d['embed_auc']:.3f} vs confidence {d['conf_auc']:.3f}", "WIN")
    print("  [ok] source.png")

# ---------------- C1 consensus ----------------
def fig_consensus():
    d = load("consensus")
    if not d: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(d["contrarian_scores"], bins=40, color="#8172b3")
    ax1.set_xlabel("dissent score (1 = orthogonal to desk consensus)")
    ax1.set_ylabel("facts"); ax1.set_title("C1 consensus: dissent distribution")
    m = d["market"]
    tz, rz = m["topdecile_z"], m["rest_z"]
    (pt, lo, hi) = boot_mean(tz); (pr, rlo, rhi) = boot_mean(rz)
    ax2.bar(["top-decile\ncontrarian", "rest"], [pt, pr],
            yerr=[[pt-lo, pr-rlo],[hi-pt, rhi-pr]], capsize=5, color=["#c44e52", "#999999"])
    ax2.set_ylabel("mean |zscore| of later move")
    ax2.set_title(f"do dissenters precede bigger moves? ratio {m['ratio']:.2f}x")
    fig.tight_layout(); fig.savefig(FIG/"consensus.png", dpi=120); plt.close(fig)
    SCORE["C1 consensus"] = (f"move ratio {m['ratio']:.2f}x (top-decile vs rest)", "EXPLORATORY")
    print("  [ok] consensus.png")

# ---------------- C2 regime ----------------
def fig_regime():
    d = load("regime")
    if not d: return
    tl = d["timeline"]
    dates = [t["date"] for t in tl]; act = [t["actual"] for t in tl]; proj = [t["proj"] for t in tl]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dates))
    ax.plot(x, act, marker="o", label="actual risk-on (prices)")
    ax.plot(x, proj, marker="s", label="projected (fact content)")
    for i, t in enumerate(tl):
        if t.get("is_test"): ax.axvspan(i-0.5, i+0.5, color="orange", alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("risk-on score")
    ax.set_title(f"C2 regime: ρ={d['spearman']:+.2f} on held-out (orange), n_days={d['n_days']} (exploratory)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"regime.png", dpi=120); plt.close(fig)
    SCORE["C2 regime"] = (f"held-out ρ={d['spearman']:+.2f}, n_days={d['n_days']}", "EXPLORATORY (thin)")
    print("  [ok] regime.png")

# ---------------- C3 arc ----------------
def fig_arc():
    d = load("arc")
    if not d: return
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["Recall@10", "MRR"]
    cos = [d["cosine_r10"], d["cosine_mrr"]]; arc = [d["arc_r10"], d["arc_mrr"]]
    x = np.arange(2); w = 0.38
    ax.bar(x-w/2, cos, w, label="cosine", color="#999999")
    ax.bar(x+w/2, arc, w, label="arc operator", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("score")
    ax.set_title("C3 narrative-arc: next-2-day continuation"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"arc.png", dpi=120); plt.close(fig)
    SCORE["C3 arc"] = (f"R@10 arc {d['arc_r10']:.3f} vs cosine {d['cosine_r10']:.3f}", "NEGATIVE")
    print("  [ok] arc.png")

def fig_diagnostic():
    pl = load("probe_ladder"); pt = load("probe_temporal")
    if not pl: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: linear vs best-nonlinear per task (classification acc / regression rho)
    names, lin, nl, kind = [], [], [], []
    for r in pl:
        names.append(r["task"].split(" (")[0])
        if "linear_acc" in r:
            lin.append(r["linear_acc"]); nl.append(max(r["mlp_acc"], r["gb_acc"])); kind.append("acc")
        else:
            lin.append(r["linear_spearman"]); nl.append(max(r["mlp_spearman"], r["gb_spearman"])); kind.append("rho")
    x = np.arange(len(names)); w = 0.38
    ax1.bar(x-w/2, lin, w, label="linear probe", color="#4c72b0")
    ax1.bar(x+w/2, nl, w, label="best nonlinear (MLP/GB)", color="#dd8452")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("held-out score (acc or ρ)"); ax1.legend()
    ax1.set_title("probing ladder: is signal linear, nonlinear, or absent?")
    # right: direction & impact random vs temporal
    if pt:
        d = pt["direction"]; im = pt["impact"]
        cats = ["dir\nrandom", "dir\ntemporal", "impact\nrandom", "impact\ntemporal"]
        linv = [d[0]["linear_auc"], d[1]["linear_auc"], im[0]["linear_rho"], im[1]["linear_rho"]]
        nlv  = [d[0]["gb_auc"],     d[1]["gb_auc"],     im[0]["gb_rho"],     im[1]["gb_rho"]]
        x2 = np.arange(4)
        ax2.bar(x2-w/2, linv, w, label="linear", color="#4c72b0")
        ax2.bar(x2+w/2, nlv, w, label="gradient boosting", color="#dd8452")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xticks(x2); ax2.set_xticklabels(cats, fontsize=8)
        ax2.set_ylabel("AUC (direction) / ρ (impact)"); ax2.legend()
        ax2.set_title("does the headroom survive a temporal split?")
    fig.tight_layout(); fig.savefig(FIG/"diagnostic.png", dpi=120); plt.close(fig)
    print("  [ok] diagnostic.png")

def scoreboard():
    if not SCORE: return
    order = ["A1 mechanism","A2 atlas","A3 impact","B1 latency","B2 spillover",
             "B3 confound","B4 source","B5 metric","C1 consensus","C2 regime","C3 arc"]
    rows = [(k, *SCORE[k]) for k in order if k in SCORE]
    cmap = {"WIN":"#55a868","NEGATIVE":"#c44e52","EXPLORATORY":"#dd8452","EXPLORATORY (thin)":"#dd8452"}
    fig, ax = plt.subplots(figsize=(11, 0.5*len(rows)+1))
    ax.axis("off")
    for i,(k,h,v) in enumerate(reversed(rows)):
        c = cmap.get(v, "#888")
        ax.barh(i, 1, color=c, alpha=0.15)
        ax.text(0.01, i, f"{k}", va="center", fontsize=10, weight="bold")
        ax.text(0.24, i, h, va="center", fontsize=9)
        ax.text(0.95, i, v, va="center", ha="right", fontsize=9, color=c, weight="bold")
    ax.set_xlim(0,1); ax.set_ylim(-0.5, len(rows)-0.5)
    ax.set_title("Receptor scoreboard — headline metric vs baseline", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"scoreboard.png", dpi=120); plt.close(fig)
    # markdown table
    md = ["# Receptor metrics\n", "| receptor | headline (vs baseline) | verdict |", "|---|---|---|"]
    for k,h,v in rows:
        md.append(f"| {k} | {h} | {v} |")
    (ROOT/"RECEPTOR_METRICS.md").write_text("\n".join(md)+"\n")
    print("  [ok] scoreboard.png + RECEPTOR_METRICS.md")

if __name__ == "__main__":
    print("rendering figures ->", FIG)
    for f in [fig_mechanism, fig_atlas, fig_spectrum, fig_metric, fig_impact,
              fig_latency, fig_spillover, fig_confound, fig_source,
              fig_consensus, fig_regime, fig_arc, fig_diagnostic]:
        try:
            f()
        except Exception as e:
            print(f"  [ERR] {f.__name__}: {e}")
    scoreboard()
    print("done.")
