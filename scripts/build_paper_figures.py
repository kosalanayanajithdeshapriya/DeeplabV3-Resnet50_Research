"""Build paper figures from saved CSV/JSON evidence plus the inference-only re-scored
confusion matrices and per-image scores in paper_report/data/. No training."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
UNET = Path(r"d:/Reasearch/Unet")
OUT = ROOT / "paper_report" / "figures"
DATA = ROOT / "paper_report" / "data"
OUT.mkdir(parents=True, exist_ok=True)

C = ["developing", "flowering", "fruiting", "seeding"]
CL = ["Developing", "Flowering", "Fruiting", "Seeding"]
DLV3, UNETC, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def cm_panel(ax, cm, title):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), CL, rotation=30, ha="right")
    ax.set_yticks(range(4), CL)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


rec = json.loads((DATA / "recomputed_cms.json").read_text())
rows = json.loads((DATA / "paired_rows.json").read_text())


def cm_from(pred_key):
    m = np.zeros((4, 4), int)
    for r in rows:
        m[C.index(r["true"]), C.index(r[pred_key])] += 1
    return m


# ---- Fig 1: dataset composition ----
counts = {"Train": [139 + 3, 210, 71, 264], "Validation": [20, 30, 9, 76], "Test": [41, 60, 21, 38]}
tot = np.sum(list(counts.values()), axis=0)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
left = np.zeros(4)
for (k, v), col in zip(counts.items(), [DLV3, UNETC, THIRD]):
    ax.barh(CL, v, left=left, color=col, edgecolor="white", linewidth=1.5, label=k, height=0.6)
    left += v
for i, t in enumerate(tot):
    ax.text(t + 6, i, f"{t} ({t / tot.sum():.1%})", va="center", color=INK, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 450)
ax.set_xlabel("Images")
ax.grid(axis="y", visible=False)
ax.legend(ncol=3, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.22))
ax.set_title("Class distribution (982 images; imbalance ratio seeding:fruiting = 3.74:1)", fontsize=9.5, loc="left")
save(fig, "fig1_class_distribution.png")

# ---- Fig 2: fixed-split pipeline confusion matrices ----
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
panels = [
    (rec["resnet_train_original_eval_original"]["cm"], "(a) ResNet50 on original images\n(no segmentation) - 96.88%"),
    (rec["resnet_train_masked_eval_masked"]["cm"], "(b) Masked ResNet50, ground-truth masks\n(oracle upper bound) - 91.88%"),
    (rec["resnet_train_original_eval_masked"]["cm"], "(c) Original ResNet50 on GT-masked images\n(background removed) - 76.88%"),
    (cm_from("dlv3_pred"), "(d) DeepLabV3 -> ResNet50 pipeline\n87.50%"),
    (cm_from("unet_pred"), "(e) U-Net (crop-aligned) -> ResNet50\n89.38%"),
    ([[39, 2, 0, 0], [2, 31, 27, 0], [0, 2, 18, 1], [0, 0, 0, 38]], "(f) U-Net (native resize) -> ResNet50\n78.75%"),
]
for ax, (cm, t) in zip(axes.flat, panels):
    cm_panel(ax, cm, t)
fig.suptitle("Fixed-split confusion matrices (160 test images; counts and row-normalised recall)", fontsize=10.5, x=0.02, ha="left")
fig.tight_layout()
save(fig, "fig2_fixed_split_confusion_matrices.png")

# ---- Fig 3: U-Net CV fold confusion matrices ----
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
accs = [0.85, 0.8625, 0.83125, 0.89375, 0.84375]
for k in range(5):
    cm_panel(axes.flat[k], rec["unet_cv_fold_cms"][k], f"Fold {k + 1} U-Net -> fixed ResNet50 - {accs[k]:.2%}")
cm_panel(axes.flat[5], rec["unet_cv_pooled_cm"], "Pooled over 5 folds (800 predictions) - 85.63%")
fig.suptitle("U-Net five-fold CV: per-fold confusion matrices on the fixed 160-image test set", fontsize=10.5, x=0.02, ha="left")
fig.tight_layout()
save(fig, "fig3_unet_cv_confusion_matrices.png")

# ---- Fig 4: paired segmentation by stage ----
d = {c: [r["dlv3_fg_iou"] for r in rows if r["true"] == c] for c in C}
u = {c: [r["unet_fg_iou"] for r in rows if r["true"] == c] for c in C}
d["all"] = [r["dlv3_fg_iou"] for r in rows]
u["all"] = [r["unet_fg_iou"] for r in rows]
keys = C + ["all"]
labels = CL + ["All (n=160)"]
pv = {"developing": "n.s.", "flowering": "p<0.001", "fruiting": "n.s.", "seeding": "p<0.001", "all": "p<0.001"}
x = np.arange(5)
fig, ax = plt.subplots(figsize=(7, 3.4))
w = 0.36
for off, src, col, lab in [(-w / 2, d, DLV3, "DeepLabV3-ResNet50"), (w / 2, u, UNETC, "U-Net (crop-aligned)")]:
    m = [np.mean(src[k]) for k in keys]
    sd = [np.std(src[k], ddof=1) for k in keys]
    ax.bar(x + off, m, w, color=col, label=lab, edgecolor="white", linewidth=1.5)
    ax.errorbar(x + off, m, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5)
for i, k in enumerate(keys):
    ax.text(i, 1.0, pv[k], ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [f"{l}\n(n={len(d[k])})" if k != "all" else l for l, k in zip(labels, keys)])
ax.set_ylim(0, 1.06)
ax.set_ylabel("Plant (foreground) IoU, mean +/- SD")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.2))
ax.set_title("Segmentation-only comparison per growth stage (paired Wilcoxon signed-rank)", fontsize=9.5, loc="left")
save(fig, "fig4_segmentation_by_stage.png")


# ---- Fig 5: CV fold-level dot plot ----
def rcsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


dcv = rcsv(ROOT / "deeplabv3_pipeline_kfold_cv_results.csv")
ucv = rcsv(UNET / "outputs/cross_validation/fusion_pipeline_cv_results.csv")
mets = [("Accuracy", "pipeline_test_accuracy", "accuracy"), ("Macro precision", "pipeline_test_precision_macro", "macro_precision"),
        ("Macro recall", "pipeline_test_recall_macro", "macro_recall"), ("Macro F1", "pipeline_test_f1_macro", "macro_f1")]
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, (lab, dk, uk) in enumerate(mets):
    for off, vals, col in [(-0.17, [float(r[dk]) for r in dcv], DLV3), (0.17, [float(r[uk]) for r in ucv], UNETC)]:
        ax.scatter(np.full(5, i + off), vals, s=34, color=col, edgecolor="white", linewidth=1, zorder=3)
        m, s = np.mean(vals), np.std(vals, ddof=1)
        side = -1 if off < 0 else 1  # DeepLabV3 summary sits left of its dots, U-Net right
        ax.errorbar(i + off + side * 0.09, m, yerr=s, fmt="_", color=INK, markersize=12, elinewidth=1.1, capsize=3, zorder=4)
        ax.text(i + off + side * 0.13, m, f"{m:.3f}\n+/-{s:.3f}", fontsize=6.8, color=INK2, va="center",
                ha="right" if side < 0 else "left")
ax.scatter([], [], color=DLV3, label="DeepLabV3 pipeline CV (both models retrained, 5 disjoint outer folds)")
ax.scatter([], [], color=UNETC, label="U-Net pipeline CV (U-Net retrained, fixed classifier, fixed test set)")
ax.set_xticks(range(4), [m[0] for m in mets])
ax.set_ylim(0.6, 0.97)
ax.set_xlim(-0.7, 3.75)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Score (dots = folds; bar = mean +/- SD)")
ax.legend(loc="lower left", fontsize=7.3)
ax.set_title("Five-fold CV of the full segmentation -> classification pipelines", fontsize=9.5, loc="left")
save(fig, "fig5_pipeline_cv_folds.png")

# ---- Fig 6: why SD differs ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
for i, c in enumerate(C):
    for off, m, col in [(-0.2, "precision", DLV3), (0, "recall", UNETC), (0.2, "f1", THIRD)]:
        v = [float(r[f"{c}_{m}"]) for r in ucv]
        a1.scatter(np.full(5, i + off), v, s=26, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                   label=m.capitalize() if i == 0 else None)
    sd = np.std([float(r[f"{c}_f1"]) for r in ucv], ddof=1)
    a1.text(i + 0.32, 0.43, f"F1 SD\n{sd:.3f}", ha="center", fontsize=7.5, color=INK2)
a1.set_xticks(range(4), [f"{l}\n(n={n})" for l, n in zip(CL, [41, 60, 21, 38])])
a1.set_ylim(0.4, 1.04)
a1.grid(axis="x", visible=False)
a1.legend(loc="upper center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.17))
a1.set_title("(a) U-Net CV per-class scores by fold", fontsize=9.5, loc="left")
labs = ["Seg. IoU\n(DLV3 CV)", "Pipeline acc.\n(DLV3 CV)", "Binomial SE\nacc., n=196", "Macro F1\n(DLV3 CV)", "Macro prec.\n(DLV3 CV)", "Binomial SE\nfruiting recall,\nn=20"]
vals = [0.37, 2.16, 2.73, 4.30, 6.74, 9.55]
cols = [THIRD, DLV3, "#b9b8b2", DLV3, DLV3, "#b9b8b2"]
a2.bar(range(6), vals, color=cols, edgecolor="white", linewidth=1.5, width=0.62)
for i, v in enumerate(vals):
    a2.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=8, color=INK)
a2.set_xticks(range(6), labs, fontsize=7.2)
a2.set_ylabel("SD across folds (percentage points)")
a2.grid(axis="x", visible=False)
a2.set_title("(b) Fold SD vs. finite-sample expectation (grey)", fontsize=9.5, loc="left")
fig.tight_layout()
save(fig, "fig6_std_analysis.png")

# ---- Fig 7: segmentation CV folds ----
seg_only = rcsv(ROOT / "checkpoints_segmentation/deeplabv3_kfold_cv_results.csv")
useg = rcsv(UNET / "outputs/cross_validation/unet_segmentation_cv_results.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
f = np.arange(1, 6)
a1.plot(f, [float(r["test_iou"]) for r in seg_only], "-o", color=DLV3, lw=2, ms=6, label="DLV3 segmenter-only CV")
a1.plot(f, [float(r["seg_test_iou"]) for r in dcv], "-o", color=THIRD, lw=2, ms=6, label="DLV3 within pipeline CV")
a1.set_ylabel("Both-class mean IoU (batch-avg.)")
a1.set_ylim(0.70, 0.80)
a1.set_title("(a) DeepLabV3 segmentation CV", fontsize=9.5, loc="left")
a1.legend(fontsize=7.5, loc="lower left")
uf1 = [float(r["test_leaf_f1"]) for r in useg]
a2.plot(f, uf1, "-o", color=UNETC, lw=2, ms=6, label="U-Net leaf Dice/F1 (pooled pixels)")
a2.plot(f, [x / (2 - x) for x in uf1], "--o", color=UNETC, lw=2, ms=6, alpha=0.6, label="U-Net leaf IoU (derived)")
a2.set_ylim(0.55, 0.85)
a2.set_title("(b) U-Net segmentation CV (fixed test set)", fontsize=9.5, loc="left")
a2.legend(fontsize=7.5, loc="lower left")
for a in (a1, a2):
    a.set_xticks(f, [f"Fold {i}" for i in f])
    a.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig7_segmentation_cv_folds.png")
print("figures written to", OUT)
