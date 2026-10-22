"""Pixel-level confusion matrices for the five U-Net CV folds, each scored on the fixed 160-image
test set with pooled pixels -- same preprocessing as unet_cross_validation.ipynb (CVSegmentationDataset:
cv2 resize to 224x224, nearest-neighbour masks, threshold 0.5). Checks each fold against
d:/Reasearch/Unet/outputs/cross_validation/unet_segmentation_cv_results.csv."""
import os, sys, json
import numpy as np, pandas as pd, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_cms_models import UNet
dev = torch.device("cuda"); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
U = r"d:/Reasearch/Unet"; PROC = os.path.join(U, "processed", "test")
FIG = r"d:/Reasearch/Resnet50test/paper_report/figures"; DATA = os.path.dirname(os.path.abspath(__file__))
mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
df = pd.read_csv(os.path.join(PROC, "stage_mapping.csv"))
ref = pd.read_csv(os.path.join(U, "outputs", "cross_validation", "unet_segmentation_cv_results.csv"))

def load(row):
    img = cv2.cvtColor(cv2.imread(os.path.join(PROC, "images", row["filename"])), cv2.COLOR_BGR2RGB)
    msk = cv2.imread(os.path.join(PROC, "masks", row["mask_filename"]), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    msk = (cv2.resize(msk, (224, 224), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    x = torch.from_numpy(np.transpose((img.astype(np.float32) / 255.0 - mean) / std, (2, 0, 1)).copy())
    return x, msk
data = [load(r) for _, r in df.iterrows()]

def metrics(cm):
    tn, fp, fn, tp = cm.ravel().astype(float)
    return {"pixel_accuracy": (tp + tn) / cm.sum(), "leaf_precision": tp / (tp + fp), "leaf_recall": tp / (tp + fn),
            "leaf_f1": 2 * tp / (2 * tp + fp + fn), "leaf_iou": tp / (tp + fp + fn)}

cms = []
with torch.no_grad():
    for fold in range(5):
        net = UNet().to(dev)
        net.load_state_dict(torch.load(os.path.join(U, "checkpoints", "cv_folds", f"unet_fold{fold}.pth"), map_location=dev, weights_only=True)); net.eval()
        cm = np.zeros((2, 2), np.int64)
        for i in range(0, len(data), 8):
            xb = torch.stack([d[0] for d in data[i:i + 8]]).to(dev)
            pb = (torch.sigmoid(net(xb)) > 0.5)[:, 0].cpu().numpy().astype(np.uint8)
            for p, (_, t) in zip(pb, data[i:i + 8]):
                cm += np.array([[np.sum((t == 0) & (p == 0)), np.sum((t == 0) & (p == 1))],
                                [np.sum((t == 1) & (p == 0)), np.sum((t == 1) & (p == 1))]])
        cms.append(cm); m = metrics(cm); r = ref.iloc[fold]
        print(f"fold {fold + 1}: acc {m['pixel_accuracy']:.4f} (saved {r.test_accuracy:.4f})  P {m['leaf_precision']:.4f} ({r.test_leaf_precision:.4f})"
              f"  R {m['leaf_recall']:.4f} ({r.test_leaf_recall:.4f})  F1 {m['leaf_f1']:.4f} ({r.test_leaf_f1:.4f})\n{cm}")

ms = {k: np.array([metrics(c)[k] for c in cms]) for k in metrics(cms[0])}
print("mean+/-SD (sample SD, ddof=1): " + "  ".join(f"{k}={v.mean():.4f}+/-{v.std(ddof=1):.4f}" for k, v in ms.items()))
total = np.sum(cms, axis=0); avg = total / 5
json.dump({"folds": [{"fold": f + 1, "cm": c.tolist(), **metrics(c)} for f, c in enumerate(cms)],
           "mean_cm": avg.tolist(), "summed_cm": total.tolist(),
           "mean_sd": {k: [float(v.mean()), float(v.std(ddof=1))] for k, v in ms.items()}},
          open(os.path.join(DATA, "unet_cv_pixel_cms.json"), "w"), indent=1)

LBL = ["Background", "Leaf"]
def panel(ax, cm, title, fmt="{:,.0f}"):
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]:.2f}%\n({fmt.format(cm[i, j])} px)", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks([0, 1], LBL); ax.set_yticks([0, 1], LBL); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=9.5)

fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
for f, (ax, c) in enumerate(zip(axes, cms)):
    m = metrics(c); panel(ax, c, f"Fold {f + 1}\nacc {m['pixel_accuracy']:.4f} | P {m['leaf_precision']:.4f} | R {m['leaf_recall']:.4f} | F1 {m['leaf_f1']:.4f}")
fig.suptitle("U-Net five-fold CV: pixel confusion matrices on the fixed 160-image test set (segmentation only, row-normalised)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "seg_pixel_cm_unet_cv_folds.png"), dpi=300, facecolor="white")

s = {k: (v.mean(), v.std(ddof=1)) for k, v in ms.items()}
fig, ax = plt.subplots(figsize=(5.6, 5))
panel(ax, avg, "U-Net (5-fold CV) - segmentation only\nmean confusion matrix over 5 folds, 160 test images\n"
      f"acc {s['pixel_accuracy'][0]:.4f}±{s['pixel_accuracy'][1]:.4f} | P {s['leaf_precision'][0]:.4f}±{s['leaf_precision'][1]:.4f}\n"
      f"R {s['leaf_recall'][0]:.4f}±{s['leaf_recall'][1]:.4f} | F1 {s['leaf_f1'][0]:.4f}±{s['leaf_f1'][1]:.4f}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "seg_pixel_cm_unet_cv_mean.png"), dpi=300, facecolor="white")
