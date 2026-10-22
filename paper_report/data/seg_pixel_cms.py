"""Segmentation-only evaluation: pixel-level confusion matrices (background vs plant) for
DeepLabV3 and crop-aligned U-Net on the same 160 test images, Resize(255)->CenterCrop(224)
geometry and GT masks as paired_eval.py. No classifier involved."""
import os, sys, json
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
sys.path.insert(0, r"d:/Reasearch/Resnet50test/src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentation_dataset_utils import build_binary_mask
from recompute_cms_models import UNet
dev = torch.device("cuda"); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
R = r"d:/Reasearch/Resnet50test"; TEST = os.path.join(R, "tomato final dataset", "test")
FIG = os.path.join(R, "paper_report", "figures"); DATA = os.path.dirname(os.path.abspath(__file__))
C = ["developing", "flowering", "fruiting", "seeding"]; nz = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
coco = json.load(open(os.path.join(TEST, "_annotations.coco.json"), encoding="utf-8"))
f2i = {im["file_name"]: im for im in coco["images"]}; anns = {}
for a in coco["annotations"]: anns.setdefault(a["image_id"], []).append(a)
samples = [(os.path.join(TEST, c, f), c, f2i[f]) for c in C for f in sorted(os.listdir(os.path.join(TEST, c)))
           if f.lower().endswith((".jpg",".jpeg",".png")) and f in f2i and anns.get(f2i[f]["id"])]
geo = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224)]); tt = transforms.ToTensor()
def gt(e):
    m = Image.fromarray(build_binary_mask(e["height"], e["width"], anns[e["id"]]) * 255)
    m = transforms.functional.center_crop(transforms.functional.resize(m, 255, interpolation=transforms.InterpolationMode.NEAREST), 224)
    return (np.array(m) > 127).astype(np.uint8)
def pixel_cm(p, t):  # rows = true [background, plant], cols = predicted
    return np.array([[np.sum((t == 0) & (p == 0)), np.sum((t == 0) & (p == 1))],
                     [np.sum((t == 1) & (p == 0)), np.sum((t == 1) & (p == 1))]], dtype=np.int64)
def metrics(cm):
    tn, fp, fn, tp = cm.ravel().astype(float)
    return {"pixel_accuracy": (tp + tn) / cm.sum(), "precision_plant": tp / (tp + fp), "recall_plant": tp / (tp + fn),
            "specificity_background": tn / (tn + fp), "iou_plant": tp / (tp + fp + fn), "iou_background": tn / (tn + fp + fn),
            "dice_plant": 2 * tp / (2 * tp + fp + fn)}

dl = deeplabv3_resnet50(weights=None, aux_loss=True)
dl.classifier[4] = nn.Conv2d(256, 2, 1); dl.aux_classifier[4] = nn.Conv2d(256, 2, 1)
dl.load_state_dict(torch.load(os.path.join(R, "checkpoints_segmentation", "deeplabv3_leaf_seg_best.pt"), map_location=dev, weights_only=False)["model_state_dict"]); dl = dl.to(dev).eval()
un = UNet().to(dev); un.load_state_dict(torch.load(r"d:/Reasearch/Unet/export/unet_tomato_final.pth", map_location=dev, weights_only=True)); un.eval()
models = {"DeepLabV3-ResNet50": lambda x: dl(x)["out"].argmax(1), "U-Net (crop-aligned)": lambda x: (torch.sigmoid(un(x)) > 0.5)[:, 0]}
cms = {m: {k: np.zeros((2, 2), np.int64) for k in C + ["all"]} for m in models}
img_iou = {m: [] for m in models}; img_cms = {m: [] for m in models}
with torch.no_grad():
    for path, c, e in samples:
        x = nz(tt(geo(Image.open(path).convert("RGB")))).unsqueeze(0).to(dev); t = gt(e)
        for m, fn in models.items():
            cm = pixel_cm(fn(x)[0].cpu().numpy().astype(np.uint8), t)
            cms[m][c] += cm; cms[m]["all"] += cm
            img_iou[m].append((cm[1, 1] + 1e-6) / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-6))
            img_cms[m].append(cm)

out = {}
for m in models:
    out[m] = {k: {"cm": cms[m][k].tolist(), **metrics(cms[m][k])} for k in C + ["all"]}
    out[m]["mean_per_image_iou_plant"] = float(np.mean(img_iou[m]))
    # per-image mean +/- SD (the aggregation used in leaf_segmentation.ipynb's test summary)
    e = 1e-6; a = np.array(img_cms[m], dtype=float); tn, fp, fn, tp = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
    per_img = {"pixel_accuracy": (tp + tn) / a.sum((1, 2)), "precision_plant": (tp + e) / (tp + fp + e),
               "recall_plant": (tp + e) / (tp + fn + e), "f1_plant": (2 * tp + e) / (2 * tp + fp + fn + e)}
    out[m]["per_image_mean_sd"] = {k: [float(v.mean()), float(v.std())] for k, v in per_img.items()}
    print("   per-image mean+/-SD: " + "  ".join(f"{k}={v.mean():.4f}+/-{v.std():.4f}" for k, v in per_img.items()))
    print(f"== {m}  (n={len(samples)} images, {cms[m]['all'].sum():,} pixels)  mean per-image plant IoU {np.mean(img_iou[m]):.4f}")
    print("   rows=true [bg, plant], cols=pred\n", cms[m]["all"])
    for k in C + ["all"]:
        print(f"   {k:<11}" + "  ".join(f"{n}={v:.4f}" for n, v in metrics(cms[m][k]).items()))
json.dump(out, open(os.path.join(DATA, "seg_pixel_cms.json"), "w"), indent=1)

LBL = ["Background", "Plant"]
def panel(ax, cm, title):
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]:.2f}%\n({cm[i, j]:,} px)", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks([0, 1], LBL); ax.set_yticks([0, 1], LBL); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=9.5)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, m in zip(axes, models):
    s = out[m]["all"]
    panel(ax, cms[m]["all"], f"{m}\npixel acc {s['pixel_accuracy']*100:.2f}% | plant IoU {s['iou_plant']*100:.2f}% | Dice {s['dice_plant']*100:.2f}%")
fig.suptitle(f"Segmentation-only pixel confusion matrices ({len(samples)} test images, 224x224, row-normalised)", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_segmentation_pixel_confusion_matrices.png"), dpi=300, facecolor="white")

for m, fname in zip(models, ["seg_pixel_cm_deeplabv3.png", "seg_pixel_cm_unet.png"]):
    fig, ax = plt.subplots(figsize=(5.6, 5)); s = out[m]["per_image_mean_sd"]
    panel(ax, cms[m]["all"], f"{m} - segmentation only\npixel counts summed over {len(samples)} test images; per-image mean ± SD:\n"
          f"acc {s['pixel_accuracy'][0]:.4f}±{s['pixel_accuracy'][1]:.4f} | P {s['precision_plant'][0]:.4f}±{s['precision_plant'][1]:.4f}\n"
          f"R {s['recall_plant'][0]:.4f}±{s['recall_plant'][1]:.4f} | F1 {s['f1_plant'][0]:.4f}±{s['f1_plant'][1]:.4f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=300, facecolor="white")

fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
for r, m in enumerate(models):
    for ax, k in zip(axes[r], C):
        n = sum(1 for s in samples if s[1] == k)
        panel(ax, cms[m][k], f"{m} - {k} (n={n})\nplant IoU {out[m][k]['iou_plant']*100:.2f}%")
fig.suptitle("Segmentation-only pixel confusion matrices per growth stage (row-normalised)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_segmentation_pixel_confusion_matrices_by_stage.png"), dpi=300, facecolor="white")
