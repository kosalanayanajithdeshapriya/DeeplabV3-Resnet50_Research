"""Leaf Pixel Fraction (LPF = predicted plant pixels / total pixels) per growth stage for
DeepLabV3 and U-Net, plus ground-truth LPF from the COCO masks, over train/valid/test.
DeepLabV3, U-Net (crop-aligned) and GT share the Resize(255)->CenterCrop(224) geometry used in
seg_pixel_cms.py; "unet_native" uses U-Net's own direct 224x224 resize (unet_lpf_extractor.py),
which reproduces d:/Reasearch/Unet/export/lpf_full_dataset.csv."""
import os, sys, json, csv
import numpy as np, torch, torch.nn as nn, cv2
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
sys.path.insert(0, r"d:/Reasearch/Resnet50test/src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentation_dataset_utils import build_binary_mask
from recompute_cms_models import UNet
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.deterministic = True
R = r"d:/Reasearch/Resnet50test"; DS = os.path.join(R, "tomato final dataset"); DATA = os.path.dirname(os.path.abspath(__file__))
C = ["developing", "flowering", "fruiting", "seeding"]; SPLITS = ["train", "valid", "test"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]; nz = transforms.Normalize(MEAN, STD); tt = transforms.ToTensor()
geo = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224)])

dl = deeplabv3_resnet50(weights=None, aux_loss=True)
dl.classifier[4] = nn.Conv2d(256, 2, 1); dl.aux_classifier[4] = nn.Conv2d(256, 2, 1)
dl.load_state_dict(torch.load(os.path.join(R, "checkpoints_segmentation", "deeplabv3_leaf_seg_best.pt"), map_location=dev, weights_only=False)["model_state_dict"]); dl = dl.to(dev).eval()
un = UNet().to(dev); un.load_state_dict(torch.load(r"d:/Reasearch/Unet/export/unet_tomato_final.pth", map_location=dev, weights_only=True)); un.eval()

def gt_lpf(e, anns):
    if not anns: return None
    m = Image.fromarray(build_binary_mask(e["height"], e["width"], anns) * 255)
    m = transforms.functional.center_crop(transforms.functional.resize(m, 255, interpolation=transforms.InterpolationMode.NEAREST), 224)
    return float((np.array(m) > 127).mean())

def native_tensor(path):  # unet_lpf_extractor._preprocess
    img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)

rows = []
with torch.no_grad():
    for sp in SPLITS:
        coco = json.load(open(os.path.join(DS, sp, "_annotations.coco.json"), encoding="utf-8"))
        f2i = {im["file_name"]: im for im in coco["images"]}; anns = {}
        for a in coco["annotations"]: anns.setdefault(a["image_id"], []).append(a)
        for c in C:
            for f in sorted(os.listdir(os.path.join(DS, sp, c))):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
                p = os.path.join(DS, sp, c, f); x = nz(tt(geo(Image.open(p).convert("RGB")))).unsqueeze(0).to(dev)
                e = f2i.get(f)
                rows.append({"split": sp, "growth_stage": c, "filename": f,
                             "gt_lpf": gt_lpf(e, anns.get(e["id"])) if e else None,
                             "deeplabv3_lpf": float(dl(x)["out"].argmax(1).float().mean()),
                             "unet_lpf": float((torch.sigmoid(un(x)) > 0.5).float().mean()),
                             "unet_native_lpf": float((torch.sigmoid(un(native_tensor(p))) > 0.5).float().mean())})

with open(os.path.join(DATA, "lpf_per_image.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

K = ["gt_lpf", "deeplabv3_lpf", "unet_lpf", "unet_native_lpf"]
def stats(sub):
    out = {"n": len(sub)}
    for k in K:
        v = np.array([r[k] for r in sub if r[k] is not None])
        out[k] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
                  "min": float(v.min()), "max": float(v.max())}
    for k in ["deeplabv3_lpf", "unet_lpf"]:  # error vs GT on images that have GT
        d = np.array([r[k] - r["gt_lpf"] for r in sub if r["gt_lpf"] is not None])
        out[k]["bias_vs_gt"] = float(d.mean()); out[k]["mae_vs_gt"] = float(np.abs(d).mean())
    return out

summary = {}
for scope, sel in [("all_splits", lambda r: True), ("test", lambda r: r["split"] == "test")]:
    summary[scope] = {c: stats([r for r in rows if sel(r) and r["growth_stage"] == c]) for c in C}
    summary[scope]["all"] = stats([r for r in rows if sel(r)])
json.dump(summary, open(os.path.join(DATA, "lpf_by_stage.json"), "w"), indent=1)

for scope in summary:
    print(f"\n== {scope}   LPF mean ± SD (median)")
    print(f"{'stage':<11}{'n':>5}  " + "".join(f"{k:>26}" for k in K) + f"{'DL MAE':>9}{'UNet MAE':>10}")
    for c in C + ["all"]:
        s = summary[scope][c]
        print(f"{c:<11}{s['n']:>5}  " + "".join(f"{s[k]['mean']:>10.4f} ± {s[k]['sd']:.4f} ({s[k]['median']:.4f})" for k in K)
              + f"{s['deeplabv3_lpf']['mae_vs_gt']:>9.4f}{s['unet_lpf']['mae_vs_gt']:>10.4f}")
