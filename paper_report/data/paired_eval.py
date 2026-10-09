"""Inference-only: per-image DeepLabV3 vs crop-aligned U-Net on the same 160 test images,
same Resize(255)->CenterCrop(224) geometry and same GT masks (deeplabv3_pipeline_evaluation.ipynb
cells 4/8/10/12 + unet_leaf_pipeline_comparison crop-aligned variant)."""
import os, sys, json
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from sklearn.metrics import confusion_matrix
sys.path.insert(0, r"d:/Reasearch/Resnet50test"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentation_dataset_utils import build_binary_mask
from recompute_cms_models import UNet
dev = torch.device("cuda"); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
R = r"d:/Reasearch/Resnet50test"; TEST = os.path.join(R, "tomato final dataset", "test")
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
def scores(p, t):
    i = np.logical_and(p, t).sum(); u = np.logical_or(p, t).sum(); e = 1e-6
    fg = (i + e) / (u + e); dice = (2*i + e) / (p.sum() + t.sum() + e)
    ib = np.logical_and(p == 0, t == 0).sum(); ub = np.logical_or(p == 0, t == 0).sum()
    return float(fg), float(dice), float((fg + ib/ub) / 2) if ub else float(fg)
def clf(path):
    m = models.resnet50(weights=None); m.fc = nn.Linear(2048, 4)
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck); return m.to(dev).eval()
dl = deeplabv3_resnet50(weights=None, aux_loss=True)
dl.classifier[4] = nn.Conv2d(256, 2, 1); dl.aux_classifier[4] = nn.Conv2d(256, 2, 1)
dl.load_state_dict(torch.load(os.path.join(R, "checkpoints_segmentation", "deeplabv3_leaf_seg_best.pt"), map_location=dev, weights_only=False)["model_state_dict"]); dl = dl.to(dev).eval()
un = UNet().to(dev); un.load_state_dict(torch.load(r"d:/Reasearch/Unet/export/unet_tomato_final.pth", map_location=dev, weights_only=True)); un.eval()
dl_clf = clf(os.path.join(R, "export_leaf_pipeline", "resnet50_leaf_classifier.pth"))
un_clf = clf(os.path.join(R, "checkpoints_masked", "resnet50_masked_best.pt"))
rows = []
with torch.no_grad():
    for path, c, e in samples:
        raw = tt(geo(Image.open(path).convert("RGB"))).unsqueeze(0).to(dev); x = nz(raw[0]).unsqueeze(0); t = gt(e)
        r = {"file": os.path.basename(path), "true": c}
        for name, mask_fn, cl in [("dlv3", lambda: dl(x)["out"].argmax(1, keepdim=True).float(), dl_clf),
                                  ("unet", lambda: (torch.sigmoid(un(x)) > 0.5).float(), un_clf)]:
            m = mask_fn(); pm = m[0, 0].cpu().numpy().astype(np.uint8)
            r[name + "_fg_iou"], r[name + "_dice"], r[name + "_both_iou"] = scores(pm, t)
            r[name + "_pred"] = C[int(cl(nz((raw * m)[0]).unsqueeze(0)).argmax(1))]
        rows.append(r)
for n in ["dlv3", "unet"]:
    v = [np.mean([r[n+k] for r in rows]) for k in ["_fg_iou", "_dice", "_both_iou"]] + [np.mean([r[n+"_pred"] == r["true"] for r in rows])]
    print(n, "fgIoU %.4f Dice %.4f bothIoU %.4f acc %.5f" % tuple(v))
    print(confusion_matrix([r["true"] for r in rows], [r[n+"_pred"] for r in rows], labels=C))
json.dump(rows, open(sys.argv[1], "w"), indent=0)
