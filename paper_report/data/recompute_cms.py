"""Inference-only re-scoring of saved checkpoints to recover confusion matrices.
Replicates unet_cross_validation.ipynb cells 8/20/22 and resnet50_masked_comparison.ipynb cell 13."""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
sys.path.insert(0, r"d:/Reasearch/Resnet50test/src")
from masked_comparison_utils import SplitFirstFolderDataset
from torch.utils.data import DataLoader

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
C = ["developing", "flowering", "fruiting", "seeding"]
M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class DoubleConv(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.block = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(s, x): return s.block(x)
class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.enc1, s.enc2, s.enc3, s.enc4 = DoubleConv(3, c), DoubleConv(c, 2*c), DoubleConv(2*c, 4*c), DoubleConv(4*c, 8*c)
        s.pool = nn.MaxPool2d(2); s.bottleneck = DoubleConv(8*c, 16*c)
        s.up4, s.dec4 = nn.ConvTranspose2d(16*c, 8*c, 2, stride=2), DoubleConv(16*c, 8*c)
        s.up3, s.dec3 = nn.ConvTranspose2d(8*c, 4*c, 2, stride=2), DoubleConv(8*c, 4*c)
        s.up2, s.dec2 = nn.ConvTranspose2d(4*c, 2*c, 2, stride=2), DoubleConv(4*c, 2*c)
        s.up1, s.dec1 = nn.ConvTranspose2d(2*c, c, 2, stride=2), DoubleConv(2*c, c)
        s.out_conv = nn.Conv2d(c, 1, 1)
    def forward(s, x):
        e1 = s.enc1(x); e2 = s.enc2(s.pool(e1)); e3 = s.enc3(s.pool(e2)); e4 = s.enc4(s.pool(e3))
        b = s.bottleneck(s.pool(e4))
        d4 = s.dec4(torch.cat([s.up4(b), e4], 1)); d3 = s.dec3(torch.cat([s.up3(d4), e3], 1))
        d2 = s.dec2(torch.cat([s.up2(d3), e2], 1)); d1 = s.dec1(torch.cat([s.up1(d2), e1], 1))
        return s.out_conv(d1)

def resnet(path):
    m = models.resnet50(weights=None); m.fc = nn.Linear(2048, 4)
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck)
    return m.to(dev).eval()

out = {}
# ---- U-Net CV folds -> fixed masked ResNet50, fixed 160-image test ----
clf = resnet(r"d:/Reasearch/Resnet50test/checkpoints_masked/resnet50_masked_best.pt")
test = pd.read_csv(r"d:/Reasearch/Unet/processed/test/stage_mapping.csv")
crop = transforms.Compose([transforms.Resize(int(224*1.14)), transforms.CenterCrop(224)])
tt, nz = transforms.ToTensor(), transforms.Normalize(M, S)
imgs = [tt(crop(Image.open(os.path.join(r"d:/Reasearch/Unet/processed/test/images", f)).convert("RGB"))) for f in test.filename]
labels = list(test.growth_stage)
fold_cms = []
with torch.no_grad():
    for k in range(5):
        u = UNet().to(dev); u.load_state_dict(torch.load(rf"d:/Reasearch/Unet/checkpoints/cv_folds/unet_fold{k}.pth", map_location=dev, weights_only=True)); u.eval()
        preds = []
        for raw in imgs:
            m = (torch.sigmoid(u(nz(raw).unsqueeze(0).to(dev))) > 0.5).float()
            x = nz((raw.unsqueeze(0).to(dev) * m).squeeze(0)).unsqueeze(0)
            preds.append(C[int(clf(x).argmax(1))])
        cm = confusion_matrix(labels, preds, labels=C)
        fold_cms.append(cm.tolist())
        print(f"U-Net fold {k+1}: acc={accuracy_score(labels, preds):.5f}")
        print(cm)
out["unet_cv_fold_cms"] = fold_cms
out["unet_cv_pooled_cm"] = np.sum(fold_cms, axis=0).tolist()

# ---- ResNet50 background ablation 2x2 ----
ev = transforms.Compose([transforms.Resize(int(224*1.14)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(M, S)])
R = r"d:/Reasearch/Resnet50test"
models_ = {"original": resnet(os.path.join(R, "export", "resnet50_tomato_final.pth")), "masked": clf}
data_ = {"original": os.path.join(R, "tomato final dataset"), "masked": os.path.join(R, "tomato final dataset_masked")}
for tm, mdl in models_.items():
    for ed, root in data_.items():
        ds = SplitFirstFolderDataset(root, "test", ev, classes=C)
        P, L = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=32):
                P += mdl(x.to(dev)).argmax(1).cpu().tolist(); L += y.tolist()
        cm = confusion_matrix(L, P, labels=range(4))
        rep = classification_report(L, P, labels=range(4), target_names=C, output_dict=True, zero_division=0)
        out[f"resnet_train_{tm}_eval_{ed}"] = {"acc": accuracy_score(L, P), "cm": cm.tolist(),
            "per_class": {c: [rep[c]["precision"], rep[c]["recall"], rep[c]["f1-score"]] for c in C},
            "macro": [rep["macro avg"]["precision"], rep["macro avg"]["recall"], rep["macro avg"]["f1-score"]]}
        print(f"ResNet train={tm} eval={ed}: acc={accuracy_score(L, P):.5f}"); print(cm)
json.dump(out, open(sys.argv[1], "w"), indent=1)
