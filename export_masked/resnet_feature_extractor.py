"""Standalone loader for the MASKED-TRAINED tomato growth-stage ResNet50 checkpoint,
used as a background-reliance comparison point alongside export/resnet_feature_extractor.py.
Copy-paste this file into another project with no other dependency on training code.
"""
import torch
import torch.nn as nn
from torchvision import transforms, models

NUM_CLASSES = 4
CLASS_NAMES = ['developing', 'flowering', 'fruiting', 'seeding']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_resnet_feature_extractor(checkpoint_path, device=None):
    """Loads the masked-trained checkpoint and returns it as a 2048-dim feature extractor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model
