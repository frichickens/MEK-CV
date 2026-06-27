"""ResNet model factory. ImageNet-pretrained backbone + dropout-Linear head."""
import torch.nn as nn
from torchvision import models


# (factory_fn, default pretrained weights)
# ResNet-50 uses V2 weights (~+3 ImageNet pts over V1, transfers down).
_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(arch: str, num_classes: int, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Pick one of {list(_REGISTRY)}.")
    factory, weights = _REGISTRY[arch]
    net = factory(weights=weights if pretrained else None)
    in_feats = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feats, num_classes),
    )
    return net


AVAILABLE_ARCHS = list(_REGISTRY.keys())
