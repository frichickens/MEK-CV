"""ResNet + CAM-style head used by MEK.

The original MEK paper uses Swin-T whose `forward_features` produces 7×7 patch
embeddings at 224×224 input. ResNet's `layer4` output is also 7×7 at 224×224
input, so the same CAM-head pattern transplants cleanly:

    encoder(x)  →  feat   [B, C, 7, 7]              (C = 512 for r18/r34, 2048 for r50)
    1×1 conv    →  hm     [B, K, 7, 7]              per-class attention maps
    GAP         →  logits [B, K]

`forward()` returns (logits, hm) so the trainer can compute both the
cross-entropy / re-balanced-smooth-label loss on logits and the flip-consistency
ACLoss on hm.
"""
import hashlib
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


# SHA-256 digests of the trusted, published face-recognition .pkl exports we permit
# unpickling. Pinned in SOURCE on purpose: a hash supplied at runtime (CLI/arg) provides
# NO security — an attacker who swaps the .pkl can supply a matching hash too. The only
# meaningful pin is one the caller cannot control. To trust a new .pkl, add its digest
# here in a reviewed code change, or (preferred) convert it to .pth and load that.
#   resnet50_ft_weight.pkl  — cydonia999/VGGFace2-pytorch
_TRUSTED_FACE_PKL_SHA256 = {
    "9a954e4046df21aea62b46c032de4b56f7230dfc3705b6b1bda348983e0eb26e",
}


def _load_face_state(path: str):
    """Read a face-recognition checkpoint. Prefer a torch .pth/.pt export — loaded with
    weights_only=True, no arbitrary-code risk. A .pkl (e.g. the canonical numpy-dict
    export from cydonia999/VGGFace2-pytorch) is unpickled ONLY when its SHA-256 is in the
    source-pinned `_TRUSTED_FACE_PKL_SHA256` allow-list; it fails closed otherwise.
    """
    if not path or not os.path.exists(path):
        return None

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest not in _TRUSTED_FACE_PKL_SHA256:
            raise ValueError(
                f"Refusing to unpickle {path!r}: its SHA-256 ({digest}) is not in the "
                "source-pinned trusted set (mek/model.py:_TRUSTED_FACE_PKL_SHA256). "
                "Convert the weights to .pth and load that (weights_only=True, safe), or "
                "add this digest to the allow-list in a reviewed change if you trust the file."
            )
        raw = pickle.loads(data)                          # digest in source-pinned allow-list
        return {k: torch.as_tensor(np.asarray(v)) for k, v in raw.items()}

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    except Exception as e:
        print(f"WARNING: could not read face weights {path!r}: {e}")
        return None


def build_face_backbone(arch: str, path: str):
    """Return a torchvision `arch` net initialized from face-recognition weights
    (e.g. VGGFace2), or None to let the caller fall back to ImageNet. Robust to
    key-naming differences: matches by name+shape, then by order+shape, keeps
    whichever fills more encoder tensors, and accepts the face init only if
    >=80% of encoder tensors were filled."""
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Pick one of {list(_REGISTRY)}.")
    factory, _, _ = _REGISTRY[arch]

    sd = _load_face_state(path)
    if sd is None:
        print(f"Face weights not found at {path!r} — using ImageNet init.")
        return None
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    net = factory(weights=None)
    tgt = net.state_dict()
    # Exclude the classifier (fc.*) and BN step counters (num_batches_tracked) —
    # the latter aren't weights and aren't present in face exports.
    enc_keys = [k for k in tgt if not k.startswith("fc.") and not k.endswith("num_batches_tracked")]

    # strategy 1 — match by name + shape (works when the port uses torchvision names)
    by_name = {k: sd[k] for k in enc_keys
               if k in sd and tuple(sd[k].shape) == tuple(tgt[k].shape)}

    # strategy 2 — match by order + shape (works across different naming schemes)
    src = [v for v in sd.values() if hasattr(v, "shape")]
    by_order, si = {}, 0
    for k in enc_keys:
        while si < len(src) and tuple(src[si].shape) != tuple(tgt[k].shape):
            si += 1
        if si < len(src):
            by_order[k] = src[si]; si += 1

    best = by_name if len(by_name) >= len(by_order) else by_order
    strat = "name" if best is by_name else "order"
    net.load_state_dict({**tgt, **best}, strict=True)
    print(f"Face init ({arch}): filled {len(best)}/{len(enc_keys)} encoder tensors (match by {strat}).")
    if len(best) < 0.8 * len(enc_keys):
        print("WARNING: <80% of encoder tensors matched — keys differ too much; "
              "falling back to ImageNet init.")
        return None
    return net


class MEKResNet(nn.Module):
    def __init__(self, arch: str, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.4,
                 face_weights: str = None):
        super().__init__()
        if arch not in _REGISTRY:
            raise ValueError(f"Unknown arch: {arch}. Pick one of {list(_REGISTRY)}.")
        factory, weights, feat_dim = _REGISTRY[arch]
        # Prefer a face-recognition (e.g. VGGFace2/MS-Celeb) init — this is the paper's
        # setup and the single biggest lever on FER accuracy. Falls back to ImageNet
        # when no/unmatched face weights are given.
        net = build_face_backbone(arch, face_weights) if face_weights else None
        # Record the init that actually took effect (face loader may fall back to
        # ImageNet on a missing/unmatched file) so callers can log it to W&B.
        self.backbone_source = "face-weights" if net is not None else ("imagenet" if pretrained else "random")
        if net is None:
            net = factory(weights=weights if pretrained else None)
        # Drop avgpool + fc; keep only the spatial feature extractor.
        self.encoder = nn.Sequential(*list(net.children())[:-2])
        self.feat_dim    = feat_dim
        self.num_classes = num_classes
        self.dropout    = nn.Dropout2d(p=dropout)
        # CAM head: per-class 1×1 conv. With BN before the head we keep the
        # pretrained feature scale stable when fine-tuning.
        self.bn         = nn.BatchNorm2d(feat_dim)
        self.classifier = nn.Conv2d(feat_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)            # [B, C, H, W]   H=W=7 at 224 input
        feat = self.bn(feat)
        feat = self.dropout(feat)
        hm = self.classifier(feat)        # [B, K, H, W]   per-class attention maps
        logits = F.adaptive_avg_pool2d(hm, 1).flatten(1)   # [B, K]
        return logits, hm


AVAILABLE_ARCHS = list(_REGISTRY.keys())
