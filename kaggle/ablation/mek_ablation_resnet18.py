"""
MEK ablation (RSL × RAC) + ResNet-18 (MS-Celeb-1M) — single-file Kaggle script (RAF-DB | FER-2013).

Reproduces the MEK paper's two-module ablation (Zhang et al., NeurIPS 2023, arxiv 2310.19636):
  "Ablation study of our proposed two modules re-balanced attention consistency (RAC)
   and re-balanced smooth labels (RSL). Both of the two modules can improve the
   performance based on the baseline, while they can cooperate to achieve the best."

Trains the 2×2 back-to-back on the SAME data and backbone, isolating each module:

  baseline   -- plain CrossEntropy, no attention consistency        (neither module)
  rsl        -- + Re-balanced Smooth Labels                         (RSL only)
  rac        -- + Re-balanced Attention Consistency (flip AC loss)  (RAC only)
  rsl_rac    -- RSL + RAC                                            (full MEK, best)

Backbone is ResNet-18 (the paper's ablation backbone). For a faithful reproduction add
the MS-Celeb-1M checkpoint (resnet18_msceleb.pth — the paper's init) as a Kaggle dataset
and point FACE_WEIGHTS_PATH at it; without it the study still runs (on ImageNet) and the
relative module gains still hold.

This is the standalone Kaggle counterpart of ablation/train.py.
"""

# ════════════════════════════════════════════════════════════════════
# User toggles
# ════════════════════════════════════════════════════════════════════
DATASET     = "rafdb"        # "rafdb" (paper's ablation) or "fer2013"
USE_WANDB   = False           # set True to log this run to Weights & Biases

FER_ROOT = "/kaggle/input/datasets/msambare/fer2013"
RAF_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"

# MS-Celeb-1M face-recognition backbone — the paper's init (the biggest accuracy lever).
# Add resnet18_msceleb.pth as a Kaggle dataset and set the path. The loader tolerates
# key-naming differences and FALLS BACK to ImageNet if missing/unmatched, so it never
# hard-fails. NB: the ablation's *relative* module gains hold either way, but the
# absolute numbers only match the paper with the MS-Celeb init.
USE_FACE_BACKBONE = True
FACE_WEIGHTS_PATH = "/kaggle/input/resnet18-msceleb/resnet18_msceleb.pth"
# A .pth (resnet18_msceleb.pth) loads safely with weights_only=True and needs NO hash —
# leave this empty. It is MANDATORY only if you instead load a .pkl:
#   import hashlib; print(hashlib.sha256(open(FACE_WEIGHTS_PATH,'rb').read()).hexdigest())
FACE_WEIGHTS_SHA256 = ""

# ════════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════════
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

from torchmetrics import Accuracy, Precision, Recall, F1Score


# ════════════════════════════════════════════════════════════════════
# Arch (fixed to ResNet-18 — the paper's ablation backbone)
# ════════════════════════════════════════════════════════════════════
ARCH         = "resnet18"
ARCH_FACTORY = models.resnet18
ARCH_WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1     # only used if face weights missing
FEAT_DIM     = 512
DROPOUT      = 0.40


# ════════════════════════════════════════════════════════════════════
# Dataset config (both forced to 224 — MEK needs a 7×7 attention map)
# ════════════════════════════════════════════════════════════════════
if DATASET == "fer2013":
    DATA_ROOT          = FER_ROOT
    EPOCHS             = 80
    ROTATION_DEG       = 10
    COLOR_JITTER       = 0.2
    RANDOM_ERASING_P   = 0.5
elif DATASET == "rafdb":
    DATA_ROOT          = RAF_ROOT
    EPOCHS             = 60
    ROTATION_DEG       = 15
    COLOR_JITTER       = 0.3
    RANDOM_ERASING_P   = 0.25
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}.")

TRAIN_DIR   = f"{DATA_ROOT}/train"
TEST_DIR    = f"{DATA_ROOT}/test"
NUM_CLASSES = 7
IMG_SIZE    = 224
CROP_SIZE   = 224
CROP_PADDING = 8

# Paper's ResNet recipe (Adam @ 1e-4, ε=0.1, λ_flip=2). The ablation toggles whether
# RSL (the ε-smoothed re-balanced loss) and RAC (the λ_flip AC term) are active.
BATCH_SIZE          = 64
LR                  = 1e-4
LABEL_SMOOTH        = 0.1          # ε for RSL (only used by the rsl / rsl_rac variants)
FLIP_LOSS_WEIGHT    = 2.0          # λ for RAC (zeroed for the baseline / rsl variants)
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 5.0
VAL_SPLIT           = 0.1
SEED                = 42
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 25
USE_WEIGHTED_SAMPLER = True

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The paper's 2×2: each module independently toggled on the baseline.
ALL_VARIANTS = {
    "baseline": {"rsl": False, "rac": False},   # plain CE, no AC
    "rsl":      {"rsl": True,  "rac": False},   # + re-balanced smooth labels
    "rac":      {"rsl": False, "rac": True},    # + re-balanced attention consistency
    "rsl_rac":  {"rsl": True,  "rac": True},    # full MEK (both modules)
}
# Subset of the 2×2 to run in THIS kernel (lets the study be split across GPU
# sessions — e.g. {baseline, rsl_rac} on one, {rsl, rac} on another — and merged
# afterwards; all cells share the same seed/data/backbone so they stay comparable).
# Default = all four (single-kernel full ablation). Set by prep_ablation.py.
VARIANT_SUBSET = ["baseline", "rsl", "rac", "rsl_rac"]
VARIANTS = {k: ALL_VARIANTS[k] for k in VARIANT_SUBSET}

print(f"Device:   {DEVICE}")
print(f"Method:   MEK ablation (RSL × RAC)  — arxiv 2310.19636")
print(f"Arch:     {ARCH}  (dropout={DROPOUT})")
print(f"Dataset:  {DATASET}  →  {DATA_ROOT}")
print(f"Backbone: {'MS-Celeb-1M @ ' + FACE_WEIGHTS_PATH if USE_FACE_BACKBONE else 'ImageNet'}")
print(f"Train:    lr={LR:.2e} bs={BATCH_SIZE} epochs={EPOCHS} ε_rsl={LABEL_SMOOTH} λ_rac={FLIP_LOSS_WEIGHT}")
print(f"Variants: {list(VARIANTS.keys())}\n")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════
# Data — built ONCE and reused across all variants (same data for a fair ablation).
# Train loader yields (img, label, img_hflip); the deterministic flip is what the
# AC (RAC) loss consumes, so RandomHorizontalFlip is deliberately NOT used.
# ════════════════════════════════════════════════════════════════════
set_seed(SEED)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomCrop(CROP_SIZE, padding=CROP_PADDING),
    transforms.RandomRotation(ROTATION_DEG),
    transforms.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=RANDOM_ERASING_P, scale=(0.02, 0.2)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class PairedFlipDataset(Dataset):
    """Wraps any (img, label) dataset and additionally returns H-flipped img."""
    def __init__(self, base):
        self.base = base
        self._flip = transforms.RandomHorizontalFlip(p=1.0)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, self._flip(img)


full_aug  = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

CLASSES = full_aug.classes
print(f"Classes:  {CLASSES}")

idx       = np.random.permutation(len(full_aug))
val_size  = int(len(full_aug) * VAL_SPLIT)
val_idx   = idx[:val_size]
train_idx = idx[val_size:]

train_dataset = PairedFlipDataset(Subset(full_aug, train_idx))
val_dataset   = Subset(full_eval, val_idx)

train_labels = np.array(full_aug.targets)[train_idx]
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
print(f"Counts:   {dict(zip(CLASSES, class_counts.astype(int)))}")
print(f"Sizes:    Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")

_inv = 1.0 / np.maximum(class_counts, 1.0)
balance_weights = torch.from_numpy(_inv * (NUM_CLASSES / _inv.sum())).float().to(DEVICE)
print(f"Balance weights (mean=1): {balance_weights.cpu().numpy().round(3).tolist()}")

if USE_WEIGHTED_SAMPLER:
    sample_w = torch.from_numpy(_inv[train_labels.astype(int)]).double()
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(train_labels), replacement=True)
    shuffle  = False
else:
    sampler, shuffle = None, True

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ════════════════════════════════════════════════════════════════════
# Model — ResNet-18 encoder (optionally MS-Celeb-1M) + 1×1 conv CAM head
# ════════════════════════════════════════════════════════════════════
def _load_state_any(path):
    """Read a face-recognition checkpoint: a torch .pth/.pt (e.g. resnet18_msceleb.pth)
    or a numpy-dict .pkl. .pth/.pt uses weights_only=True (safe); a .pkl is a plain
    pickle (arbitrary-code risk) gated by the MANDATORY FACE_WEIGHTS_SHA256 pin below."""
    import os, pickle, hashlib
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = f.read()
        if not FACE_WEIGHTS_SHA256:
            raise RuntimeError(
                "FACE_WEIGHTS_SHA256 must be set before loading a .pkl face-weights file. "
                "Compute it with:\n"
                "  python -c \"import hashlib;print(hashlib.sha256(open(PATH,'rb').read()).hexdigest())\"\n"
                "or use a .pth export instead (weights_only=True, no hash needed)."
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != FACE_WEIGHTS_SHA256:
            raise ValueError(f"Face-weights SHA-256 mismatch ({digest}) — refusing to unpickle {path!r}.")
        raw = pickle.loads(data)                          # integrity-verified weights file
        return {k: torch.as_tensor(np.asarray(v)) for k, v in raw.items()}
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    except Exception as e:
        print(f"WARNING: could not read face weights {path!r}: {e}")
        return None


def build_face_backbone():
    """Return a torchvision resnet18 initialized from MS-Celeb-1M (or any face) weights,
    or None to fall back to ImageNet. Robust to key naming: matches by name+shape, then
    order+shape, accepts only if >=80% of encoder tensors are filled.
    (resnet18_msceleb.pth maps 100/100 by name.)"""
    if not USE_FACE_BACKBONE:
        return None
    sd = _load_state_any(FACE_WEIGHTS_PATH)
    if sd is None:
        print(f"Face weights not found at {FACE_WEIGHTS_PATH!r} — using ImageNet init.")
        return None
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    net = ARCH_FACTORY(weights=None)
    tgt = net.state_dict()
    enc_keys = [k for k in tgt if not k.startswith("fc.") and not k.endswith("num_batches_tracked")]

    by_name = {k: sd[k] for k in enc_keys
               if k in sd and tuple(sd[k].shape) == tuple(tgt[k].shape)}

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
    print(f"Face init: filled {len(best)}/{len(enc_keys)} encoder tensors (match by {strat}).")
    if len(best) < 0.8 * len(enc_keys):
        print("WARNING: <80% of encoder tensors matched — falling back to ImageNet init.")
        return None
    return net


class MEKResNet(nn.Module):
    def __init__(self, num_classes, dropout, backbone=None):
        super().__init__()
        net = backbone if backbone is not None else ARCH_FACTORY(weights=ARCH_WEIGHTS)
        self.encoder = nn.Sequential(*list(net.children())[:-2])
        self.bn         = nn.BatchNorm2d(FEAT_DIM)
        self.drop       = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(FEAT_DIM, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.bn(feat)
        feat = self.drop(feat)
        hm = self.classifier(feat)                # [B, K, 7, 7]
        logits = F.adaptive_avg_pool2d(hm, 1).flatten(1)
        return logits, hm


# ════════════════════════════════════════════════════════════════════
# Losses — RSL (re-balanced smooth labels) and RAC (re-balanced attention consistency)
# ════════════════════════════════════════════════════════════════════
class ReBalancedLabelSmoothing(nn.Module):
    """RSL: label smoothing where the ε mass is redistributed toward minority classes."""
    def __init__(self, epsilon, balance_w):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("bw", balance_w)

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=1)
        smooth = torch.zeros_like(logits)
        smooth.scatter_(1, target.view(-1, 1), 1.0 - self.epsilon)
        mask  = (smooth == 0)
        bw    = self.bw.unsqueeze(0).expand_as(logits)
        bw_m  = bw * mask
        bw_n  = bw_m / bw_m.sum(dim=1, keepdim=True).clamp_min(1e-8)
        smooth = smooth + bw_n * self.epsilon
        return -(log_p * smooth).sum(dim=1).mean()


def make_flip_grid(h, w):
    x = torch.arange(w).view(1, -1).expand(h, -1)
    y = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x, y], 0).float().unsqueeze(0)
    grid[:, 0] = 2 * grid[:, 0] / (w - 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / (h - 1) - 1
    grid[:, 0] = -grid[:, 0]
    return grid


def ac_loss(hm_o, hm_f, flip_grid, balance_w):
    """RAC: per-class inverse-freq-weighted MSE between the attention map and the
    un-flipped attention map of the flipped input."""
    B = hm_o.size(0)
    grid = flip_grid.expand(B, -1, -1, -1).permute(0, 2, 3, 1)
    hm_unflip = F.grid_sample(hm_f, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
    fl = F.mse_loss(hm_o, hm_unflip, reduction="none").mean(dim=[-1, -2])  # [B, K]
    return (fl @ balance_w).mean()


eval_criterion = nn.CrossEntropyLoss()
_flip_grid = None    # built lazily once we know the hm spatial size (7×7), shared across variants


# ════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════
def make_metrics():
    return {
        "accuracy":  Accuracy( task="multiclass", num_classes=NUM_CLASSES).to(DEVICE),
        "precision": Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
        "recall":    Recall(   task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
        "f1":        F1Score(  task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
    }


def per_class_accuracy(confmat):
    rs = confmat.sum(axis=1)
    return np.where(rs > 0, np.diag(confmat) / np.maximum(rs, 1), 0.0)


# ════════════════════════════════════════════════════════════════════
# Train / Eval (parameterized by model / optimizer / criterion / flip weight)
# ════════════════════════════════════════════════════════════════════
def train_one_epoch(model, optimizer, scaler, criterion, flip_weight, amp_enabled):
    global _flip_grid
    model.train()
    metrics = make_metrics()
    total_loss, sum_lsr, sum_flip, total_n = 0.0, 0.0, 0.0, 0

    for img, label, img_flip in train_loader:
        img      = img.to(DEVICE,      non_blocking=True)
        label    = label.to(DEVICE,    non_blocking=True)
        img_flip = img_flip.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits_o, hm_o = model(img)
            lsr = criterion(logits_o, label)
            if flip_weight > 0:
                logits_f, hm_f = model(img_flip)
                if _flip_grid is None:
                    h, w = hm_o.shape[-2:]
                    _flip_grid = make_flip_grid(h, w).to(DEVICE)
                fl = ac_loss(hm_o, hm_f, _flip_grid, balance_weights)
            else:
                fl = torch.zeros((), device=DEVICE)
            loss = lsr + flip_weight * fl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * img.size(0)
        sum_lsr    += lsr.item()  * img.size(0)
        sum_flip   += float(fl)   * img.size(0)
        total_n    += img.size(0)
        for m in metrics.values():
            m.update(logits_o.softmax(1), label)

    out = {k: v.compute().item() for k, v in metrics.items()}
    out["loss"]      = total_loss / total_n
    out["lsr_loss"]  = sum_lsr   / total_n
    out["flip_loss"] = sum_flip  / total_n
    return out


@torch.no_grad()
def evaluate(model, loader, return_per_class=False):
    model.eval()
    metrics = make_metrics()
    total_loss, total_n = 0.0, 0
    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) if return_per_class else None

    for img, label in loader:
        img   = img.to(DEVICE,   non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        logits, _ = model(img)
        loss = eval_criterion(logits, label)
        total_loss += loss.item() * img.size(0)
        total_n    += img.size(0)
        for m in metrics.values():
            m.update(logits.softmax(1), label)
        if confmat is not None:
            preds = logits.argmax(1)
            for t, p in zip(label.cpu().numpy(), preds.cpu().numpy()):
                confmat[t, p] += 1

    out = {k: v.compute().item() for k, v in metrics.items()}
    out["loss"] = total_loss / total_n
    if confmat is not None:
        pca = per_class_accuracy(confmat)
        out["per_class_acc"]  = pca.tolist()
        out["mean_class_acc"] = float(pca.mean())
        out["confmat"]        = confmat.tolist()
    return out


# ════════════════════════════════════════════════════════════════════
# Weights & Biases (optional)
# ════════════════════════════════════════════════════════════════════
def _init_wandb(run_name, config):
    if not USE_WANDB:
        return None
    import os
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            key = None
    if not key:
        print("WARNING: WANDB_API_KEY not found (env or Kaggle Secrets) — running without W&B.")
        return None
    import wandb
    wandb.login(key=key)
    wandb.init(project="fer-emotion-recognition", name=run_name, config=config)
    return wandb


# Probe the backbone once (loads the .pth, reflects any ImageNet fallback) so the run
# config records the ACTUAL init shared by all four variants.
_probe = build_face_backbone()
BACKBONE = "MS-Celeb-1M" if _probe is not None else "ImageNet"
del _probe
print(f"Backbone: {BACKBONE}")

wb = _init_wandb(f"ablation_{ARCH}_{DATASET}", {
    "arch": ARCH, "dataset": DATASET, "method": "ablation-RSL-RAC",
    "variants": list(VARIANTS.keys()), "epochs": EPOCHS, "lr": LR,
    "batch_size": BATCH_SIZE, "label_smooth": LABEL_SMOOTH,
    "flip_loss_weight": FLIP_LOSS_WEIGHT, "optimizer": "Adam",
    "backbone": BACKBONE,
})


# ════════════════════════════════════════════════════════════════════
# One variant: fresh model + optimizer, train EPOCHS, restore best, test
# ════════════════════════════════════════════════════════════════════
def run_variant(name, use_rsl, use_rac):
    print(f"{'-'*60}\n  [{name}]  RSL={use_rsl}  RAC={use_rac}\n{'-'*60}")
    set_seed(SEED)   # same init/order across variants for a fair comparison

    model = MEKResNet(NUM_CLASSES, DROPOUT, backbone=build_face_backbone()).to(DEVICE)
    criterion = (ReBalancedLabelSmoothing(LABEL_SMOOTH, balance_weights).to(DEVICE)
                 if use_rsl else nn.CrossEntropyLoss().to(DEVICE))
    flip_weight = FLIP_LOSS_WEIGHT if use_rac else 0.0

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    amp_enabled = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    keys = ["loss", "accuracy", "precision", "recall", "f1"]
    history = {k: [] for k in keys + [f"val_{k}" for k in keys]}
    history["lsr_loss"], history["flip_loss"] = [], []

    best_val_acc, best_state, patience = 0.0, None, 0
    ckpt_path = f"ablation_{name}_{ARCH}_{DATASET}_best.pth"
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, optimizer, scaler, criterion, flip_weight, amp_enabled)
        va = evaluate(model, val_loader)
        scheduler.step()

        if va["accuracy"] > best_val_acc:
            best_val_acc = va["accuracy"]
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            patience, flag = 0, "+"
        else:
            patience += 1
            flag = " "

        for k in keys:
            history[k].append(tr[k])
            history[f"val_{k}"].append(va[k])
        history["lsr_loss"].append(tr["lsr_loss"])
        history["flip_loss"].append(tr["flip_loss"])

        if wb is not None:
            wb.log({
                f"{name}/epoch": epoch,
                f"{name}/train/loss": tr["loss"], f"{name}/train/accuracy": tr["accuracy"],
                f"{name}/train/f1": tr["f1"],
                f"{name}/train/lsr_loss": tr["lsr_loss"], f"{name}/train/flip_loss": tr["flip_loss"],
                f"{name}/val/loss": va["loss"], f"{name}/val/accuracy": va["accuracy"],
                f"{name}/val/f1": va["f1"], f"{name}/lr": optimizer.param_groups[0]["lr"],
            })

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  ep {epoch:3d}/{EPOCHS} {flag} | loss={tr['loss']:.4f} "
                  f"(lsr={tr['lsr_loss']:.4f} flip={tr['flip_loss']:.4f}) "
                  f"acc={tr['accuracy']:.4f} | val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f}")

        if patience >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch} (no val gain for {EARLY_STOP_PATIENCE}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate(model, test_loader, return_per_class=True)
    elapsed = time.time() - t0
    print(f"  -> val_acc={best_val_acc:.4f}  test_acc={test_res['accuracy']:.4f}  "
          f"mean_cls_acc={test_res['mean_class_acc']:.4f}  ({elapsed:.0f}s)\n")

    return {
        "variant": name, "rsl": use_rsl, "rac": use_rac,
        "best_val_acc": round(best_val_acc, 6),
        "test_acc": round(test_res["accuracy"], 6),
        "test_f1": round(test_res["f1"], 6),
        "test_mean_class_acc": round(test_res["mean_class_acc"], 6),
        "per_class_acc": {cls: round(a, 4) for cls, a in zip(CLASSES, test_res["per_class_acc"])},
        "train_time_s": round(elapsed, 1),
        "history": history,
    }


# ════════════════════════════════════════════════════════════════════
# Run all four variants
# ════════════════════════════════════════════════════════════════════
results = []
for name, cfg in VARIANTS.items():
    r = run_variant(name, cfg["rsl"], cfg["rac"])
    results.append(r)
    if wb is not None:
        wb.log({
            f"{name}/test_accuracy": r["test_acc"],
            f"{name}/test_f1": r["test_f1"],
            f"{name}/test_mean_class_acc": r["test_mean_class_acc"],
        })


# ════════════════════════════════════════════════════════════════════
# Summary (paper format: RSL / RAC toggles + gain over baseline)
# ════════════════════════════════════════════════════════════════════
def _tick(b):
    return "✓" if b else "✗"

print(f"\n{'='*72}")
print(f"  Ablation results -- {ARCH} / {DATASET}  (RSL × RAC)")
print(f"{'='*72}")
print(f"{'Variant':<10} {'RSL':>4} {'RAC':>4} {'Val Acc':>9} {'Test Acc':>9} {'Mean Cls':>10} {'Time':>8}")
print(f"{'-'*72}")
for r in results:
    print(f"{r['variant']:<10} {_tick(r['rsl']):>4} {_tick(r['rac']):>4} "
          f"{r['best_val_acc']:>9.4f} {r['test_acc']:>9.4f} "
          f"{r['test_mean_class_acc']:>10.4f} {r['train_time_s']:>7.1f}s")
print(f"{'-'*72}")

by_name = {r["variant"]: r for r in results}
if "baseline" in by_name:
    base = by_name["baseline"]["test_mean_class_acc"]
    print(f"\nMean-class-acc gain over baseline (the paper's headline metric):")
    for name in ("rsl", "rac", "rsl_rac"):
        if name in by_name:
            print(f"  {name:<10} {by_name[name]['test_mean_class_acc'] - base:+.4f}")

print(f"\nPer-class accuracy:")
print(f"  {'Variant':<10}", end="")
for c in CLASSES:
    print(f"  {c:>12}", end="")
print()
for r in results:
    print(f"  {r['variant']:<10}", end="")
    for c in CLASSES:
        print(f"  {r['per_class_acc'][c]:>12.4f}", end="")
    print()

# Save JSON
out_path = f"ablation_{ARCH}_{DATASET}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

if wb is not None:
    wb.finish()


# ════════════════════════════════════════════════════════════════════
# Combined training curves
# ════════════════════════════════════════════════════════════════════
def plot_ablation(results):
    colors = {"baseline": "#9467bd", "rsl": "#ff7f0e", "rac": "#1f77b4", "rsl_rac": "#2ca02c"}
    panels = [("accuracy", "Accuracy"), ("loss", "Loss"), ("f1", "F1-score")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    fig.suptitle(f"MEK ablation (RSL × RAC) — {ARCH} / {DATASET}")
    for ax, (k, title) in zip(axes, panels):
        for r in results:
            h, vh = r["history"][k], r["history"][f"val_{k}"]
            ep = range(1, len(h) + 1)
            c = colors.get(r["variant"], "gray")
            ax.plot(ep, h, color=c, lw=1.2, label=f"{r['variant']} (train)")
            ax.plot(ep, vh, color=c, lw=1.8, ls="--", label=f"{r['variant']} (val)")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"ablation_{ARCH}_{DATASET}_curves.png", dpi=150)
    plt.show()


plot_ablation(results)
