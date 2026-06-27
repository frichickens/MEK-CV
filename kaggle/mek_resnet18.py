"""
MEK + ResNet-18 — single-file Kaggle script (FER-2013 | RAF-DB).

Method: Mine Extra Knowledge (Zhang et al., NeurIPS 2023, arxiv 2310.19636).
This is a faithful port of the original Swin-T pipeline (zyh-uaiaaaa/
Mine-Extra-Knowledge) onto a torchvision ResNet, with the same two losses:

  1. Re-balanced Smooth Label (LSR2): label-smoothing where the ε mass is
     redistributed proportionally to inverse class frequency, pulling the
     model toward minority classes.

  2. Re-balanced Attention Map (RAM, ACLoss): for every batch the network is
     run twice — on `img` and on horizontally-flipped `img` — and an MSE
     consistency loss is enforced between the two attention maps after
     un-flipping the second one. The per-class consistency error is weighted
     by inverse class frequency so the regularizer is strongest on minor
     classes (fear / disgust).

Toggle DATASET below. To resume from a saved checkpoint, set RESUME_CKPT.
To skip training and only evaluate the checkpoint, set EVAL_ONLY = True.
"""

# ════════════════════════════════════════════════════════════════════
# User toggles
# ════════════════════════════════════════════════════════════════════
DATASET     = "fer2013"      # "fer2013" or "rafdb"
USE_WANDB   = False           # set True to log this run to Weights & Biases

RESUME_CKPT = None            # e.g. "/kaggle/input/my-mek-ckpt/mek_resnet18_rafdb_best.pth"
EVAL_ONLY   = False           # True → load RESUME_CKPT, skip training, just test

FER_ROOT = "/kaggle/input/datasets/msambare/fer2013"
RAF_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"

# ════════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════════
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

from torchmetrics import Accuracy, Precision, Recall, F1Score


# ════════════════════════════════════════════════════════════════════
# Arch-specific config (this is the only block that differs between
# kaggle/mek_resnet18.py / mek_resnet34.py / mek_resnet50.py)
# ════════════════════════════════════════════════════════════════════
ARCH         = "resnet18"
ARCH_FACTORY = models.resnet18
ARCH_WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1
FEAT_DIM     = 512
ARCH_LR_MULT = 1.0
ARCH_DROPOUT = 0.40


# ════════════════════════════════════════════════════════════════════
# Dataset-specific config
# ════════════════════════════════════════════════════════════════════
if DATASET == "fer2013":
    DATA_ROOT          = FER_ROOT
    TRAIN_DIR          = f"{DATA_ROOT}/train"
    TEST_DIR           = f"{DATA_ROOT}/test"
    NUM_CLASSES        = 7
    IMG_SIZE           = 224          # MEK needs 224 → 7×7 attention maps
    CROP_SIZE          = 224
    CROP_PADDING       = 8
    BATCH_SIZE         = 64
    EPOCHS             = 80
    LR                 = 1e-4         # validation-tuned best (FER MEK-RN18)
    LABEL_SMOOTH       = 0.1          # ε
    FLIP_LOSS_WEIGHT   = 0.25         # λ — FER wants a much lighter AC weight than RAF-DB
    ROTATION_DEG       = 10
    COLOR_JITTER       = 0.2
    RANDOM_ERASING_P   = 0.5
elif DATASET == "rafdb":
    DATA_ROOT          = RAF_ROOT
    TRAIN_DIR          = f"{DATA_ROOT}/train"
    TEST_DIR           = f"{DATA_ROOT}/test"
    NUM_CLASSES        = 7
    IMG_SIZE           = 224
    CROP_SIZE          = 224
    CROP_PADDING       = 8
    BATCH_SIZE         = 64
    EPOCHS             = 60
    LR                 = 3e-4         # validation-tuned best (RAF-DB MEK-RN18)
    LABEL_SMOOTH       = 0.1          # ε
    FLIP_LOSS_WEIGHT   = 2.0          # λ — paper recipe is well-chosen on RAF-DB
    ROTATION_DEG       = 15
    COLOR_JITTER       = 0.3
    RANDOM_ERASING_P   = 0.25
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}.")

# Shared MEK hyperparameters. LR / ε / λ are validation-tuned per dataset above
# (results-CV-project.md); the optimizer (Adam) + ExponentialLR γ=0.9 follow the
# paper's ResNet recipe.
DROPOUT             = ARCH_DROPOUT
MOMENTUM            = 0.9
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 5.0
VAL_SPLIT           = 0.1
SEED                = 42
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 25
USE_WEIGHTED_SAMPLER = True

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = f"mek_{ARCH}_{DATASET}_best.pth"

print(f"Device:   {DEVICE}")
print(f"Method:   MEK (Mine Extra Knowledge)  — arxiv 2310.19636")
print(f"Arch:     {ARCH}  (lr_mult={ARCH_LR_MULT}, dropout={DROPOUT})")
print(f"Dataset:  {DATASET}  →  {DATA_ROOT}")
print(f"Train:    lr={LR:.2e} bs={BATCH_SIZE} epochs={EPOCHS} ε_lsr={LABEL_SMOOTH} λ_flip={FLIP_LOSS_WEIGHT}")
if RESUME_CKPT:
    print(f"Resume:   {RESUME_CKPT}")
if EVAL_ONLY:
    print("Eval-only mode: skipping training.")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════
# Data — train loader yields (img, label, img_hflipped) for the AC loss.
# We deliberately do NOT use RandomHorizontalFlip in the train transform;
# the deterministic flip in `_PairedFlipDataset` IS the augmentation MEK
# expects, and randomizing it would break the AC-loss assumption.
# ════════════════════════════════════════════════════════════════════
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

# Inverse-frequency balance weights, normalized to mean = 1.
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
# Model — ResNet encoder + 1×1 conv CAM head returning (logits, hm).
# At 224 input, hm has shape [B, K, 7, 7] (per-class attention maps).
# ════════════════════════════════════════════════════════════════════
class MEKResNet(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        net = ARCH_FACTORY(weights=ARCH_WEIGHTS)
        self.encoder = nn.Sequential(*list(net.children())[:-2])
        self.bn         = nn.BatchNorm2d(FEAT_DIM)
        self.drop       = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(FEAT_DIM, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        feat = self.encoder(x)                    # [B, FEAT_DIM, 7, 7]
        feat = self.bn(feat)
        feat = self.drop(feat)
        hm = self.classifier(feat)                # [B, K, 7, 7]
        logits = F.adaptive_avg_pool2d(hm, 1).flatten(1)
        return logits, hm


model = MEKResNet(NUM_CLASSES, DROPOUT).to(DEVICE)

with torch.no_grad():
    _l, _hm = model(torch.zeros(2, 3, CROP_SIZE, CROP_SIZE, device=DEVICE))
    assert _l.shape == (2, NUM_CLASSES) and _hm.shape[:2] == (2, NUM_CLASSES)
print(f"Forward OK: logits={tuple(_l.shape)} hm={tuple(_hm.shape)}")

if RESUME_CKPT is not None:
    state = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {RESUME_CKPT}")


# ════════════════════════════════════════════════════════════════════
# Losses — Re-balanced Smooth Label (LSR2) and Re-balanced Attention (ACLoss).
# Direct ports of train_exp.py from the MEK reference repo.
# ════════════════════════════════════════════════════════════════════
class ReBalancedLabelSmoothing(nn.Module):
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
    B = hm_o.size(0)
    grid = flip_grid.expand(B, -1, -1, -1).permute(0, 2, 3, 1)
    hm_unflip = F.grid_sample(hm_f, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
    fl = F.mse_loss(hm_o, hm_unflip, reduction="none").mean(dim=[-1, -2])  # [B, K]
    return (fl @ balance_w).mean()


criterion      = ReBalancedLabelSmoothing(LABEL_SMOOTH, balance_weights).to(DEVICE)
eval_criterion = nn.CrossEntropyLoss()


# ════════════════════════════════════════════════════════════════════
# Optimizer / Scheduler / AMP
# ════════════════════════════════════════════════════════════════════
# Paper's ResNet recipe: Adam + ExponentialLR(gamma=0.9).
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
amp_enabled = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
flip_grid = None    # built lazily once we know the hm spatial size


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
# Train / Eval
# ════════════════════════════════════════════════════════════════════
def train_one_epoch(loader):
    global flip_grid
    model.train()
    metrics = make_metrics()
    total_loss, sum_lsr, sum_flip, total_n = 0.0, 0.0, 0.0, 0

    for img, label, img_flip in loader:
        img      = img.to(DEVICE,      non_blocking=True)
        label    = label.to(DEVICE,    non_blocking=True)
        img_flip = img_flip.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits_o, hm_o = model(img)
            logits_f, hm_f = model(img_flip)

            if flip_grid is None:
                h, w = hm_o.shape[-2:]
                flip_grid = make_flip_grid(h, w).to(DEVICE)

            lsr  = criterion(logits_o, label)
            fl   = ac_loss(hm_o, hm_f, flip_grid, balance_weights)
            loss = lsr + FLIP_LOSS_WEIGHT * fl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * img.size(0)
        sum_lsr    += lsr.item()  * img.size(0)
        sum_flip   += fl.item()   * img.size(0)
        total_n    += img.size(0)
        for m in metrics.values():
            m.update(logits_o.softmax(1), label)

    out = {k: v.compute().item() for k, v in metrics.items()}
    out["loss"]      = total_loss / total_n
    out["lsr_loss"]  = sum_lsr   / total_n
    out["flip_loss"] = sum_flip  / total_n
    return out


@torch.no_grad()
def evaluate(loader, return_per_class=False):
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
# Main loop
# ════════════════════════════════════════════════════════════════════
_keys = ["loss", "accuracy", "precision", "recall", "f1"]
history = {k: [] for k in _keys + [f"val_{k}" for k in _keys]}
history["lsr_loss"]  = []
history["flip_loss"] = []


# ── Weights & Biases (optional) ──────────────────────────────────
def _init_wandb(run_name, config):
    """Best-effort W&B init for Kaggle. Reads WANDB_API_KEY from the environment
    or a Kaggle Secret of the same name. Returns the wandb module when logging is
    active, else None — never hard-fails (the run proceeds without logging)."""
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


wb = _init_wandb(f"mek_{ARCH}_{DATASET}", {
    "arch": ARCH, "dataset": DATASET, "method": "MEK", "epochs": EPOCHS, "lr": LR,
    "batch_size": BATCH_SIZE, "dropout": DROPOUT, "label_smooth": LABEL_SMOOTH,
    "flip_loss_weight": FLIP_LOSS_WEIGHT, "optimizer": "Adam", "backbone": "ImageNet",
})

best_val_acc = 0.0
best_state   = None

if not EVAL_ONLY:
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(train_loader)
        va = evaluate(val_loader)
        scheduler.step()

        if va["accuracy"] > best_val_acc:
            best_val_acc = va["accuracy"]
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_PATH)
            patience, flag = 0, "✓"
        else:
            patience += 1
            flag = " "

        for k in _keys:
            history[k].append(tr[k])
            history[f"val_{k}"].append(va[k])
        history["lsr_loss"].append(tr["lsr_loss"])
        history["flip_loss"].append(tr["flip_loss"])

        if wb is not None:
            wb.log({
                "epoch": epoch,
                "train/loss": tr["loss"], "train/accuracy": tr["accuracy"], "train/f1": tr["f1"],
                "train/lsr_loss": tr["lsr_loss"], "train/flip_loss": tr["flip_loss"],
                "val/loss": va["loss"], "val/accuracy": va["accuracy"], "val/f1": va["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} {flag} | "
            f"loss={tr['loss']:.4f} (lsr={tr['lsr_loss']:.4f} flip={tr['flip_loss']:.4f}) "
            f"acc={tr['accuracy']:.4f} | "
            f"val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{time.time()-t0:.0f}s"
        )

        if patience >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val_acc gain for {EARLY_STOP_PATIENCE} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights (val_acc={best_val_acc:.4f}).")
else:
    print("\nEval-only — using the loaded checkpoint as-is.\n")


# ════════════════════════════════════════════════════════════════════
# Final test evaluation (with per-class breakdown — MEK's headline metric)
# ════════════════════════════════════════════════════════════════════
test_res = evaluate(test_loader, return_per_class=True)
print(f"\nTest results:")
print(f"  acc:            {test_res['accuracy']:.4f}")
print(f"  f1 (macro):     {test_res['f1']:.4f}")
print(f"  precision:      {test_res['precision']:.4f}")
print(f"  recall:         {test_res['recall']:.4f}")
print(f"  mean class acc: {test_res['mean_class_acc']:.4f}    "
      f"← MEK headline metric (robust to class imbalance)")
print(f"  per-class acc:")
for cls, acc in zip(CLASSES, test_res["per_class_acc"]):
    print(f"    {cls:12s} {acc:.4f}")

if wb is not None:
    wb.log({
        "test/accuracy": test_res["accuracy"], "test/f1": test_res["f1"],
        "test/mean_class_acc": test_res["mean_class_acc"],
    })
    wb.finish()


# ════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════
def plot_history(h):
    if not h["accuracy"]:
        return
    panels = [
        ("accuracy",  "Accuracy"),
        ("loss",      "Loss"),
        ("f1",        "F1-score"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    fig.suptitle(f"MEK / {ARCH} / {DATASET} — training curves")
    for ax, (k, t) in zip(axes, panels):
        ep = range(1, len(h[k]) + 1)
        ax.plot(ep, h[k], label="train")
        ax.plot(ep, h[f"val_{k}"], label="val")
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.set_ylabel(t); ax.legend()
    plt.tight_layout(); plt.show()

    if h["lsr_loss"]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(h["lsr_loss"],  label="LSR2 (label smoothing)")
        ax.plot(h["flip_loss"], label="ACLoss (flip-consistency)")
        ax.set_title(f"MEK loss components — {ARCH} / {DATASET}")
        ax.set_xlabel("Epoch"); ax.legend()
        plt.tight_layout(); plt.show()


plot_history(history)
