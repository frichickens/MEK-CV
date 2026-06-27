"""
ResNet-34 + (FER2013 | RAF-DB) — single-file Kaggle training script.

Copy-paste into a Kaggle notebook cell. To switch datasets, change DATASET below
to "fer2013" or "rafdb". Default paths match the popular Kaggle datasets:
  • FER-2013: msambare/fer2013       (DATASET="fer2013")
  • RAF-DB:   shuvoalok/raf-db-dataset (DATASET="rafdb")
Override FER_ROOT / RAF_ROOT below if your dataset is mounted elsewhere.

Recipe sources:
  • LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch  (73.7% SOTA single-net)
  • WuJie1010/Facial-Expression-Recognition.Pytorch       (~71-73% w/ TenCrop)

Per-dataset recipe differences (auto-applied below):
                       FER-2013                  RAF-DB
  input size           48 → RandomCrop 44        224 (aligned faces, larger)
  base lr              1e-2                      1e-3
  epochs               100                       60
  batch size           128                       64
  mixup                α=0.2                     OFF (cleaner data)
  label smoothing      0.1                       0.05
  rotation             ±10°                      ±15°
  random erasing p     0.5                       0.25

ResNet-34-specific tuning: lr_mult=0.7 (smaller step than ResNet-18 because
the deeper backbone is more sensitive), dropout=0.45.
"""

# ════════════════════════════════════════════════════════════════════
# Toggle dataset here.  Optional path overrides directly below.
# ════════════════════════════════════════════════════════════════════
DATASET  = "fer2013"     # "fer2013" or "rafdb"
USE_WANDB = False        # set True to log this run to Weights & Biases

FER_ROOT = "/kaggle/input/datasets/msambare/fer2013"
RAF_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"

# ════════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════════
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models

from torchmetrics import Accuracy, Precision, Recall, F1Score


# ════════════════════════════════════════════════════════════════════
# 0.  Arch-specific config (the only block that changes between
#     resnet18.py / resnet34.py / resnet50.py)
# ════════════════════════════════════════════════════════════════════
ARCH            = "resnet34"
ARCH_FACTORY    = models.resnet34
ARCH_WEIGHTS    = models.ResNet34_Weights.IMAGENET1K_V1
ARCH_LR_MULT    = 1.0          # BASE_LR below is already the validation-tuned per-dataset LR
ARCH_DROPOUT    = 0.45


# ════════════════════════════════════════════════════════════════════
# 1.  Dataset-specific config
# ════════════════════════════════════════════════════════════════════
if DATASET == "fer2013":
    DATA_ROOT          = FER_ROOT
    TRAIN_DIR          = f"{DATA_ROOT}/train"
    TEST_DIR           = f"{DATA_ROOT}/test"
    NUM_CLASSES        = 7
    IMG_SIZE           = 48
    CROP_SIZE          = 44
    CROP_PADDING       = 2
    BATCH_SIZE         = 128
    EPOCHS             = 100
    BASE_LR            = 7e-3         # validation-tuned best (default lr); see results-CV-project.md
    LABEL_SMOOTH       = 0.1
    MIXUP_ALPHA        = 0.2
    ROTATION_DEG       = 10
    COLOR_JITTER       = 0.2
    RANDOM_ERASING_P   = 0.5
    USE_WEIGHTED_SAMPLER = True
elif DATASET == "rafdb":
    DATA_ROOT          = RAF_ROOT
    TRAIN_DIR          = f"{DATA_ROOT}/train"
    TEST_DIR           = f"{DATA_ROOT}/test"
    NUM_CLASSES        = 7
    IMG_SIZE           = 224
    CROP_SIZE          = 224
    CROP_PADDING       = 8
    BATCH_SIZE         = 64
    EPOCHS             = 80         # validation-tuned best (80 ep > default 60)
    BASE_LR            = 1e-3       # validation-tuned best (lr1e3); see results-CV-project.md
    LABEL_SMOOTH       = 0.05
    MIXUP_ALPHA        = 0.0
    ROTATION_DEG       = 15
    COLOR_JITTER       = 0.3
    RANDOM_ERASING_P   = 0.25
    USE_WEIGHTED_SAMPLER = True
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}. Pick 'fer2013' or 'rafdb'.")

# ════════════════════════════════════════════════════════════════════
# 2.  Shared training config
# ════════════════════════════════════════════════════════════════════
LR                  = BASE_LR * ARCH_LR_MULT
DROPOUT             = ARCH_DROPOUT
MOMENTUM            = 0.9
WEIGHT_DECAY        = 1e-4
VAL_SPLIT           = 0.1
SEED                = 42
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 25

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = f"{ARCH}_{DATASET}_best.pth"

print(f"Device:   {DEVICE}")
print(f"Arch:     {ARCH}  (lr_mult={ARCH_LR_MULT}, dropout={DROPOUT})")
print(f"Dataset:  {DATASET}  →  {DATA_ROOT}")
print(f"Train:    lr={LR:.2e} bs={BATCH_SIZE} epochs={EPOCHS} mixup={MIXUP_ALPHA} smooth={LABEL_SMOOTH}")
print(f"Image:    Resize {IMG_SIZE} → Crop {CROP_SIZE}")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════
# 3.  Data
# ════════════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomCrop(CROP_SIZE, padding=CROP_PADDING),
    transforms.RandomHorizontalFlip(),
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

full_aug  = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

CLASSES = full_aug.classes
print(f"Classes:  {CLASSES}")

idx       = np.random.permutation(len(full_aug))
val_size  = int(len(full_aug) * VAL_SPLIT)
val_idx   = idx[:val_size]
train_idx = idx[val_size:]

train_dataset = Subset(full_aug,  train_idx)
val_dataset   = Subset(full_eval, val_idx)

print(f"Sizes:    Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")

if USE_WEIGHTED_SAMPLER:
    train_labels = np.array(full_aug.targets)[train_idx]
    counts  = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_w = torch.from_numpy(weights[train_labels]).double()
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(train_labels), replacement=True)
    print(f"Counts:   {dict(zip(CLASSES, counts.astype(int)))}")
else:
    sampler = None

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=(sampler is None), sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)


# ════════════════════════════════════════════════════════════════════
# 4.  Model
# ════════════════════════════════════════════════════════════════════
def build_model() -> nn.Module:
    net = ARCH_FACTORY(weights=ARCH_WEIGHTS)
    in_feats = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_feats, NUM_CLASSES),
    )
    return net


model = build_model().to(DEVICE)

with torch.no_grad():
    dummy = torch.zeros(2, 3, CROP_SIZE, CROP_SIZE, device=DEVICE)
    assert model(dummy).shape == (2, NUM_CLASSES), "Bad output shape"
print("Model output shape OK.")


# ════════════════════════════════════════════════════════════════════
# 5.  Loss / Optimizer / Scheduler / AMP
# ════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

amp_enabled = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)


# ════════════════════════════════════════════════════════════════════
# 6.  Mixup
# ════════════════════════════════════════════════════════════════════
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_loss(logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ════════════════════════════════════════════════════════════════════
# 7.  Metrics
# ════════════════════════════════════════════════════════════════════
def make_metrics() -> dict:
    return {
        "accuracy":  Accuracy( task="multiclass", num_classes=NUM_CLASSES).to(DEVICE),
        "precision": Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
        "recall":    Recall(   task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
        "f1":        F1Score(  task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
    }


# ════════════════════════════════════════════════════════════════════
# 8.  Train / Eval epoch
# ════════════════════════════════════════════════════════════════════
def train_one_epoch(loader: DataLoader) -> dict:
    model.train()
    metrics = make_metrics()
    total_loss, total_n = 0.0, 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if MIXUP_ALPHA > 0:
                x, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
                logits = model(x)
                loss   = mixup_loss(logits, y_a, y_b, lam)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)
        for m in metrics.values():
            m.update(logits.softmax(1), labels)

    out = {k: v.compute().item() for k, v in metrics.items()}
    out["loss"] = total_loss / total_n
    return out


def per_class_accuracy(confmat):
    """Diagonal / row-sum, with safe division. Matches MEK's headline metric."""
    rs = confmat.sum(axis=1)
    return np.where(rs > 0, np.diag(confmat) / np.maximum(rs, 1), 0.0)


@torch.no_grad()
def evaluate(loader: DataLoader, return_per_class: bool = False) -> dict:
    model.eval()
    metrics = make_metrics()
    total_loss, total_n = 0.0, 0
    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) if return_per_class else None

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)
        for m in metrics.values():
            m.update(logits.softmax(1), labels)
        if confmat is not None:
            preds = logits.argmax(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
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
# 9.  TenCrop TTA
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_tencrop() -> float:
    model.eval()

    def to_tencrop(img):
        crops = transforms.TenCrop(CROP_SIZE)(transforms.Resize((IMG_SIZE, IMG_SIZE))(img))
        norm = transforms.Normalize(NORM_MEAN, NORM_STD)
        return torch.stack([norm(transforms.functional.to_tensor(c)) for c in crops])

    ds = datasets.ImageFolder(TEST_DIR, transform=to_tencrop)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    correct, total = 0, 0
    for images, labels in dl:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        bs, ncrops, c, h, w = images.shape
        logits = model(images.view(-1, c, h, w))
        logits = logits.view(bs, ncrops, -1).mean(1)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ════════════════════════════════════════════════════════════════════
# 10. Main loop
# ════════════════════════════════════════════════════════════════════
_keys = ["loss", "accuracy", "precision", "recall", "f1"]
history = {k: [] for k in _keys + [f"val_{k}" for k in _keys]}


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


wb = _init_wandb(f"baseline_{ARCH}_{DATASET}", {
    "arch": ARCH, "dataset": DATASET, "epochs": EPOCHS, "lr": LR,
    "batch_size": BATCH_SIZE, "mixup_alpha": MIXUP_ALPHA, "dropout": DROPOUT,
    "label_smooth": LABEL_SMOOTH, "optimizer": "SGD",
})

best_val_acc = 0.0
best_state   = None
patience     = 0

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

    if wb is not None:
        wb.log({
            "epoch": epoch,
            "train/loss": tr["loss"], "train/accuracy": tr["accuracy"], "train/f1": tr["f1"],
            "train/precision": tr["precision"], "train/recall": tr["recall"],
            "val/loss": va["loss"], "val/accuracy": va["accuracy"], "val/f1": va["f1"],
            "val/precision": va["precision"], "val/recall": va["recall"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

    print(
        f"Epoch {epoch:3d}/{EPOCHS} {flag} | "
        f"loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} f1={tr['f1']:.4f} | "
        f"val_loss={va['loss']:.4f} val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} | "
        f"lr={optimizer.param_groups[0]['lr']:.2e} | "
        f"{time.time()-t0:.0f}s"
    )

    if patience >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val_acc improvement for {EARLY_STOP_PATIENCE} epochs).")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best weights from {CKPT_PATH}  (val_acc={best_val_acc:.4f}).")


# ════════════════════════════════════════════════════════════════════
# 11. Final evaluation
# ════════════════════════════════════════════════════════════════════
test_res = evaluate(test_loader, return_per_class=True)
print(f"\nTest results:")
print(f"  acc:            {test_res['accuracy']:.4f}")
print(f"  f1 (macro):     {test_res['f1']:.4f}")
print(f"  precision:      {test_res['precision']:.4f}")
print(f"  recall:         {test_res['recall']:.4f}")
print(f"  mean class acc: {test_res['mean_class_acc']:.4f}    "
      f"← imbalance-robust headline metric (matches MEK)")
print(f"  per-class acc:")
for cls, acc in zip(CLASSES, test_res["per_class_acc"]):
    print(f"    {cls:12s} {acc:.4f}")

tta_acc = evaluate_tencrop()
print(f"  10-crop TTA:    {tta_acc:.4f}")

if wb is not None:
    wb.log({
        "test/accuracy": test_res["accuracy"], "test/f1": test_res["f1"],
        "test/mean_class_acc": test_res["mean_class_acc"], "test/tta_accuracy": tta_acc,
    })
    wb.finish()


# ════════════════════════════════════════════════════════════════════
# 12. Plots
# ════════════════════════════════════════════════════════════════════
def plot_history(h: dict) -> None:
    panels = [
        ("accuracy",  "Accuracy"),
        ("loss",      "Loss"),
        ("f1",        "F1-score"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    fig.suptitle(f"{ARCH} / {DATASET} — training curves")
    for ax, (k, t) in zip(axes, panels):
        ep = range(1, len(h[k]) + 1)
        ax.plot(ep, h[k],          label="train")
        ax.plot(ep, h[f"val_{k}"], label="val")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(t)
        ax.legend()
    plt.tight_layout()
    plt.show()


plot_history(history)
