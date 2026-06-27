"""
Webcam-robust MEK + ResNet-18 (MS-Celeb-1M backbone) — single-file Kaggle script (RAF-DB | FER-2013).

The webcam-deployment counterpart of kaggle/mek_resnet18_MS-Celeb-1M.py: same MEK
method (Re-balanced Smooth Label + Re-balanced Attention Consistency) and the same
MS-Celeb-1M face-recognition backbone (the best init), but tuned for a LIVE webcam
(demo.py) rather than the last point of benchmark accuracy:

  • heavier "in-the-wild" augmentation — random-resized-crop (scale), Gaussian
    blur (focus/motion), strong colour jitter + occasional grayscale (lighting),
    mild perspective (off-angle), random erasing (occlusion). This shrinks the
    dataset → webcam domain gap, which is what usually breaks live demos.
  • EMA weight averaging for a small, reliable stability/accuracy gain.
  • a LIVE-TUNED loss recipe (ε=0.15, λ_flip=0.1) — gentler than the benchmark
    recipe (ε=0.1, λ=2) so the model commits to confident live predictions.
  • Defaults to RAF-DB: in-the-wild RGB, generalizes to a camera far better than
    FER-2013's grayscale 48×48.

ResNet-18 is also the fastest backbone for real-time/CPU webcam FPS — together with
the MS-Celeb-1M init this is the best webcam setup. Add resnet18_msceleb.pth as a
Kaggle dataset and point FACE_WEIGHTS_PATH at it (falls back to ImageNet if missing).

The deterministic paired flip (NOT RandomHorizontalFlip) is kept — it is the
augmentation the AC loss consumes. Saves mek_webcam_resnet18_<dataset>_best.pth,
which demo.py auto-discovers.

Toggle DATASET below. Set RESUME_CKPT / EVAL_ONLY to resume or evaluate only.
"""

# ════════════════════════════════════════════════════════════════════
# User toggles
# ════════════════════════════════════════════════════════════════════
DATASET     = "rafdb"        # "rafdb" (recommended for webcam) or "fer2013"
USE_WANDB   = False           # set True to log this run to Weights & Biases

RESUME_CKPT = None
EVAL_ONLY   = False

FER_ROOT = "/kaggle/input/datasets/msambare/fer2013"
RAF_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"

# Face-pretrained backbone (MS-Celeb-1M) — the biggest lever for real-face generalization
# and the paper's init. Add resnet18_msceleb.pth as a Kaggle dataset and set the path below.
# The loader tolerates key-naming differences and FALLS BACK to ImageNet if the file is
# missing/unmatched, so the run never hard-fails. The saved checkpoint stays a standard
# ResNet-18 (demo loads it unchanged).
USE_FACE_BACKBONE = True
FACE_WEIGHTS_PATH = "/kaggle/input/resnet18-msceleb/resnet18_msceleb.pth"
# A .pth (resnet18_msceleb.pth) loads safely with weights_only=True and needs NO hash —
# leave this empty. It is MANDATORY only if you instead load a .pkl (then the run HALTS
# unless it matches):
#   import hashlib; print(hashlib.sha256(open(FACE_WEIGHTS_PATH,'rb').read()).hexdigest())
FACE_WEIGHTS_SHA256 = ""

# ════════════════════════════════════════════════════════════════════
# Imports
# ════════════════════════════════════════════════════════════════════
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

from torchmetrics import Accuracy, Precision, Recall, F1Score


# ════════════════════════════════════════════════════════════════════
# Arch-specific config
# ════════════════════════════════════════════════════════════════════
ARCH         = "resnet18"
ARCH_FACTORY = models.resnet18
ARCH_WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1     # only used if face weights are missing
FEAT_DIM     = 512
ARCH_LR_MULT = 1.0
ARCH_DROPOUT = 0.40


# ════════════════════════════════════════════════════════════════════
# Dataset-specific config (both forced to 224 — MEK needs a 7×7 attn map)
# ════════════════════════════════════════════════════════════════════
if DATASET == "fer2013":
    DATA_ROOT, BASE_LR, EPOCHS = FER_ROOT, 5e-3, 80
    ROTATION_DEG = 10
elif DATASET == "rafdb":
    DATA_ROOT, BASE_LR, EPOCHS = RAF_ROOT, 1e-3, 80   # validation-tuned webcam best: 80 ep (was 60)
    ROTATION_DEG = 15
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}.")

TRAIN_DIR   = f"{DATA_ROOT}/train"
TEST_DIR    = f"{DATA_ROOT}/test"
NUM_CLASSES = 7
IMG_SIZE    = 224
CROP_SIZE   = 224

BATCH_SIZE          = 64
LR                  = 1e-4        # paper's ResNet recipe: Adam @ a fixed 1e-4
DROPOUT             = ARCH_DROPOUT
LABEL_SMOOTH        = 0.15        # live-tuned (gentler than the benchmark ε=0.1) — confident live preds
FLIP_LOSS_WEIGHT    = 0.1         # live-tuned (lighter than the benchmark λ=2)
MOMENTUM            = 0.9
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 5.0
VAL_SPLIT           = 0.1
SEED                = 42
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 25
USE_WEIGHTED_SAMPLER = True
EMA_DECAY           = 0.999
USE_CLAHE           = True        # apply demo.py's CLAHE in training too → train/serve match

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = f"mek_webcam_{ARCH}_{DATASET}_best.pth"

print(f"Device:   {DEVICE}")
print(f"Method:   MEK (webcam-robust)  — arxiv 2310.19636 + heavy aug + EMA")
print(f"Arch:     {ARCH}  (lr_mult={ARCH_LR_MULT}, dropout={DROPOUT})")
print(f"Dataset:  {DATASET}  →  {DATA_ROOT}")
print(f"Train:    lr={LR:.2e} bs={BATCH_SIZE} epochs={EPOCHS} ε_lsr={LABEL_SMOOTH} "
      f"λ_flip={FLIP_LOSS_WEIGHT} ema={EMA_DECAY}")
if RESUME_CKPT:
    print(f"Resume:   {RESUME_CKPT}")
if EVAL_ONLY:
    print("Eval-only mode: skipping training.")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════
# Data — webcam-oriented augmentation; train loader yields (img, label, img_hflip).
# ════════════════════════════════════════════════════════════════════
class CLAHEEqualize:
    """CLAHE on the LAB L channel — identical to demo.py's apply_clahe, so the model
    trains on the same lighting-normalized distribution the webcam demo feeds it.
    For FER2013 the demo also greyscales the frame; FER is already grey, so the
    CLAHE'd-grey training distribution still matches. No-op if OpenCV is missing."""
    def __call__(self, img):
        try:
            import cv2
        except ImportError:
            return img
        rgb = np.asarray(img.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)


_clahe = [CLAHEEqualize()] if USE_CLAHE else []

train_transform = transforms.Compose(_clahe + [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomRotation(ROTATION_DEG),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

eval_transform = transforms.Compose(_clahe + [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class PairedFlipDataset(Dataset):
    """Returns (img, label, H-flipped img). The flip is applied to the already
    augmented tensor, so img_flip == hflip(img) exactly (AC-loss assumption)."""
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
# Model — ResNet encoder + 1×1 conv CAM head → (logits, hm[B,K,7,7])
# ════════════════════════════════════════════════════════════════════
def _load_state_any(path):
    """Read a face-recognition checkpoint: a torch .pth/.pt (e.g. resnet18_msceleb.pth)
    or a numpy-dict .pkl.

    Security note: this loads ONLY a trusted, user-supplied weights file added as a
    Kaggle dataset, never untrusted input. The .pth/.pt path uses weights_only=True
    (safe); a .pkl is a plain pickle (arbitrary-code risk) and is gated by the
    MANDATORY FACE_WEIGHTS_SHA256 pin below.
    """
    import os, pickle, hashlib
    if not path or not os.path.exists(path):
        return None

    # .pkl is unpickled (arbitrary-code-execution risk), so its integrity check is
    # MANDATORY and fails closed: without a matching pinned SHA-256 we raise and halt
    # (never silently fall back) so a misconfigured run stops loudly. Prefer a .pth
    # export (loaded below with weights_only=True) to avoid pickle entirely.
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = f.read()
        if not FACE_WEIGHTS_SHA256:
            raise RuntimeError(
                "FACE_WEIGHTS_SHA256 must be set before loading a .pkl face-weights file. "
                "Compute it with:\n"
                "  python -c \"import hashlib;print(hashlib.sha256(open(PATH,'rb').read()).hexdigest())\"\n"
                "or convert the weights to .pth and load that instead (weights_only=True, no hash needed)."
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
    or None to let MEKResNet fall back to ImageNet. Robust to key-naming differences:
    tries name+shape matching first, then ordered shape matching, keeps whichever fills
    more encoder tensors, and only accepts the face init if ≥80% were filled.
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
    # Exclude the classifier (fc.*) and BN step counters (num_batches_tracked) — the
    # latter aren't weights and aren't present in the face export, so counting them
    # would understate the match. What's left is the real conv/BN weight set.
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
    print(f"Face init: filled {len(best)}/{len(enc_keys)} encoder tensors (match by {strat}).")
    if len(best) < 0.8 * len(enc_keys):
        print("WARNING: <80% of encoder tensors matched — keys differ too much; "
              "falling back to ImageNet init.")
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
        hm = self.classifier(feat)
        logits = F.adaptive_avg_pool2d(hm, 1).flatten(1)
        return logits, hm


_backbone_net = build_face_backbone()
BACKBONE = "MS-Celeb-1M" if _backbone_net is not None else "ImageNet"   # actual init (reflects fallback)
print(f"Backbone: {BACKBONE}")
model = MEKResNet(NUM_CLASSES, DROPOUT, backbone=_backbone_net).to(DEVICE)

with torch.no_grad():
    _l, _hm = model(torch.zeros(2, 3, CROP_SIZE, CROP_SIZE, device=DEVICE))
    assert _l.shape == (2, NUM_CLASSES) and _hm.shape[:2] == (2, NUM_CLASSES)
print(f"Forward OK: logits={tuple(_l.shape)} hm={tuple(_hm.shape)}")

if RESUME_CKPT is not None:
    state = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {RESUME_CKPT}")


# ════════════════════════════════════════════════════════════════════
# EMA — exponential moving average of the weights
# ════════════════════════════════════════════════════════════════════
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(mv.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(mv)


ema = ModelEMA(model, decay=EMA_DECAY)


# ════════════════════════════════════════════════════════════════════
# Losses — LSR2 + ACLoss (direct ports of the MEK reference repo)
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
    fl = F.mse_loss(hm_o, hm_unflip, reduction="none").mean(dim=[-1, -2])
    return (fl @ balance_w).mean()


criterion      = ReBalancedLabelSmoothing(LABEL_SMOOTH, balance_weights).to(DEVICE)
eval_criterion = nn.CrossEntropyLoss()


# ════════════════════════════════════════════════════════════════════
# Optimizer / Scheduler / AMP
# ════════════════════════════════════════════════════════════════════
# Adam + ExponentialLR(gamma=0.95) — validation-tuned webcam best (slower decay than the
# benchmark γ=0.9; matches the package webcam trainer train_mek_webcam.py).
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
amp_enabled = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
flip_grid = None


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
        ema.update(model)

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
def evaluate(loader, eval_model, return_per_class=False):
    eval_model.eval()
    metrics = make_metrics()
    total_loss, total_n = 0.0, 0
    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) if return_per_class else None

    for img, label in loader:
        img   = img.to(DEVICE,   non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        logits, _ = eval_model(img)
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
# Main loop (evaluation + checkpointing use the EMA weights)
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


wb = _init_wandb(f"mek_webcam_{ARCH}_{DATASET}", {
    "arch": ARCH, "dataset": DATASET, "method": "MEK-webcam", "epochs": EPOCHS, "lr": LR,
    "batch_size": BATCH_SIZE, "dropout": DROPOUT, "label_smooth": LABEL_SMOOTH,
    "flip_loss_weight": FLIP_LOSS_WEIGHT, "ema_decay": EMA_DECAY, "augmentation": "webcam",
    "optimizer": "Adam", "backbone": BACKBONE,
})

best_val_acc = 0.0
best_state   = None

if not EVAL_ONLY:
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(train_loader)
        va = evaluate(val_loader, ema.ema)
        scheduler.step()

        if va["accuracy"] > best_val_acc:
            best_val_acc = va["accuracy"]
            best_state   = copy.deepcopy(ema.ema.state_dict())
            torch.save(best_state, CKPT_PATH)
            patience, flag = 0, "+"
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
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s"
        )

        if patience >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val_acc gain for {EARLY_STOP_PATIENCE} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        ema.ema.load_state_dict(best_state)
        print(f"\nRestored best EMA weights (val_acc={best_val_acc:.4f}).")
else:
    print("\nEval-only — using the loaded checkpoint as-is.\n")


# ════════════════════════════════════════════════════════════════════
# Final test evaluation
# ════════════════════════════════════════════════════════════════════
test_res = evaluate(test_loader, model, return_per_class=True)
print(f"\nTest results:")
print(f"  acc:            {test_res['accuracy']:.4f}")
print(f"  f1 (macro):     {test_res['f1']:.4f}")
print(f"  mean class acc: {test_res['mean_class_acc']:.4f}    ← MEK headline metric")
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
    panels = [("accuracy", "Accuracy"), ("loss", "Loss"), ("f1", "F1-score"),
              ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    fig.suptitle(f"MEK-webcam / {ARCH} / {DATASET} — training curves")
    for ax, (k, t) in zip(axes, panels):
        ep = range(1, len(h[k]) + 1)
        ax.plot(ep, h[k], label="train")
        ax.plot(ep, h[f"val_{k}"], label="val")
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.set_ylabel(t); ax.legend()
    plt.tight_layout(); plt.show()


plot_history(history)
