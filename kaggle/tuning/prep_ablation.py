"""
Prepare RAF-DB MEK-ResNet18 (MS-Celeb-1M) RSL×RAC ABLATION kernels and emit a queue.

Base script: kaggle/ablation/mek_ablation_resnet18.py — runs the 2×2 {baseline, rsl,
rac, rsl_rac} back-to-back on the SAME data/seed/backbone. To use both Kaggle GPU
sessions we split the four cells across TWO kernels (different accounts) and merge:

  Kernel A (baodqhust): VARIANT_SUBSET = baseline, rsl_rac   (the two "corners")
  Kernel B (baobaoo)  : VARIANT_SUBSET = rsl, rac            (the two single modules)

All four cells share seed=42, the same val/train split, the same MS-Celeb-1M init and
the SAME recipe, so they remain directly comparable across the two kernels.

RECIPE = the CURRENT BEST MS-Celeb MEK-RN18 config (results-CV-project.md, 2026-06-09):
  Adam LR=3e-4, wd=1e-4, ExponentialLR γ=0.9, 60 epochs, λ_flip=2, ε=0.1, 224px,
  MS-Celeb-1M backbone, weighted sampler. (NOT the paper's LR=1e-4 — the user asked
  for the best-results hyperparameters.) The ablation only toggles whether RSL/RAC are
  active; everything else is held fixed.

INTEGRITY: this transformer never edits the model / losses / data pipeline. It only
  (1) selects which of the 4 already-defined variants run, (2) sets LR (a documented
  best-results hyperparameter), (3) auto-detects the dataset root + face-weights mount
  (paths aren't knowable ahead of time on Kaggle), (4) prepends the P100-compatible
  torch bootstrap. The script writes its own ablation_resnet18_rafdb.json with all
  per-variant metrics, which the runner fetches.

  python kaggle/tuning/prep_ablation.py            # full runs (60 ep)
  python kaggle/tuning/prep_ablation.py --epochs 2 # 2-epoch smoke
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "kaggle", "ablation", "mek_ablation_resnet18.py")
RUNS = os.path.join(REPO, "kaggle", "tuning", "runs", "ablation")
RAF_SRC = "shuvoalok/raf-db-dataset"
FACE_SRC = "baodqhust/resnet18-msceleb"   # public; any account can attach it

# (account, slug, [variant subset])
SPLITS = [
    ("baodqhust", "rafdb-abl-msceleb-corners", ["baseline", "rsl_rac"]),
    ("baobaoo",   "rafdb-abl-msceleb-singles", ["rsl", "rac"]),
]

# ---- the P100-compatible torch bootstrap (verbatim from prepare_kernel.py) ----
_GPU_COMPAT_TORCH_BLOCK = '''\
# ===== injected by prep_ablation.py: P100-compatible torch bootstrap =====
import subprocess as _sp, sys as _sys
_gpu_name = ""
try:
    _gpu_name = _sp.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True, stderr=_sp.DEVNULL).strip()
except Exception:
    _gpu_name = ""
if any(_p in _gpu_name for _p in ("P100", "P40", "P4")):
    print("Pascal GPU detected (%s) -> installing P100-compatible torch..." % _gpu_name, flush=True)
    _sp.run([_sys.executable, "-m", "pip", "install", "-q",
             "torch==2.5.1", "torchvision==0.20.1",
             "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
else:
    print("GPU=%r -> keeping stock torch." % _gpu_name, flush=True)
# ===== end injected block =====

'''

# ---- dataset-root auto-detect, replaces the two hard-coded *_ROOT lines ----
_ROOT_OLD = (
    'FER_ROOT = "/kaggle/input/datasets/msambare/fer2013"\n'
    'RAF_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"'
)
_ROOT_NEW = '''# ===== injected by prep_ablation.py: robust dataset-root auto-detection =====
def _autodetect_split_root(base="/kaggle/input"):
    import os
    matches = []
    for root, dirs, _ in os.walk(base):
        if "train" in dirs and "test" in dirs:
            tr = os.path.join(root, "train")
            if any(os.path.isdir(os.path.join(tr, d)) for d in os.listdir(tr)):
                matches.append(root)
    if not matches:
        raise RuntimeError("Could not locate a train/+test/ dataset under %s" % base)
    matches.sort(key=len)
    return matches[0]
_DETECTED_ROOT = _autodetect_split_root()
FER_ROOT = _DETECTED_ROOT
RAF_ROOT = _DETECTED_ROOT
print("Auto-detected dataset root:", _DETECTED_ROOT)
# ===== end injected block ====='''

# ---- face-weights mount auto-detect, replaces the hard-coded FACE_WEIGHTS_PATH ----
_FACE_OLD = 'FACE_WEIGHTS_PATH = "/kaggle/input/resnet18-msceleb/resnet18_msceleb.pth"'
_FACE_NEW = '''# ===== injected by prep_ablation.py: face-weights path auto-detection =====
def _autodetect_face_weights(_name="resnet18_msceleb.pth", _base="/kaggle/input"):
    import os
    for _root, _dirs, _files in os.walk(_base):
        if _name in _files:
            return os.path.join(_root, _name)
    return None
FACE_WEIGHTS_PATH = _autodetect_face_weights() or "/kaggle/input/resnet18-msceleb/resnet18_msceleb.pth"
print("Face weights path:", FACE_WEIGHTS_PATH)
# ===== end injected block ====='''


def transform(src: str, subset: list, lr: str, epochs: int) -> str:
    # 1) variant subset
    old_sub = 'VARIANT_SUBSET = ["baseline", "rsl", "rac", "rsl_rac"]'
    if old_sub not in src:
        raise RuntimeError("VARIANT_SUBSET line not found in base script.")
    src = src.replace(old_sub, "VARIANT_SUBSET = %r" % (list(subset),), 1)

    # 2) LR (best-results hyperparameter)
    if "LR                  = 1e-4" not in src:
        raise RuntimeError("LR line not found.")
    src = src.replace("LR                  = 1e-4",
                      "LR                  = %s  # best MS-Celeb config" % lr, 1)

    # 3) dataset-root + face-weights auto-detect
    if _ROOT_OLD not in src:
        raise RuntimeError("*_ROOT lines not found.")
    src = src.replace(_ROOT_OLD, _ROOT_NEW, 1)
    if _FACE_OLD not in src:
        raise RuntimeError("FACE_WEIGHTS_PATH line not found.")
    src = src.replace(_FACE_OLD, _FACE_NEW, 1)

    # 4) optional epochs override (smoke). RAF-DB branch sets EPOCHS = 60.
    if epochs is not None:
        if "    EPOCHS             = 60" not in src:
            raise RuntimeError("rafdb EPOCHS line not found.")
        src = src.replace("    EPOCHS             = 60",
                          "    EPOCHS             = %d" % epochs, 1)

    # 5) prepend the P100 torch bootstrap (before any torch import)
    src = _GPU_COMPAT_TORCH_BLOCK + src
    return src


def prepare(account, slug, subset, lr, epochs):
    out = os.path.join(RUNS, slug)
    with open(BASE, "r", encoding="utf-8") as f:
        src = f.read()
    out_src = transform(src, subset, lr, epochs)
    compile(out_src, slug + "/script.py", "exec")   # fail fast on bad transform
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "script.py"), "w", encoding="utf-8") as f:
        f.write(out_src)
    meta = {
        "id": "%s/%s" % (account, slug),
        "title": slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [RAF_SRC, FACE_SRC],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Prepared %s -> %s  (variants=%s, LR=%s, epochs=%s)"
          % (meta["id"], out, subset, lr, epochs or 60))
    return account, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", default="3e-4", help="best MS-Celeb LR (default 3e-4)")
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. 2 for smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="prepare a single 2-epoch baodqhust kernel for end-to-end validation")
    args = ap.parse_args()

    os.environ["PYTHONUTF8"] = "1"
    if args.smoke:
        acc, out = prepare("baodqhust", "rafdb-abl-msceleb-smoke",
                           ["baseline", "rac"], args.lr, 2)
        qpath = os.path.join(RUNS, "queue_smoke.json")
        os.makedirs(RUNS, exist_ok=True)
        with open(qpath, "w") as f:
            json.dump([{"account": acc, "dir": out}], f, indent=2)
        print("queue ->", qpath)
        return

    queue = []
    for account, slug, subset in SPLITS:
        acc, out = prepare(account, slug, subset, args.lr, args.epochs)
        queue.append({"account": acc, "dir": out})
    qpath = os.path.join(RUNS, "queue.json")
    os.makedirs(RUNS, exist_ok=True)
    with open(qpath, "w") as f:
        json.dump(queue, f, indent=2)
    print("queue ->", qpath, "(%d kernels)" % len(queue))


if __name__ == "__main__":
    main()
