"""
Prepare a self-contained Kaggle kernel from one of the repo's single-file scripts
(kaggle/resnet18.py, resnet34.py, mek_resnet18.py, mek_resnet34.py).

INTEGRITY CONTRACT — this transformer only ever:
  1. sets DATASET to the chosen value,
  2. replaces the hard-coded RAF_ROOT/FER_ROOT with a robust auto-detector
     (Kaggle mounts an attached dataset at /kaggle/input/<slug>/..., NOT at the
     /kaggle/input/datasets/<owner>/<slug> path baked into the scripts),
  3. optionally overrides a whitelist of *hyperparameters* (LR, EPOCHS, ...),
  4. appends a metrics.json dump that only READS results (test_res, best_val_acc),
  5. optionally (--face-weights-name) re-points FACE_WEIGHTS_PATH to wherever the
     named weights file actually mounted under /kaggle/input (same robustness reason
     as the dataset root: the mount path is not knowable ahead of time). This only
     LOCATES the already-specified weights file — it does not change the method.
  6. optionally (--gpu-compat-torch) PREPENDS an environment-only bootstrap that, IF
     the assigned GPU is a Pascal card (P100/P40, sm_60 — Kaggle's API GPU default)
     unsupported by the stock image's torch, reinstalls a Pascal-compatible torch
     BEFORE any torch import. GPU is detected via nvidia-smi (NOT by importing torch,
     which would pin the stale module). T4 (sm_75) keeps the stock build. This changes
     only the runtime environment — never the model, losses, math, or data pipeline.
It never touches the model, the losses, the optimizer math, or the data pipeline.

Output: <out_dir>/script.py  +  <out_dir>/kernel-metadata.json

Usage:
  python kaggle/tuning/prepare_kernel.py kaggle/mek_resnet18.py \
      --slug mek-rn18-rafdb-default --username baodqhust \
      --dataset rafdb --dataset-source shuvoalok/raf-db-dataset \
      --out kaggle/tuning/runs/mek-rn18-rafdb-default
  # optional overrides (search phase / smoke test):
      --set EPOCHS=2 --set LR=5e-4
"""
import argparse
import json
import os
import re

# Only these names may be overridden from the CLI. Anything else is rejected so a
# typo can't silently no-op or, worse, inject arbitrary code into the kernel.
_ALLOWED_OVERRIDES = {
    "LR", "EPOCHS", "BATCH_SIZE", "DROPOUT", "LABEL_SMOOTH",
    "FLIP_LOSS_WEIGHT", "WEIGHT_DECAY", "EARLY_STOP_PATIENCE", "VAL_SPLIT", "SEED",
}

# Inserted right after `torch.backends.cudnn.benchmark = True`, which both script
# families contain and which sits after ALL config is defined but before the data
# pipeline / model / optimizer are built — so overrides take effect everywhere.
_SEED_MARKER = "torch.backends.cudnn.benchmark = True"

_AUTODETECT_BLOCK = '''
# ===== injected by prepare_kernel.py: robust dataset-root auto-detection =====
def _autodetect_split_root(base="/kaggle/input"):
    """Find the directory that directly contains both train/ and test/ subdirs.
    Kaggle mounts an attached dataset at /kaggle/input/<slug>/... and the exact
    depth varies (e.g. RAF-DB nests under .../DATASET/), so we search instead of
    hard-coding. Picks the shallowest match whose train/ holds class subfolders."""
    import os
    matches = []
    for root, dirs, _ in os.walk(base):
        if "train" in dirs and "test" in dirs:
            tr = os.path.join(root, "train")
            if any(os.path.isdir(os.path.join(tr, d)) for d in os.listdir(tr)):
                matches.append(root)
    if not matches:
        raise RuntimeError(f"Could not locate a train/+test/ dataset under {base}. "
                           f"Contents: {os.listdir(base) if os.path.isdir(base) else 'MISSING'}")
    matches.sort(key=len)
    return matches[0]

_DETECTED_ROOT = _autodetect_split_root()
FER_ROOT = _DETECTED_ROOT
RAF_ROOT = _DETECTED_ROOT
print("Auto-detected dataset root:", _DETECTED_ROOT)
# ===== end injected block =====
'''

_METRICS_BLOCK = '''

# ===== injected by prepare_kernel.py: write metrics.json for the tuning harness =====
def _dump_tuning_metrics():
    import json
    g = globals()
    tr = g.get("test_res") or {}
    out = {
        "arch": g.get("ARCH"),
        "dataset": g.get("DATASET"),
        "best_val_acc": g.get("best_val_acc"),
        "test_accuracy": tr.get("accuracy"),
        "test_f1": tr.get("f1"),
        "test_precision": tr.get("precision"),
        "test_recall": tr.get("recall"),
        "test_mean_class_acc": tr.get("mean_class_acc"),
        "tta_accuracy": g.get("tta_acc"),
        "confmat": tr.get("confmat"),
    }
    keys = ["LR", "BATCH_SIZE", "EPOCHS", "DROPOUT", "LABEL_SMOOTH",
            "FLIP_LOSS_WEIGHT", "WEIGHT_DECAY", "SEED"]
    out["hyperparams"] = {k: g.get(k) for k in keys if k in g}
    h = g.get("history")
    if isinstance(h, dict) and h.get("accuracy"):
        out["epochs_run"] = len(h["accuracy"])
    cls, pca = g.get("CLASSES"), tr.get("per_class_acc")
    if cls is not None and pca is not None:
        out["per_class_acc"] = {str(c): float(a) for c, a in zip(cls, pca)}
    # Validation mean-class (balanced acc) of the restored best model — a
    # selection signal aligned with the imbalance objective (no test peeking).
    ev, vl = g.get("evaluate"), g.get("val_loader")
    if callable(ev) and vl is not None:
        try:
            vres = ev(vl, return_per_class=True)
            out["val_mean_class_acc"] = vres.get("mean_class_acc")
            out["val_accuracy"] = vres.get("accuracy")
        except Exception as _ve:
            print("val mean-class eval failed:", _ve)
    with open("/kaggle/working/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("TUNING_METRICS_JSON_WRITTEN", json.dumps(out))

try:
    _dump_tuning_metrics()
except Exception as _e:
    print("WARNING: metrics dump failed:", _e)
# ===== end injected block =====
'''


_GPU_COMPAT_TORCH_BLOCK = '''\
# ===== injected by prepare_kernel.py: P100-compatible torch bootstrap =====
# Kaggle's current image ships torch 2.10+cu128 (compiled for sm_70-sm_120 only). The
# API-default GPU is a Pascal P100 (sm_60), which that build CANNOT run ("no kernel
# image"). If a Pascal card is assigned, reinstall a Pascal-compatible torch BEFORE any
# torch import. GPU is probed via nvidia-smi (importing torch here would cache the stale
# module and defeat the reinstall). A T4 (sm_75) keeps the stock build. Env-only.
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


def build_face_autodetect_block(weights_name: str) -> str:
    """Re-point FACE_WEIGHTS_PATH to the real mount of `weights_name`. Injected after
    the seed marker, i.e. BEFORE build_face_backbone() runs, so the reassignment is
    picked up. Robust to Kaggle's /kaggle/input/<slug> vs /kaggle/input/datasets/<owner>/<slug>
    mount layouts (same reason as the dataset-root auto-detector)."""
    return (
        "\n# ===== injected by prepare_kernel.py: face-weights path auto-detection =====\n"
        "def _autodetect_face_weights(_name, _base='/kaggle/input'):\n"
        "    import os\n"
        "    for _root, _dirs, _files in os.walk(_base):\n"
        "        if _name in _files:\n"
        "            return os.path.join(_root, _name)\n"
        "    return None\n"
        f"_FW = _autodetect_face_weights({weights_name!r})\n"
        "if _FW:\n"
        "    FACE_WEIGHTS_PATH = _FW\n"
        "    print('Auto-detected face weights:', _FW)\n"
        "else:\n"
        "    print('WARNING: face-weights file %r not found under /kaggle/input "
        "(will fall back to ImageNet).' % " + repr(weights_name) + ")\n"
        "# ===== end injected block =====\n"
    )


def parse_override_value(v: str):
    """Turn a CLI string into int/float/bool where it clearly is one; else keep str."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def build_override_block(overrides: dict) -> str:
    if not overrides:
        return ""
    lines = ["", "# ===== injected by prepare_kernel.py: hyperparameter overrides ====="]
    for k, v in overrides.items():
        lines.append(f"{k} = {v!r}")
    lines.append("print('Applied overrides:', " + repr(overrides) + ")")
    lines.append("# ===== end injected block =====")
    return "\n".join(lines) + "\n"


def transform(src: str, dataset: str, overrides: dict, sched_gamma: float = None,
              face_weights_name: str = None, gpu_compat_torch: bool = False) -> str:
    # 1) Force the dataset toggle. Matches `DATASET = "..."` / `DATASET  = "..."`.
    src, n = re.subn(r'^DATASET(\s*)=\s*".*?"',
                     lambda m: f'DATASET{m.group(1)}= "{dataset}"',
                     src, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError("Could not find the `DATASET = \"...\"` toggle to set.")

    # 2) Robust dataset-root auto-detection + overrides, injected right after seeds.
    if _SEED_MARKER not in src:
        raise RuntimeError(f"Marker not found: {_SEED_MARKER!r} (script layout changed?)")
    inject = _AUTODETECT_BLOCK + build_override_block(overrides)
    if face_weights_name:
        inject += build_face_autodetect_block(face_weights_name)
    src = src.replace(_SEED_MARKER, _SEED_MARKER + "\n" + inject, 1)

    # 3) Re-derive DATA_ROOT/TRAIN_DIR/TEST_DIR AFTER detection, because the scripts
    #    set them inside the dataset block (which already ran textually above the
    #    injection point at *import* time? No — Python runs top-to-bottom, and the
    #    dataset block is ABOVE the seed marker). So we must re-point them here.
    #    Both families use DATA_ROOT/TRAIN_DIR/TEST_DIR identically.
    repoint = (
        "\n# ===== injected: re-point data dirs to the detected root =====\n"
        "DATA_ROOT = _DETECTED_ROOT\n"
        'TRAIN_DIR = f"{DATA_ROOT}/train"\n'
        'TEST_DIR  = f"{DATA_ROOT}/test"\n'
        'print("Using TRAIN_DIR:", TRAIN_DIR, "| TEST_DIR:", TEST_DIR)\n'
        "# ===== end injected block =====\n"
    )
    src = src.replace(_SEED_MARKER + "\n" + inject,
                      _SEED_MARKER + "\n" + inject + repoint, 1)

    # 4) Optional: change the MEK ExponentialLR decay rate (inline kwarg, not a
    #    constant). Targeted value-only replace — legitimate hyperparameter tuning.
    if sched_gamma is not None:
        marker = "ExponentialLR(optimizer, gamma=0.9)"
        if marker not in src:
            raise RuntimeError("sched_gamma given but 'ExponentialLR(optimizer, gamma=0.9)' "
                               "not found (only MEK scripts use it).")
        src = src.replace(marker, f"ExponentialLR(optimizer, gamma={sched_gamma})", 1)

    # 5) Append the metrics dump at the very end.
    src = src + _METRICS_BLOCK

    # 6) Prepend the P100-compatible torch bootstrap (must run before any torch import).
    if gpu_compat_torch:
        src = _GPU_COMPAT_TORCH_BLOCK + src
    return src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_script")
    p.add_argument("--slug", required=True, help="kernel slug (lowercase, dashes)")
    p.add_argument("--username", required=True)
    p.add_argument("--dataset", required=True, choices=["rafdb", "fer2013"])
    p.add_argument("--dataset-source", required=True,
                   help="Kaggle owner/slug to attach, e.g. shuvoalok/raf-db-dataset")
    p.add_argument("--extra-dataset-source", action="append", default=[],
                   metavar="OWNER/SLUG",
                   help="additional dataset to attach (repeatable), e.g. the "
                        "resnet18-msceleb face-backbone weights for MS-Celeb runs")
    p.add_argument("--out", required=True, help="output kernel directory")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="hyperparameter override (repeatable)")
    p.add_argument("--sched-gamma", type=float, default=None,
                   help="MEK only: ExponentialLR decay rate (default 0.9; higher = slower decay)")
    p.add_argument("--face-weights-name", default=None, metavar="FILENAME",
                   help="face-backbone runs: re-point FACE_WEIGHTS_PATH to wherever this "
                        "file actually mounts under /kaggle/input (e.g. resnet18_msceleb.pth)")
    p.add_argument("--gpu-compat-torch", action="store_true",
                   help="prepend a bootstrap that reinstalls a Pascal-compatible torch if "
                        "Kaggle assigns a P100 (sm_60), which the stock image's torch can't run")
    args = p.parse_args()

    overrides = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in _ALLOWED_OVERRIDES:
            raise SystemExit(f"Override {k!r} not in allow-list {sorted(_ALLOWED_OVERRIDES)}")
        overrides[k] = parse_override_value(v.strip())

    with open(args.base_script, "r", encoding="utf-8") as f:
        src = f.read()
    out_src = transform(src, args.dataset, overrides, sched_gamma=args.sched_gamma,
                        face_weights_name=args.face_weights_name,
                        gpu_compat_torch=args.gpu_compat_torch)

    # Fail fast if the transform produced invalid Python.
    compile(out_src, args.slug + "/script.py", "exec")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "script.py"), "w", encoding="utf-8") as f:
        f.write(out_src)

    meta = {
        "id": f"{args.username}/{args.slug}",
        "title": args.slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # Force a T4 (sm_75). Kaggle's default P100 (sm_60) is NOT supported by the
        # current preinstalled PyTorch (sm_70+), which fails with
        # "CUDA error: no kernel image is available for execution on the device".
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,          # torchvision downloads ImageNet weights
        "dataset_sources": [args.dataset_source] + list(args.extra_dataset_source),
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(args.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Prepared kernel '{meta['id']}' -> {args.out}")
    print(f"  dataset={args.dataset}  source={args.dataset_source}  overrides={overrides or 'none'}")


if __name__ == "__main__":
    main()
