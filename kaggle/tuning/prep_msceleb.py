"""
Prepare the RAF-DB MS-Celeb-1M MEK-ResNet-18 tuning kernels (one dir per config) and
emit one queue JSON per Kaggle account for run_campaign.py.

Base script: kaggle/mek_resnet18_MS-Celeb-1M.py  (face backbone baked in; expects the
weights mounted at /kaggle/input/resnet18-msceleb/resnet18_msceleb.pth). Every kernel
attaches TWO datasets: the RAF-DB images + the public resnet18-msceleb weights.

Sweep = ONE-FACTOR-AT-A-TIME around the PAPER anchor (arXiv 2310.19636, Sec 4.2):
  Adam lr=1e-4, wd=1e-4, ExponentialLR gamma=0.9, 60 epochs, lambda_flip=2, eps=0.1, 224px.
Published RN18/RAF-DB: 89.77 acc / 82.44 mean (last epoch); 89.80 / 84.05 (best).

Split across 3 accounts (2 GPU sessions each = 6 concurrent):
  baodqhust   : paper anchor + LR sweep (the headline reproduction)
  baobaoo     : lambda + epsilon (the two MEK re-balance levers)
  quocbaohust : LR-schedule gamma + longer training

Integrity: only HP overrides on prepare_kernel's allow-list (+ --sched-gamma for the
MEK ExponentialLR decay). Never edits model / loss / data-pipeline code. Single seed 42.

  python kaggle/tuning/prep_msceleb.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(REPO, "kaggle", "tuning", "prepare_kernel.py")
RUNS = os.path.join(REPO, "kaggle", "tuning", "runs", "msceleb")
BASE = "mek_resnet18_MS-Celeb-1M.py"
RAF_SRC = "shuvoalok/raf-db-dataset"
FACE_SRC = "baodqhust/resnet18-msceleb"   # public; mounts at /kaggle/input/resnet18-msceleb/

# (account, slug, {overrides}, sched_gamma|None)
# Slugs are long/descriptive: Kaggle rejects short cryptic slugs ("Notebook not found").
CONFIGS = [
    # ----- baodqhust: paper anchor + LR sweep -----
    ("baodqhust",   "rafdb-mek-rn18-msceleb-paper",      {},                         None),
    ("baodqhust",   "rafdb-mek-rn18-msceleb-lr2e4",      {"LR": "2e-4"},             None),
    ("baodqhust",   "rafdb-mek-rn18-msceleb-lr3e4",      {"LR": "3e-4"},             None),
    # ----- baobaoo: lambda + epsilon levers -----
    ("baobaoo",     "rafdb-mek-rn18-msceleb-lam1",       {"FLIP_LOSS_WEIGHT": "1.0"}, None),
    ("baobaoo",     "rafdb-mek-rn18-msceleb-lam3",       {"FLIP_LOSS_WEIGHT": "3.0"}, None),
    ("baobaoo",     "rafdb-mek-rn18-msceleb-eps02",      {"LABEL_SMOOTH": "0.2"},     None),
    # ----- quocbaohust: schedule gamma + longer training -----
    ("quocbaohust", "rafdb-mek-rn18-msceleb-g095",       {},                         0.95),
    ("quocbaohust", "rafdb-mek-rn18-msceleb-ep80",       {"EPOCHS": "80"},           None),
    ("quocbaohust", "rafdb-mek-rn18-msceleb-ep80-g095",  {"EPOCHS": "80"},           0.95),
]


def prepare(account, slug, overrides, sched_gamma):
    out = os.path.join(RUNS, account, slug)
    cmd = [sys.executable, PREP, os.path.join("kaggle", BASE),
           "--slug", slug, "--username", account,
           "--dataset", "rafdb", "--dataset-source", RAF_SRC,
           "--extra-dataset-source", FACE_SRC,
           "--face-weights-name", "resnet18_msceleb.pth",
           "--gpu-compat-torch", "--out", out]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    if sched_gamma is not None:
        cmd += ["--sched-gamma", str(sched_gamma)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(f"FAILED {slug}:\n{r.stdout}\n{r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())
    return out


def main():
    os.environ["PYTHONUTF8"] = "1"
    by_account = {}
    for account, slug, ov, g in CONFIGS:
        out = prepare(account, slug, ov, g)
        by_account.setdefault(account, []).append(out)

    for account, dirs in by_account.items():
        qpath = os.path.join(RUNS, account, "queue.json")
        with open(qpath, "w") as f:
            json.dump(dirs, f, indent=2)
        print(f"queue[{account}] -> {qpath}  ({len(dirs)} kernels)")
    print(f"\nPrepared {len(CONFIGS)} kernels across {len(by_account)} accounts.")


if __name__ == "__main__":
    main()
