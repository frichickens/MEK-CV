"""
Prepare a clean lambda_flip sweep at the BEST LR (3e-4) for the MS-Celeb MEK-RN18,
so we can plot a faithful comparison against the original paper's Fig 3 (left panel).

Everything is held fixed at the validation-selected winner `msceleb-lr3e4`
(Adam LR=3e-4, wd=1e-4, ExponentialLR gamma=0.9, 60 epochs, eps=0.1, 224px, MS-Celeb
backbone). ONLY lambda_flip changes. Combined with the existing lr3e4 run (lambda=2,
mean-class 0.8137) this yields a 4-point curve lambda in {1, 2, 3, 4}.

One config per account so all three run in parallel.

  python kaggle/tuning/prep_lambda3e4.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(REPO, "kaggle", "tuning", "prepare_kernel.py")
RUNS = os.path.join(REPO, "kaggle", "tuning", "runs", "lambda3e4")
BASE = "mek_resnet18_MS-Celeb-1M.py"
RAF_SRC = "shuvoalok/raf-db-dataset"
FACE_SRC = "baodqhust/resnet18-msceleb"

# (account, slug, {overrides})  -- LR fixed 3e-4, only lambda_flip varies.
CONFIGS = [
    ("baodqhust",   "rafdb-mek-rn18-msceleb-lr3e4-lam1", {"LR": "3e-4", "FLIP_LOSS_WEIGHT": "1.0"}),
    ("baobaoo",     "rafdb-mek-rn18-msceleb-lr3e4-lam3", {"LR": "3e-4", "FLIP_LOSS_WEIGHT": "3.0"}),
    ("quocbaohust", "rafdb-mek-rn18-msceleb-lr3e4-lam4", {"LR": "3e-4", "FLIP_LOSS_WEIGHT": "4.0"}),
]


def prepare(account, slug, overrides):
    out = os.path.join(RUNS, account, slug)
    cmd = [sys.executable, PREP, os.path.join("kaggle", BASE),
           "--slug", slug, "--username", account,
           "--dataset", "rafdb", "--dataset-source", RAF_SRC,
           "--extra-dataset-source", FACE_SRC,
           "--face-weights-name", "resnet18_msceleb.pth",
           "--gpu-compat-torch", "--out", out]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(f"FAILED {slug}:\n{r.stdout}\n{r.stderr}")
        sys.exit(1)
    print(r.stdout.strip())
    return out


def main():
    os.environ["PYTHONUTF8"] = "1"
    by_account = {}
    for account, slug, ov in CONFIGS:
        out = prepare(account, slug, ov)
        by_account.setdefault(account, []).append(out)
    for account, dirs in by_account.items():
        qpath = os.path.join(RUNS, account, "queue.json")
        os.makedirs(os.path.dirname(qpath), exist_ok=True)
        with open(qpath, "w") as f:
            json.dump(dirs, f, indent=2)
        print(f"queue[{account}] -> {qpath}  ({len(dirs)} kernels)")
    print(f"\nPrepared {len(CONFIGS)} kernels across {len(by_account)} accounts.")


if __name__ == "__main__":
    main()
