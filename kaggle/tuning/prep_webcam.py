"""
Prepare the RAF-DB webcam-DEPLOY MEK-ResNet18 tuning kernels (MS-Celeb-1M backbone,
CLAHE + OpenCV face-crop, heavy webcam aug + EMA) and emit one queue per Kaggle account.

Base script: kaggle/mek_webcam_resnet18_clahe_facecrop.py — train/eval transforms match
demo.py's live preprocessing exactly (Haar face detect&crop -> CLAHE), so the deployed
model sees the distribution the webcam feeds it.

Search = ONE-FACTOR-AT-A-TIME around the webcam recipe (ε=0.15, λ_flip=0.1, Adam lr=1e-4,
ExponentialLR γ=0.9, EMA=0.999, 60 epochs). The heavy webcam augmentation + EMA + face-crop
+ CLAHE are kept FIXED on every run — they are the deployment-robustness levers, not things
to tune away to chase clean-test accuracy. We only tune the optimisation (LR, γ, epochs) and
the loss shape (ε, λ). Selection = VALIDATION mean-class (overall-acc guardrail); the val/test
sets are themselves face-crop+CLAHE'd, so the proxy matches the deployment distribution.

Split across 3 accounts (2 GPU sessions each = 6 concurrent). P100 + the --gpu-compat-torch
bootstrap (auto-uses a T4 natively if Kaggle ever assigns one).

  python kaggle/tuning/prep_webcam.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(REPO, "kaggle", "tuning", "prepare_kernel.py")
RUNS = os.path.join(REPO, "kaggle", "tuning", "runs", "webcam")
BASE = "mek_webcam_resnet18_clahe_facecrop.py"
RAF_SRC = "shuvoalok/raf-db-dataset"
FACE_SRC = "baodqhust/resnet18-msceleb"

# (account, slug, {overrides}, sched_gamma|None)
CONFIGS = [
    # ----- baodqhust: webcam recipe + LR sweep -----
    ("baodqhust",   "rafdb-webcam-cf-default",     {},                          None),
    ("baodqhust",   "rafdb-webcam-cf-lr2e4",       {"LR": "2e-4"},              None),
    ("baodqhust",   "rafdb-webcam-cf-lr3e4",       {"LR": "3e-4"},              None),
    # ----- baobaoo: schedule + epochs + epsilon -----
    ("baobaoo",     "rafdb-webcam-cf-g095",        {},                          0.95),
    ("baobaoo",     "rafdb-webcam-cf-ep80-g095",   {"EPOCHS": "80"},            0.95),
    ("baobaoo",     "rafdb-webcam-cf-eps01",       {"LABEL_SMOOTH": "0.1"},     None),
    # ----- quocbaohust: lambda (attention-consistency) + best-combo -----
    ("quocbaohust", "rafdb-webcam-cf-lam05",       {"FLIP_LOSS_WEIGHT": "0.5"}, None),
    ("quocbaohust", "rafdb-webcam-cf-lam1",        {"FLIP_LOSS_WEIGHT": "1.0"}, None),
    ("quocbaohust", "rafdb-webcam-cf-lr2e4-g095",  {"LR": "2e-4"},              0.95),
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
