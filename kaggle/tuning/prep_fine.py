"""
Prepare the FER-2013 FINE-sweep kernels (round 2) and emit queue_fine.json.

Driven by the coarse results (single seed):
  * MEK improves monotonically as lambda_flip drops (2.0 -> 1.0 -> 0.5); LR best at 1e-4.
    Fine: probe EVEN LOWER lambda (0.25, 0.1) for both archs to find the peak / limit.
  * Baselines plateau ~0.66 at LR1e-2/WD1e-4/dropout0.40; not visibly overfitting.
    Fine: probe LESS dropout (0.30) for both archs (may lift the slightly-underfit nets,
    and try to give RN34 the overall-acc win too, not just mean-class).

Long descriptive slugs (Kaggle rejects short slugs). Single seed (42).
  python kaggle/tuning/prep_fine.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(REPO, "kaggle", "tuning", "prepare_kernel.py")
RUNS = os.path.join(REPO, "kaggle", "tuning", "runs")
USER = "baobaoo"
SRC = "msambare/fer2013"

CONFIGS = [
    # ---- MEK: push lambda lower (LR 1e-4, dropout = each script's default) ----
    ("fer2013-mek-resnet18-lr1e4-l025", "mek_resnet18.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.25"}),
    ("fer2013-mek-resnet18-lr1e4-l01",  "mek_resnet18.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.1"}),
    ("fer2013-mek-resnet34-lr1e4-l025", "mek_resnet34.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.25"}),
    ("fer2013-mek-resnet34-lr1e4-l01",  "mek_resnet34.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.1"}),
    # ---- baselines: less dropout at the winning LR1e-2/WD1e-4 ----
    ("fer2013-resnet18-lr1e2-wd1e4-do30", "resnet18.py", {"LR": "1e-2", "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.30"}),
    ("fer2013-resnet34-lr1e2-wd1e4-do30", "resnet34.py", {"LR": "1e-2", "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.30"}),
]


def main():
    os.environ["PYTHONUTF8"] = "1"
    queue = []
    for slug, base, ov in CONFIGS:
        out = os.path.join(RUNS, slug)
        cmd = [sys.executable, PREP, os.path.join("kaggle", base),
               "--slug", slug, "--username", USER,
               "--dataset", "fer2013", "--dataset-source", SRC, "--out", out]
        for k, v in ov.items():
            cmd += ["--set", f"{k}={v}"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            print(f"FAILED {slug}:\n{r.stdout}\n{r.stderr}")
            sys.exit(1)
        print(r.stdout.strip())
        queue.append(out)

    qpath = os.path.join(REPO, "kaggle", "tuning", "queue_fine.json")
    with open(qpath, "w") as f:
        json.dump(queue, f, indent=2)
    print(f"\nPrepared {len(queue)} fine kernels. Queue -> {qpath}")


if __name__ == "__main__":
    main()
