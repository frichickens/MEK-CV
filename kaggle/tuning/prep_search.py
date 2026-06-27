"""
Prepare the FER-2013 coarse-search kernels (one dir per config) and emit a queue JSON
for run_campaign.py. Calls prepare_kernel.py as a subprocess per config so the integrity
contract (HP-overrides-only, allow-listed names, compile-check) is enforced identically.

Single seed (42, the script default). Selection later = best validation accuracy
(the in-script model-selection criterion) with mean-class acc reported as headline.

  python kaggle/tuning/prep_search.py
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

# (slug, base_script, {overrides})  -- baselines first (cheap, fast feedback), MEK after.
# Baselines: grid over LR x WEIGHT_DECAY, dropout fixed 0.40 for a fair RN18-vs-RN34
#   comparison (RN18 default LR1e-2/WD1e-4/do0.4 already ran -> excluded).
# MEK: L-shaped sweep around the default (LR1e-4, lambda=2) on the two known MEK levers,
#   LR (compensates the aggressive ExponentialLR gamma=0.9 decay) and FLIP_LOSS_WEIGHT.
# NB: Kaggle rejects short kernel slugs with "Notebook not found" -- slugs must be long &
# descriptive (verified: "r18-lr7e3-wd1e4" fails, "fer2013-resnet18-lr7e3-wd1e4" works).
# The (resnet18, LR7e-3, WD1e-4) cell is already running from that proven push, so it is
# salvaged separately (run dir fer2013-resnet18-lr7e3-wd1e4) and omitted here.
CONFIGS = [
    # ---- baseline RN18 (kaggle/resnet18.py), dropout 0.40 ----
    ("fer2013-resnet18-lr7e3-wd5e4",  "resnet18.py", {"LR": "7e-3",  "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet18-lr1e2-wd5e4",  "resnet18.py", {"LR": "1e-2",  "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet18-lr14e3-wd1e4", "resnet18.py", {"LR": "1.4e-2", "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet18-lr14e3-wd5e4", "resnet18.py", {"LR": "1.4e-2", "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    # ---- baseline RN34 (kaggle/resnet34.py), dropout 0.40 (override from 0.45) ----
    ("fer2013-resnet34-lr7e3-wd1e4",  "resnet34.py", {"LR": "7e-3",  "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet34-lr7e3-wd5e4",  "resnet34.py", {"LR": "7e-3",  "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet34-lr1e2-wd1e4",  "resnet34.py", {"LR": "1e-2",  "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet34-lr1e2-wd5e4",  "resnet34.py", {"LR": "1e-2",  "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet34-lr14e3-wd1e4", "resnet34.py", {"LR": "1.4e-2", "WEIGHT_DECAY": "1e-4", "DROPOUT": "0.40"}),
    ("fer2013-resnet34-lr14e3-wd5e4", "resnet34.py", {"LR": "1.4e-2", "WEIGHT_DECAY": "5e-4", "DROPOUT": "0.40"}),
    # ---- MEK RN18 (kaggle/mek_resnet18.py) ----
    ("fer2013-mek-resnet18-lr3e4-l2",  "mek_resnet18.py", {"LR": "3e-4", "FLIP_LOSS_WEIGHT": "2.0"}),
    ("fer2013-mek-resnet18-lr5e4-l2",  "mek_resnet18.py", {"LR": "5e-4", "FLIP_LOSS_WEIGHT": "2.0"}),
    ("fer2013-mek-resnet18-lr1e4-l1",  "mek_resnet18.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "1.0"}),
    ("fer2013-mek-resnet18-lr1e4-l05", "mek_resnet18.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.5"}),
    # ---- MEK RN34 (kaggle/mek_resnet34.py) ----
    ("fer2013-mek-resnet34-lr3e4-l2",  "mek_resnet34.py", {"LR": "3e-4", "FLIP_LOSS_WEIGHT": "2.0"}),
    ("fer2013-mek-resnet34-lr5e4-l2",  "mek_resnet34.py", {"LR": "5e-4", "FLIP_LOSS_WEIGHT": "2.0"}),
    ("fer2013-mek-resnet34-lr1e4-l1",  "mek_resnet34.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "1.0"}),
    ("fer2013-mek-resnet34-lr1e4-l05", "mek_resnet34.py", {"LR": "1e-4", "FLIP_LOSS_WEIGHT": "0.5"}),
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

    qpath = os.path.join(REPO, "kaggle", "tuning", "queue_coarse.json")
    with open(qpath, "w") as f:
        json.dump(queue, f, indent=2)
    print(f"\nPrepared {len(queue)} kernels. Queue -> {qpath}")


if __name__ == "__main__":
    main()
