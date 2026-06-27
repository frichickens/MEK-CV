"""
Campaign orchestrator: drive a QUEUE of already-prepared kernels through Kaggle while
respecting the account's 2-concurrent-GPU-session cap.

Design (poison-safe). Kaggle CORRUPTS a kernel's server record if you push it repeatedly
before it has been created (it then returns "Notebook not found" forever). So this
orchestrator NEVER hammers:
  * Each cycle it FIRST adopts/harvests every queue kernel that is already running or
    complete (status check), updating `active`.
  * THEN, only with the leftover believed-free capacity (max_concurrent - len(active)),
    it pushes new kernels, attempting EACH remaining kernel AT MOST ONCE PER CYCLE.
  * A push that fails with a "quota full" message (cap / "notebook not found" on a new
    slug) is NOT a real failure -> the kernel goes to the back of the queue and is retried
    on a LATER cycle (≈poll_secs apart, gated on free capacity) — never in a tight loop.
Resumable: any run dir that already has result.json is DONE and skipped.

  python kaggle/tuning/run_campaign.py --queue kaggle/tuning/queue_coarse.json \
      --config-dir C:\\Users\\baolo\\.kaggle2 --max-concurrent 2 --poll-secs 120

INTEGRITY: only pushes kernels already prepared by prepare_kernel.py; never edits a script
or hyperparameter; only reads results back.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque


def _find_kaggle_exe():
    """Locate the kaggle console script. This kaggle version has no runnable
    `python -m kaggle` entry point, so we call the installed `kaggle[.exe]`
    console script directly (also avoids the repo's own `kaggle/` dir shadowing
    the installed package on sys.path)."""
    exe = shutil.which("kaggle")
    if exe:
        return exe
    scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
    cand = os.path.join(scripts, "kaggle.exe" if os.name == "nt" else "kaggle")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(os.path.dirname(sys.executable), "kaggle")
    if os.path.exists(cand2):
        return cand2
    raise SystemExit("Could not locate the `kaggle` console script. "
                     "Install it into this env: pip install kaggle")


KAGGLE = [_find_kaggle_exe()]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def kernel_id(run_dir):
    with open(os.path.join(run_dir, "kernel-metadata.json")) as f:
        return json.load(f)["id"]


def push(run_dir, accelerator=None):
    cmd = KAGGLE + ["kernels", "push", "-p", run_dir]
    if accelerator:
        # `--accelerator` (e.g. NvidiaTeslaT4) needs the newer kaggle-cli (GitHub main);
        # PyPI kaggle<=1.7.4.5 doesn't have it and will error if it's passed.
        cmd += ["--accelerator", accelerator]
    r = run(cmd)
    out = (r.stdout + r.stderr)
    low = out.lower()
    ok = r.returncode == 0 and "successfully pushed" in low
    quota_full = ("maximum batch gpu session" in low) or ("notebook not found" in low)
    return ok, quota_full, out.strip()


def status(kid):
    r = run(KAGGLE + ["kernels", "status", kid])
    text = (r.stdout + r.stderr).lower()
    for s in ("complete", "error", "cancelacknowledged", "cancelrequested",
              "running", "queued"):
        if s in text:
            return s
    return "absent"     # not found / never created


def fetch_and_parse(run_dir, kid):
    out_dir = os.path.join(run_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    run(KAGGLE + ["kernels", "output", kid, "-p", out_dir])
    mpath = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, "result.json"))


def load_result(run_dir):
    try:
        with open(os.path.join(run_dir, "result.json")) as f:
            return json.load(f)
    except Exception:
        return None


def write_summary(queue, summary_path):
    cols = ["slug", "arch", "dataset", "status", "test_accuracy",
            "test_mean_class_acc", "test_f1", "best_val_acc",
            "LR", "BATCH_SIZE", "EPOCHS", "DROPOUT", "LABEL_SMOOTH",
            "FLIP_LOSS_WEIGHT", "WEIGHT_DECAY", "SEED", "epochs_run"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in queue:
            slug = os.path.basename(d.rstrip("/\\"))
            m = load_result(d)
            row = {"slug": slug, "status": "done" if m else "pending"}
            if m:
                hp = m.get("hyperparams", {}) or {}
                for k in ("arch", "dataset", "test_accuracy",
                          "test_mean_class_acc", "test_f1", "best_val_acc",
                          "epochs_run"):
                    row[k] = m.get(k)
                for k in ("LR", "BATCH_SIZE", "EPOCHS", "DROPOUT", "LABEL_SMOOTH",
                          "FLIP_LOSS_WEIGHT", "WEIGHT_DECAY", "SEED"):
                    row[k] = hp.get(k)
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--max-concurrent", type=int, default=2)
    p.add_argument("--poll-secs", type=int, default=120)
    p.add_argument("--max-hours", type=float, default=48.0)
    p.add_argument("--accelerator", default=None,
                   help="GPU accelerator id passed to `kernels push` (e.g. NvidiaTeslaT4). "
                        "Requires the newer kaggle-cli (GitHub main); omit for P100 default.")
    args = p.parse_args()

    if args.config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = args.config_dir
    os.environ["PYTHONUTF8"] = "1"

    with open(args.queue) as f:
        queue = json.load(f)
    summary_path = os.path.join(os.path.dirname(args.queue), "campaign_summary.csv")

    remaining = deque(d for d in queue if not is_done(d))
    print(f"[campaign] {len(queue)} total, {len(queue) - len(remaining)} already done, "
          f"{len(remaining)} to run.", flush=True)
    write_summary(queue, summary_path)

    active = {}        # run_dir -> kid
    fails = {}         # run_dir -> count of genuinely-different (non-quota) push errors
    deadline = time.time() + args.max_hours * 3600

    while (remaining or active) and time.time() < deadline:
        # 1) Poll active kernels; harvest finished ones.
        for d, kid in list(active.items()):
            st = status(kid)
            if st in ("complete", "error", "cancelacknowledged", "cancelrequested"):
                if st == "complete":
                    m = fetch_and_parse(d, kid)
                    print(f"[campaign] DONE {kid}: acc={m.get('test_accuracy') if m else None} "
                          f"mca={m.get('test_mean_class_acc') if m else None}", flush=True)
                else:
                    print(f"[campaign] {kid} ended status={st} (no metrics)", flush=True)
                del active[d]
                write_summary(queue, summary_path)

        # 2) Adopt any remaining kernel that is already running/complete (no push).
        for _ in range(len(remaining)):
            d = remaining.popleft()
            kid = kernel_id(d)
            st0 = status(kid)
            if st0 == "complete":
                m = fetch_and_parse(d, kid)
                print(f"[campaign] ADOPT-DONE {kid}", flush=True)
                write_summary(queue, summary_path)
            elif st0 in ("running", "queued"):
                print(f"[campaign] ADOPT-RUNNING {kid}", flush=True)
                active[d] = kid
            else:
                remaining.append(d)   # keep for the push phase

        # 3) Push new kernels with leftover capacity, EACH AT MOST ONCE this cycle.
        capacity = args.max_concurrent - len(active)
        n = len(remaining)
        i = 0
        while capacity > 0 and i < n:
            d = remaining.popleft()
            i += 1
            kid = kernel_id(d)
            ok, quota_full, msg = push(d, accelerator=args.accelerator)
            if ok:
                print(f"[campaign] PUSHED {kid}", flush=True)
                active[d] = kid
                capacity -= 1
            elif quota_full:
                remaining.append(d)   # slots really full -> retry a later cycle
                capacity -= 1         # consume the believed-free credit; no re-spin now
            else:
                fails[d] = fails.get(d, 0) + 1
                if fails[d] >= 5:
                    print(f"[campaign] PUSH FAILED {kid} x{fails[d]}: {msg[:140]} -- skip",
                          flush=True)
                else:
                    print(f"[campaign] push err {kid} ({fails[d]}/5): {msg[:100]}", flush=True)
                    remaining.append(d)
                    capacity -= 1

        if remaining or active:
            time.sleep(args.poll_secs)

    write_summary(queue, summary_path)
    print(f"[campaign] finished. summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
