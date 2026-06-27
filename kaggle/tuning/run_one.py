"""
Push one prepared kernel to Kaggle, poll until it finishes, fetch its output, and
parse metrics.json. Designed to be run in the background (a kernel can take hours).

  python kaggle/tuning/run_one.py --dir kaggle/tuning/runs/<slug>

Writes <dir>/result.json (parsed metrics) and leaves the raw kernel output in
<dir>/output/. Exit code 0 = kernel completed and metrics parsed; non-zero = error.
"""
import argparse
import json
import os
import subprocess
import sys
import time

KAGGLE = [sys.executable, "-m", "kaggle"]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kernel_id(run_dir: str) -> str:
    with open(os.path.join(run_dir, "kernel-metadata.json")) as f:
        return json.load(f)["id"]


def push(run_dir: str):
    r = run(KAGGLE + ["kernels", "push", "-p", run_dir])
    print(r.stdout.strip()); print(r.stderr.strip(), file=sys.stderr)
    if r.returncode != 0:
        raise SystemExit(f"push failed (rc={r.returncode})")


def status(kid: str) -> str:
    r = run(KAGGLE + ["kernels", "status", kid])
    text = (r.stdout + r.stderr).lower()
    # CLI prints e.g.  '<id> has status "complete"'
    for s in ("complete", "error", "cancelacknowledged", "cancelrequested",
              "running", "queued"):
        if s in text:
            return s
    return "unknown:" + text.strip()[:80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--poll-secs", type=int, default=60)
    p.add_argument("--max-hours", type=float, default=9.0)
    p.add_argument("--no-push", action="store_true", help="skip push, just poll/fetch")
    args = p.parse_args()

    kid = kernel_id(args.dir)
    print(f"[{kid}] preparing run")
    if not args.no_push:
        push(args.dir)

    deadline = time.time() + args.max_hours * 3600
    last = None
    while time.time() < deadline:
        st = status(kid)
        if st != last:
            print(f"[{kid}] status: {st}", flush=True)
            last = st
        if st in ("complete", "error", "cancelacknowledged"):
            break
        time.sleep(args.poll_secs)
    else:
        raise SystemExit(f"[{kid}] timed out after {args.max_hours}h")

    out_dir = os.path.join(args.dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    r = run(KAGGLE + ["kernels", "output", kid, "-p", out_dir])
    print(r.stdout.strip()); print(r.stderr.strip(), file=sys.stderr)

    if st != "complete":
        # Save whatever log we got to help diagnose.
        raise SystemExit(f"[{kid}] finished with status={st} (see {out_dir})")

    mpath = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"[{kid}] complete but no metrics.json in {out_dir}")
    with open(mpath) as f:
        metrics = json.load(f)
    with open(os.path.join(args.dir, "result.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{kid}] RESULT")
    print(f"  arch={metrics.get('arch')} dataset={metrics.get('dataset')}")
    print(f"  test_acc={metrics.get('test_accuracy')}")
    print(f"  mean_class_acc={metrics.get('test_mean_class_acc')}")
    print(f"  test_f1={metrics.get('test_f1')}  best_val_acc={metrics.get('best_val_acc')}")
    print(f"  per_class={metrics.get('per_class_acc')}")


if __name__ == "__main__":
    main()
