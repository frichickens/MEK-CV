"""Scan kaggle/tuning/runs/*/result.json and print one comparison table.

Source of truth for the campaign. Selection rule: BEST PER (method,arch) is chosen by
**validation** accuracy (best_val_acc) — never by test. We then report that row's test
metrics. Run dir name is the config label.
"""
import glob
import json
import os

RUNS = os.path.join(os.path.dirname(__file__), "runs")


def method_of(arch, hp, name):
    is_mek = "mek" in name.lower()
    return f"{'MEK ' if is_mek else ''}{arch}"


def load():
    rows = []
    for rj in glob.glob(os.path.join(RUNS, "*", "result.json")):
        name = os.path.basename(os.path.dirname(rj))
        if "smoke" in name:
            continue
        with open(rj) as f:
            r = json.load(f)
        hp = r.get("hyperparams", {})
        rows.append({
            "config": name,
            "method": method_of(r.get("arch"), hp, name),
            "arch": r.get("arch"),
            "val": r.get("best_val_acc"),
            "test_acc": r.get("test_accuracy"),
            "mean_cls": r.get("test_mean_class_acc"),
            "f1": r.get("test_f1"),
            "LR": hp.get("LR"),
            "epochs": r.get("epochs_run"),
        })
    return rows


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "  -  "


def main():
    rows = load()
    rows.sort(key=lambda r: (r["method"], -(r["val"] or 0)))
    print(f"{'config':<34}{'method':<12}{'val':>8}{'test':>8}{'mean_cls':>9}{'f1':>8}{'LR':>9}{'ep':>5}")
    print("-" * 93)
    for r in rows:
        print(f"{r['config']:<34}{r['method']:<12}{fmt(r['val']):>8}{fmt(r['test_acc']):>8}"
              f"{fmt(r['mean_cls']):>9}{fmt(r['f1']):>8}{str(r['LR']):>9}{str(r['epochs']):>5}")

    # Best per method by VALIDATION accuracy.
    best = {}
    for r in rows:
        m = r["method"]
        if m not in best or (r["val"] or 0) > (best[m]["val"] or 0):
            best[m] = r
    print("\n=== BEST PER METHOD (selected on validation) ===")
    print(f"{'method':<12}{'config':<34}{'val':>8}{'test':>8}{'mean_cls':>9}{'f1':>8}")
    for m in sorted(best):
        r = best[m]
        print(f"{m:<12}{r['config']:<34}{fmt(r['val']):>8}{fmt(r['test_acc']):>8}"
              f"{fmt(r['mean_cls']):>9}{fmt(r['f1']):>8}")


if __name__ == "__main__":
    main()
