"""Aggregate the RAF-DB webcam-deploy (CLAHE+facecrop) campaign (reads result.json only)."""
import json, os, glob

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "webcam")
LABELS = {
    "rafdb-webcam-cf-default":    "webcam recipe (e0.15 l0.1 LR1e-4 g0.9 60ep)",
    "rafdb-webcam-cf-lr2e4":      "LR2e-4",
    "rafdb-webcam-cf-lr3e4":      "LR3e-4",
    "rafdb-webcam-cf-g095":       "gamma=0.95",
    "rafdb-webcam-cf-ep80-g095":  "80ep + gamma=0.95",
    "rafdb-webcam-cf-eps01":      "eps=0.1",
    "rafdb-webcam-cf-lam05":      "lambda=0.5",
    "rafdb-webcam-cf-lam1":       "lambda=1.0",
    "rafdb-webcam-cf-lr2e4-g095": "LR2e-4 + gamma=0.95",
}

rows = []
for rj in glob.glob(os.path.join(RUNS, "*", "*", "result.json")):
    slug = os.path.basename(os.path.dirname(rj))
    if slug not in LABELS:
        continue
    d = json.load(open(rj))
    rows.append({
        "slug": slug, "label": LABELS[slug],
        "val_acc": d.get("val_accuracy"), "val_mca": d.get("val_mean_class_acc"),
        "acc": d.get("test_accuracy"), "mca": d.get("test_mean_class_acc"),
        "f1": d.get("test_f1"), "prec": d.get("test_precision"), "rec": d.get("test_recall"),
        "epochs_run": d.get("epochs_run"), "pca": d.get("per_class_acc"),
        "hp": d.get("hyperparams", {}),
    })

rows.sort(key=lambda r: -(r["val_mca"] or 0))
print(f"{'label':46s} {'val_acc':>8s} {'val_mca':>8s} {'TEST acc':>9s} {'mca':>7s} {'f1':>7s} {'ep':>4s}")
for r in rows:
    print(f"{r['label']:46s} {r['val_acc']:8.4f} {r['val_mca']:8.4f} "
          f"{r['acc']:9.4f} {r['mca']:7.4f} {r['f1']:7.4f} {r['epochs_run']:4d}")

best = max(rows, key=lambda r: r["val_mca"] or 0)
print("\nVAL-selected best (by val_mean_class):", best["label"], "[" + best["slug"] + "]")
print("  -> test acc %.4f / mca %.4f / f1 %.4f / prec %.4f / rec %.4f" % (
    best["acc"], best["mca"], best["f1"], best["prec"], best["rec"]))
print("  per-class:", json.dumps(best["pca"]))
print("  hp:", best["hp"])
print("\nmean across search: acc %.4f  mca %.4f  f1 %.4f  (n=%d)" % (
    sum(r["acc"] for r in rows)/len(rows), sum(r["mca"] for r in rows)/len(rows),
    sum(r["f1"] for r in rows)/len(rows), len(rows)))
