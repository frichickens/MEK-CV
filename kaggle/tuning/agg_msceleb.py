"""Aggregate the RAF-DB MS-Celeb-1M MEK-RN18 campaign results (reads result.json only)."""
import json, os, glob

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "msceleb")
LABELS = {
    "rafdb-mek-rn18-msceleb-paper": "paper (LR1e-4 g0.9 l2 e0.1 60ep)",
    "rafdb-mek-rn18-msceleb-lr2e4": "LR2e-4",
    "rafdb-mek-rn18-msceleb-lr3e4": "LR3e-4",
    "rafdb-mek-rn18-msceleb-lam1": "lambda=1",
    "rafdb-mek-rn18-msceleb-lam3": "lambda=3",
    "rafdb-mek-rn18-msceleb-eps02": "eps=0.2",
    "rafdb-mek-rn18-msceleb-g095": "gamma=0.95",
    "rafdb-mek-rn18-msceleb-ep80": "80ep (g0.9)",
    "rafdb-mek-rn18-msceleb-ep80-g095": "80ep + gamma=0.95",
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

rows.sort(key=lambda r: -(r["mca"] or 0))
print(f"{'label':36s} {'val_acc':>8s} {'val_mca':>8s} {'TEST acc':>9s} {'mca':>7s} {'f1':>7s} {'ep':>4s}")
for r in rows:
    print(f"{r['label']:36s} {r['val_acc']:8.4f} {r['val_mca']:8.4f} "
          f"{r['acc']:9.4f} {r['mca']:7.4f} {r['f1']:7.4f} {r['epochs_run']:4d}")

# Validation-selected best (mean-class), report its TEST — no leakage.
best_valmca = max(rows, key=lambda r: r["val_mca"] or 0)
best_valacc = max(rows, key=lambda r: r["val_acc"] or 0)
print("\nVAL-selected best by val_mean_class:", best_valmca["label"],
      "-> test acc %.4f / mca %.4f / f1 %.4f" % (best_valmca["acc"], best_valmca["mca"], best_valmca["f1"]))
print("VAL-selected best by val_acc:", best_valacc["label"],
      "-> test acc %.4f / mca %.4f / f1 %.4f" % (best_valacc["acc"], best_valacc["mca"], best_valacc["f1"]))
print("\nPer-class (val-mca-selected best):", json.dumps(best_valmca["pca"], indent=0))
print("HP:", best_valmca["hp"])
print("\nmean across search: acc %.4f  mca %.4f  f1 %.4f  (n=%d)" % (
    sum(r["acc"] for r in rows)/len(rows), sum(r["mca"] for r in rows)/len(rows),
    sum(r["f1"] for r in rows)/len(rows), len(rows)))
