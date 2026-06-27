"""Append a COMBINED best-results table (RAF-DB + FER-2013) to results-CV-project.md.
Best config per (dataset, method) selected on validation accuracy (tie-break: val mean-class).
All numbers read from result.json — nothing hand-typed. FER-2013 rows come from the separate
FER job (account baobaoo) and may be a snapshot if its search is still running.
"""
import glob, json, os, io

ROOT = os.path.dirname(__file__); RUNS = os.path.join(ROOT, "runs")
MD = os.path.join(ROOT, "..", "..", "results-CV-project.md")

def dataset(d): return 'fer2013' if 'fer' in d else 'rafdb'
def method(d):
    return ('RN34' if ('rn34' in d or 'resnet34' in d) else 'RN18') + (' + MEK' if 'mek' in d else '')

rows = []
for rj in glob.glob(os.path.join(RUNS, "*", "result.json")):
    d = os.path.basename(os.path.dirname(rj))
    if 'smoke' in d: continue
    r = json.load(open(rj)); r['_dir'] = d; r['_ds'] = dataset(d); r['_m'] = method(d)
    rows.append(r)

ORDER = ['RN18', 'RN34', 'RN18 + MEK', 'RN34 + MEK']
def best(ds, m):
    cs = [r for r in rows if r['_ds'] == ds and r['_m'] == m]
    if not cs: return None, 0
    n_total = len(cs)  # total search effort (drives the "searched?"/blank decision + #cfg)
    # Select from the canonical val-mean-logged reruns when available (so this table matches
    # the main RAF-DB report); else from all configs (FER job lacks that logging).
    pref = [r for r in cs if r.get('val_mean_class_acc') is not None] or cs
    pref.sort(key=lambda r: (round(r['best_val_acc'], 3), r.get('val_mean_class_acc') or 0), reverse=True)
    return pref[0], n_total

def section(ds, title):
    sel = {m: best(ds, m) for m in ORDER}
    cols = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_mean_class_acc']
    # Only methods with a real search (>=2 configs) are shown, so bold against those only.
    shown = [m for m in ORDER if sel[m][0] is not None and sel[m][1] >= 2]
    mx = {c: max((sel[m][0][c] for m in shown), default=0) for c in cols}
    L = [f"### {title}", "",
         "| Model | Accuracy | F1 | Precision | Recall | Mean-class acc | # cfg |",
         "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for m in ORDER:
        r, n = sel[m]
        # Leave empty if no result, or no real search yet (<2 configs) — don't report
        # uncertain/incomplete numbers.
        if r is None or n < 2:
            note = " *(in progress)*" if r is not None else ""
            L.append(f"| {m}{note} |  |  |  |  |  | {n} |"); continue
        def c(k):
            v = r[k]; return f"**{v:.4f}**" if abs(v - mx[k]) < 1e-9 else f"{v:.4f}"
        L.append(f"| {m} | {c('test_accuracy')} | {c('test_f1')} | {c('test_precision')} | "
                 f"{c('test_recall')} | {c('test_mean_class_acc')} | {n} |")
    return "\n".join(L)

out = []
out.append("# ════════════════════════════════════════════════════════════════════")
out.append("# COMBINED BEST RESULTS — tuned, both datasets (2026-06-04)")
out.append("# ════════════════════════════════════════════════════════════════════")
out.append("")
out.append("Best configuration per (dataset, method), **selected on validation accuracy** (no test")
out.append("leakage), ImageNet backbone. Bold = best in column within each dataset. RAF-DB from this")
out.append("job (`baodqhust`); FER-2013 from the parallel job (`baobaoo`). See the per-dataset sections")
out.append("above for full search logs, per-class accuracy, and noise caveats (±0.5 pt single-seed).")
out.append("")
out.append(section('rafdb', 'RAF-DB (peak config per method)'))
out.append("")
out.append(section('fer2013', 'FER-2013 (peak config per method — search may be in progress)'))
out.append("")

# Idempotent: drop any existing COMBINED section before appending.
lines = io.open(MD, encoding="utf-8").read().splitlines()
cut = next((i for i, l in enumerate(lines) if 'COMBINED BEST RESULTS' in l), None)
if cut is not None:
    while cut > 0 and not lines[cut].startswith('# ═'):
        cut -= 1
    lines = lines[:cut]
txt = "\n".join(lines).rstrip()
io.open(MD, "w", encoding="utf-8").write(txt + "\n\n" + "\n".join(out) + "\n")
print("Appended COMBINED BEST RESULTS section (idempotent).")
print("\n".join(out))
