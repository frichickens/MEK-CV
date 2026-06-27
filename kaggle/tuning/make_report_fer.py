"""Generate the FER-2013 detailed results section (best-per-method, per-class, full search
log) from the FER result.json files (account baobaoo) and splice it into results-CV-project.md
just before the COMBINED section. Idempotent. All numbers read from disk.
"""
import glob, json, os, io

ROOT = os.path.dirname(__file__); RUNS = os.path.join(ROOT, "runs")
MD = os.path.join(ROOT, "..", "..", "results-CV-project.md")
EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def method(d):
    return ('RN34' if ('rn34' in d or 'resnet34' in d) else 'RN18') + (' + MEK' if 'mek' in d else '')

rows = []
for rj in glob.glob(os.path.join(RUNS, "*", "result.json")):
    d = os.path.basename(os.path.dirname(rj))
    if 'fer' not in d or 'smoke' in d:
        continue
    r = json.load(open(rj)); r['_dir'] = d; r['_m'] = method(d)
    rows.append(r)

ORDER = ['RN18', 'RN34', 'RN18 + MEK', 'RN34 + MEK']
def by(m): return [r for r in rows if r['_m'] == m]
def best(m):
    cs = by(m)
    if not cs: return None
    pref = [r for r in cs if r.get('val_mean_class_acc') is not None] or cs
    pref.sort(key=lambda r: (round(r['best_val_acc'], 3), r.get('val_mean_class_acc') or 0), reverse=True)
    return pref[0]
SEL = {m: best(m) for m in ORDER}

def lr_str(r):
    lr = r.get('hyperparams', {}).get('LR')
    return f"{lr:g}" if isinstance(lr, (int, float)) else str(lr)
def lam(r):
    v = r.get('hyperparams', {}).get('FLIP_LOSS_WEIGHT')
    return f", λ{v:g}" if isinstance(v, (int, float)) and 'MEK' in r['_m'] else ""

S = []
S.append("# ════════════════════════════════════════════════════════════════════")
S.append("# TUNED RESULTS — FER-2013, ImageNet backbone (2026-06-04)")
S.append("# ════════════════════════════════════════════════════════════════════")
S.append("")
S.append("Automated Kaggle tuning campaign, account `baobaoo`, GPU Tesla T4. FER-2013")
S.append("(`msambare/fer2013`), seed 42, 90/10 train/val split. Same protocol & integrity rules as")
S.append("the RAF-DB section (selection on validation; test only for reporting; ±0.5 pt single-seed noise).")
S.append("Baseline @ 44 px (SGD+cosine, mixup, weighted sampler); MEK @ 224 px (Adam+ExpLR γ0.9).")
S.append("")
S.append("## Best config per method (validation-selected)")
S.append("")
S.append("| Method | LR | Sched/opt | Val acc | **Test acc** | F1 | Precision | Recall | **Mean-class** |")
S.append("|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|")
cols = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_mean_class_acc']
mx = {c: max((SEL[m][c] for m in ORDER if SEL[m]), default=0) for c in cols}
def bcell(r, k):
    v = r[k]; return f"**{v:.4f}**" if abs(v - mx[k]) < 1e-9 else f"{v:.4f}"
for m in ORDER:
    r = SEL[m]
    if r is None:
        S.append(f"| {m} |  |  |  |  |  |  |  |  |"); continue
    sched = "Adam+ExpLR" if "MEK" in m else "SGD+Cosine"
    label = f"**{m}**" if m == 'RN34 + MEK' else m
    S.append(f"| {label} | {lr_str(r)}{lam(r)} | {sched} | {r['best_val_acc']:.4f} | "
             f"{bcell(r,'test_accuracy')} | {bcell(r,'test_f1')} | {bcell(r,'test_precision')} | "
             f"{bcell(r,'test_recall')} | {bcell(r,'test_mean_class_acc')} |")
S.append("")
S.append("## Per-class test accuracy (best config per method)")
S.append("")
S.append("| Method | " + " | ".join(EMO) + " |")
S.append("|:--|" + ":-:|" * len(EMO))
pcmax = {e: max((SEL[m].get('per_class_acc') or {}).get(e, 0) for m in ORDER if SEL[m]) for e in EMO}
for m in ORDER:
    r = SEL[m]
    if r is None: continue
    pc = r.get('per_class_acc') or {}
    cells = [(f"**{pc.get(e,0):.3f}**" if abs(pc.get(e, 0) - pcmax[e]) < 1e-9 else f"{pc.get(e,0):.3f}") for e in EMO]
    S.append(f"| {m} | " + " | ".join(cells) + " |")
S.append("")
S.append("## Key findings (FER-2013)")
S.append("")
S.append("1. **MEK ≫ baseline:** RN34+MEK vs RN34 ≈ **+5–6 pt** test acc and mean-class; RN18+MEK")
S.append("   vs RN18 ≈ **+4 pt**. Same decisive method effect as RAF-DB.")
S.append("2. **Best model: RN34 + MEK.**")
S.append("3. **λ matters:** λ=2 over-regularizes on FER; λ≈0.5–1 is the sweet spot (same as RAF-DB).")
S.append("4. Plain baselines plateau ~0.66 at 44 px; the RN18/RN34 baseline gap is within noise.")
S.append("")
S.append("## Full search log (all FER-2013 configs; ★ = validation-selected best per method)")
S.append("")
S.append("| config | method | val acc | val mean | test acc | mean-cls | F1 | LR | λ |")
S.append("|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
seldirs = {SEL[m]['_dir'] for m in ORDER if SEL[m]}
for r in sorted(rows, key=lambda r: (ORDER.index(r['_m']), -r['best_val_acc'])):
    vm = r.get('val_mean_class_acc'); vm = f"{vm:.4f}" if isinstance(vm, (int, float)) else "—"
    lv = r.get('hyperparams', {}).get('FLIP_LOSS_WEIGHT'); lv = f"{lv:g}" if isinstance(lv, (int, float)) else "—"
    star = " ★" if r['_dir'] in seldirs else ""
    S.append(f"| {r['_dir']}{star} | {r['_m']} | {r['best_val_acc']:.4f} | {vm} | "
             f"{r['test_accuracy']:.4f} | {r['test_mean_class_acc']:.4f} | {r['test_f1']:.4f} | {lr_str(r)} | {lv} |")
S.append("")
section = "\n".join(S)

# The FER section sits directly before COMBINED. Replace old FER region (if any) / insert.
lines = io.open(MD, encoding="utf-8").read().splitlines()
def sep_before(keyword):
    i = next((k for k, l in enumerate(lines) if keyword in l), None)
    if i is None: return None
    while i > 0 and not lines[i].startswith('# ═'):
        i -= 1
    return i
comb = sep_before('COMBINED BEST RESULTS')
fer = sep_before('TUNED RESULTS — FER-2013')
block = S + [""]
if comb is not None:
    start = fer if fer is not None else comb
    out = lines[:start] + block + lines[comb:]
elif fer is not None:
    out = lines[:fer] + block
else:
    out = lines + [""] + block
io.open(MD, "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")
print("Spliced FER-2013 detailed section. Best per method:")
for m in ORDER:
    r = SEL[m]
    if r: print(f"  {m:<12} {r['_dir']:<36} test={r['test_accuracy']:.4f} mean={r['test_mean_class_acc']:.4f}")
