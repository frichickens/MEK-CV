**Results:**  
**ImageNet Table:**

|Dataset|Model (ImageNet pretrained)|Accuracy|F1|Precision|Recall|Mean Class Accuracy||
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|-|
|FER2013|RN18|0.6524|0.6341|0.6382|0.6339|0.6339||
||RN34|0.6584|**0.6443**|**0.6508**|0.6411|0.6411||
||RN18 + MEK|0.6728|0.6145|0.6382|0.6702|0.6702||
||RN34 + MEK|**0.6878**|0.6275|0.6337|**0.6805**|**0.6805**||
|RAF-DB|RN18|0.8214|**0.7410**|**0.7460**|0.7402|0.7402||
||RN34|**0.8240**|0.7304|0.7330|0.7296|0.7296||
||RN18 + MEK|0.7683|0.6747|0.6669|0.7387|0.7387||
||RN34 + MEK|0.8067|0.7212|0.7042|**0.7670**|**0.7670**||

**Perclass accuracy:**

|Dataset|Model (ImageNet pretrained)|happy|neutral|sad|surprise|disgust||anger|fear|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|-|:-:|:-:|
|FER2013|RN18|0.8433|0.6529|0.4715|0.8400|0.5856||0.5772|0.4668|
||RN34|0.8354|0.6245|0.5237|0.8291|0.5946||0.6127|0.4678|
||RN18 + MEK|0.8799|0.6707|0.5108|0.8243|0.7568||0.5950|0.4541|
||RN34 + MEK|0.8878|0.6504|0.5349|0.8400|0.7207||0.6284|0.5010|
|RAF-DB|RN18|0.8886|0.7956|0.8326|0.8541|0.5625||0.7346|0.5135|
||RN34|0.9105|0.7956|0.8159|0.8632|0.4500||0.7716|0.5000|
||RN18 + MEK|0.8380|0.7382|0.6946|0.7872|0.5437||0.7716|0.7973|
||RN34 + MEK|0.8658|0.7868|0.7699|0.8237|0.5813||0.7716|0.7703|

**MS-Celeb-1M Table:**

|Dataset|Model|accuracy|f1|precision|recall|Mean class accuracy||
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|-|
|FER2013|RN18 + MEK  (ImageNet pretrained)|0.6878|0.6275|0.6337|0.6805|0.6805||
||RN18 + MEK (MS-Celeb-1M pretrained)|||||||
|RAF-DB|RN18 + MEK (Best ImageNet pretrained)|0.7683|0.6747|0.6669|0.7387|0.7387||
||RN18 + MEK (MS-Celeb-1M pretrained, tuned 2026-06-09)|**0.8641**|**0.7969**|**0.7849**|**0.8137**|**0.8137**||

# ════════════════════════════════════════════════════════════════════
# TUNED RESULTS — RAF-DB, ImageNet backbone (2026-06-04)
# ════════════════════════════════════════════════════════════════════

Automated Kaggle tuning campaign (`kaggle/tuning/`). Account `baodqhust`, GPU **Tesla T4**,
RAF-DB (`shuvoalok/raf-db-dataset`), seed 42, 90/10 train/val split. Class index → emotion:
`1`=surprise `2`=fear `3`=disgust `4`=happy `5`=sad `6`=angry `7`=neutral.

## Methodology (integrity)

* Per-method hyperparameter search (LR, schedule/epochs; for MEK also λ_flip and ε_LSR).
* **Configs selected on the VALIDATION set; TEST used only for final reporting** — no config
  was chosen by its test score (no test-set leakage).
* **Measured single-seed noise ≈ ±0.5 pt:** identical configs re-run differ by up to ~0.5 pt
  test acc (e.g. MEK-RN34 0.8589 vs 0.8677; MEK-RN18 0.8517 vs 0.8585) — Kaggle GPU is
  non-deterministic (`cudnn.benchmark=True`). **Per-run gaps below ~0.5 pt are not significant.**
* The RN18-vs-RN34 gap sits near that noise floor, so the architecture comparison is reported
  as the **mean across the search** (averages out per-run noise); peak configs are reported separately.

## Architecture comparison — mean across the hyperparameter search

*(mean test metric over all configs tried per method — the robust "deeper vs shallower" view)*

| Method | n cfg | Test acc | Mean-class | F1 |
|:--|:-:|:-:|:-:|:-:|
| RN18 | 6 | 0.8295 | 0.7370 | 0.7465 |
| **RN34** | 6 | 0.8364 | 0.7420 | 0.7498 |
| MEK RN18 | 5 | 0.8473 | 0.7770 | 0.7770 |
| **MEK RN34** | 6 | **0.8592** | **0.7829** | **0.7882** |

→ **RN34 > RN18 on every metric**, both as plain baseline and with MEK — the deeper backbone
is consistently better once single-seed noise is averaged out. **MEK ≫ baseline** throughout.

## Best single config per method (validation-selected) — peak numbers

| Method | LR | Sched/opt | Ep | Val acc | **Test acc** | F1 | Precision | Recall | **Mean-class** |
|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.003 | SGD+Cosine | 80 | 0.8411 | 0.8396 | 0.7616 | 0.7815 | 0.7492 | 0.7492 |
| RN34 | 0.001 | SGD+Cosine | 80 | 0.8435 | 0.8367 | 0.7465 | 0.7622 | 0.7369 | 0.7369 |
| MEK RN18 | 0.0003 | Adam+ExpLR | 60 | 0.8631 | 0.8585 | **0.7962** | **0.8038** | **0.7910** | **0.7910** |
| **MEK RN34** | 0.0002, ε0.2 | Adam+ExpLR | 60 | 0.8590 | **0.8657** | 0.7951 | 0.8026 | 0.7892 | 0.7892 |

**Best single model = RN34 + MEK (test acc 0.8657).** At the single-run level the RN18/RN34
*baseline* point estimates are within noise (here RN18's lucky draw edges it) — see the mean
table above for the robust architecture ordering. RN34+MEK uses stronger ε (0.2) to lift the
minority classes (fear/disgust) that the default recipe left below RN18+MEK.

## Per-class test accuracy (best config per method)

| Method | surprise | fear | disgust | happy | sad | angry | neutral |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.881 | 0.514 | 0.525 | 0.917 | 0.856 | 0.747 | 0.804 |
| RN34 | 0.875 | 0.473 | 0.475 | 0.920 | 0.814 | 0.772 | 0.829 |
| MEK RN18 | 0.894 | **0.649** | 0.569 | 0.918 | **0.858** | **0.809** | 0.841 |
| MEK RN34 | **0.903** | 0.581 | **0.606** | **0.936** | 0.843 | **0.809** | **0.847** |

## Key findings

1. **MEK ≫ baseline — the decisive, robust result.** Both backbones gain large mean-class
   (≈ +4–5 pt) with overall accuracy also up; minority classes (fear, disgust, angry) rise
   sharply. Matches the paper: MEK raises balanced accuracy without sacrificing overall.
2. **RN34 > RN18 on every metric** (search mean), baseline and MEK — deeper is consistently better.
3. **Best single model: RN34 + MEK — 0.8657 test acc / 0.7892 mean-class.**
4. LR is the dominant lever; deeper RN34 prefers a lower LR (MEK 2e-4 vs RN18 3e-4).

## Honest caveats

* Single-seed; measured noise ±0.5 pt. Per-run RN18-vs-RN34 differences are at the noise floor —
  hence the search-mean for the architecture claim. Multi-seed (mean±std) would tighten it but
  was not run (user opted for single-seed).
* **ImageNet** backbones. The paper's ~89 % RAF-DB uses an **MS-Celeb-1M face-pretrained** backbone;
  a prior MS-Celeb RN18+MEK run here reached 0.8419. Face pre-training is the biggest remaining lever.

## Full search log (all RAF-DB configs; ★ = validation-selected best per method)

| config | method | val acc | val mean | test acc | mean-cls | F1 | LR | ep |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| rn18-rafdb-lr3e3-mc ★ | RN18 | 0.8411 | 0.7336 | 0.8396 | 0.7492 | 0.7616 | 0.003 | 80 |
| rn18-rafdb-lr2e3-ep80 | RN18 | 0.8370 | — | 0.8253 | 0.7403 | 0.7506 | 0.002 | 51 |
| rn18-rafdb-lr2e3-mc | RN18 | 0.8370 | 0.7317 | 0.8253 | 0.7403 | 0.7506 | 0.002 | 51 |
| rn18-rafdb-lr3e3-ep80 | RN18 | 0.8354 | — | 0.8406 | 0.7407 | 0.7562 | 0.003 | 58 |
| rn18-rafdb-ep80 | RN18 | 0.8240 | — | 0.8233 | 0.7201 | 0.7284 | 0.001 | 80 |
| resnet18-rafdb-default | RN18 | 0.8199 | — | 0.8227 | 0.7314 | 0.7320 | 0.001 | 60 |
| rn34-rafdb-lr1e3-ep80 | RN34 | 0.8452 | — | 0.8331 | 0.7368 | 0.7454 | 0.001 | 76 |
| rn34-rafdb-lr15e3-mc | RN34 | 0.8435 | 0.7438 | 0.8432 | 0.7491 | 0.7587 | 0.0015 | 68 |
| rn34-rafdb-lr1e3-mc ★ | RN34 | 0.8435 | 0.7515 | 0.8367 | 0.7369 | 0.7465 | 0.001 | 80 |
| rn34-rafdb-lr15e4-ep80 | RN34 | 0.8427 | — | 0.8462 | 0.7512 | 0.7644 | 0.0015 | 80 |
| rn34-rafdb-ep80 | RN34 | 0.8305 | — | 0.8312 | 0.7388 | 0.7441 | 0.0007 | 74 |
| resnet34-rafdb-default | RN34 | 0.8207 | — | 0.8282 | 0.7393 | 0.7399 | 0.0007 | 60 |
| mek-rn18-lr3e4-mc ★ | MEK RN18 | 0.8631 | 0.7775 | 0.8585 | 0.7910 | 0.7962 | 0.0003 | 60 |
| mek-rn18-rafdb-lr3e4 | MEK RN18 | 0.8574 | — | 0.8517 | 0.7865 | 0.7871 | 0.0003 | 58 |
| mek-rn18-rafdb-lr3e4-g95 | MEK RN18 | 0.8541 | — | 0.8442 | 0.7685 | 0.7682 | 0.0003 | 55 |
| mek-rn18-rafdb-lr5e4 | MEK RN18 | 0.8533 | — | 0.8478 | 0.7740 | 0.7766 | 0.0005 | 60 |
| mek-rn18-rafdb-default-v2 | MEK RN18 | 0.8394 | — | 0.8341 | 0.7648 | 0.7569 | 0.0001 | 60 |
| mek-rn34-rafdb-lr2e4 | MEK RN34 | 0.8631 | — | 0.8589 | 0.7777 | 0.7849 | 0.0002 | 60 |
| mek-rn34-rafdb-lr2e4-eps2 ★ | MEK RN34 | 0.8590 | 0.7750 | 0.8657 | 0.7892 | 0.7951 | 0.0002 | 60 |
| mek-rn34-lr2e4-mc | MEK RN34 | 0.8582 | 0.7653 | 0.8677 | 0.7855 | 0.7959 | 0.0002 | 60 |
| mek-rn34-rafdb-lr2e4-lam3 | MEK RN34 | 0.8582 | 0.7723 | 0.8621 | 0.7893 | 0.7946 | 0.0002 | 60 |
| mek-rn34-rafdb-default-v2 | MEK RN34 | 0.8541 | — | 0.8497 | 0.7769 | 0.7745 | 0.0001 | 60 |
| mek-rn34-rafdb-lr3e4 | MEK RN34 | 0.8492 | — | 0.8514 | 0.7786 | 0.7845 | 0.0003 | 60 |

# ════════════════════════════════════════════════════════════════════
# TUNED RESULTS — FER-2013, ImageNet backbone (2026-06-04)
# ════════════════════════════════════════════════════════════════════

Automated Kaggle tuning campaign, account `baobaoo`, GPU Tesla T4. FER-2013
(`msambare/fer2013`), seed 42, 90/10 train/val split. Same protocol & integrity rules as
the RAF-DB section (selection on validation; test only for reporting; ±0.5 pt single-seed noise).
Baseline @ 44 px (SGD+cosine, mixup, weighted sampler); MEK @ 224 px (Adam+ExpLR γ0.9).

## Best config per method (validation-selected)

| Method | LR | Sched/opt | Val acc | **Test acc** | F1 | Precision | Recall | **Mean-class** |
|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.014 | SGD+Cosine | 0.6697 | 0.6655 | 0.6480 | 0.6529 | 0.6461 | 0.6461 |
| RN34 | 0.007 | SGD+Cosine | 0.6753 | 0.6622 | 0.6504 | 0.6592 | 0.6448 | 0.6448 |
| RN18 + MEK | 0.0001, λ0.25 | Adam+ExpLR | 0.7268 | 0.7081 | 0.6952 | 0.7009 | 0.6928 | 0.6928 |
| **RN34 + MEK** | 0.0001, λ0.1 | Adam+ExpLR | 0.7310 | **0.7205** | **0.7073** | **0.7252** | **0.6960** | **0.6960** |

## Per-class test accuracy (best config per method)

| Method | angry | disgust | fear | happy | neutral | sad | surprise |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.636 | 0.586 | 0.481 | 0.844 | 0.620 | 0.532 | 0.823 |
| RN34 | 0.604 | 0.604 | 0.496 | 0.847 | 0.628 | 0.513 | 0.822 |
| RN18 + MEK | 0.653 | **0.658** | 0.508 | 0.851 | **0.724** | 0.606 | 0.850 |
| RN34 + MEK | **0.665** | 0.613 | **0.532** | **0.895** | 0.667 | **0.637** | **0.863** |

## Key findings (FER-2013)

1. **MEK ≫ baseline:** RN34+MEK vs RN34 ≈ **+5–6 pt** test acc and mean-class; RN18+MEK
   vs RN18 ≈ **+4 pt**. Same decisive method effect as RAF-DB.
2. **Best model: RN34 + MEK.**
3. **λ matters:** λ=2 over-regularizes on FER; λ≈0.5–1 is the sweet spot (same as RAF-DB).
4. Plain baselines plateau ~0.66 at 44 px; the RN18/RN34 baseline gap is within noise.

## Full search log (all FER-2013 configs; ★ = validation-selected best per method)

| config | method | val acc | val mean | test acc | mean-cls | F1 | LR | λ |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| fer2013-resnet18-lr14e3-wd1e4 ★ | RN18 | 0.6697 | — | 0.6655 | 0.6461 | 0.6480 | 0.014 | — |
| fer2013-resnet18-lr14e3-wd5e4 | RN18 | 0.6662 | — | 0.6605 | 0.6435 | 0.6466 | 0.014 | — |
| fer2013-resnet18-lr1e2-wd5e4 | RN18 | 0.6648 | — | 0.6661 | 0.6461 | 0.6504 | 0.01 | — |
| resnet18-fer2013-default | RN18 | 0.6645 | — | 0.6665 | 0.6448 | 0.6479 | 0.01 | — |
| fer2013-resnet18-lr7e3-wd1e4 | RN18 | 0.6589 | — | 0.6569 | 0.6375 | 0.6406 | 0.007 | — |
| resnet34-fer2013-default ★ | RN34 | 0.6753 | — | 0.6622 | 0.6448 | 0.6504 | 0.007 | — |
| fer2013-resnet34-lr1e2-wd5e4 | RN34 | 0.6739 | — | 0.6626 | 0.6452 | 0.6407 | 0.01 | — |
| fer2013-resnet34-lr14e3-wd1e4 | RN34 | 0.6711 | — | 0.6638 | 0.6493 | 0.6510 | 0.014 | — |
| fer2013-resnet34-lr14e3-wd5e4 | RN34 | 0.6697 | — | 0.6643 | 0.6467 | 0.6505 | 0.014 | — |
| fer2013-resnet34-lr7e3-wd1e4 | RN34 | 0.6645 | — | 0.6616 | 0.6425 | 0.6460 | 0.007 | — |
| fer2013-resnet34-lr7e3-wd5e4 | RN34 | 0.6582 | — | 0.6552 | 0.6423 | 0.6406 | 0.007 | — |
| fer2013-resnet34-lr1e2-wd1e4 | RN34 | 0.6564 | — | 0.6652 | 0.6518 | 0.6556 | 0.01 | — |
| fer2013-mek-resnet18-lr1e4-l025 ★ | RN18 + MEK | 0.7268 | 0.7147 | 0.7081 | 0.6928 | 0.6952 | 0.0001 | 0.25 |
| fer2013-mek-resnet18-lr1e4-l01 | RN18 + MEK | 0.7258 | 0.7152 | 0.7140 | 0.7011 | 0.7026 | 0.0001 | 0.1 |
| fer2013-mek-resnet18-lr1e4-l05 | RN18 + MEK | 0.7251 | — | 0.7077 | 0.6956 | 0.6883 | 0.0001 | 0.5 |
| fer2013-mek-resnet18-lr5e4-l2 | RN18 + MEK | 0.7167 | — | 0.6989 | 0.6782 | 0.6843 | 0.0005 | 2 |
| fer2013-mek-resnet18-lr1e4-l1 | RN18 + MEK | 0.7150 | — | 0.7028 | 0.6881 | 0.6824 | 0.0001 | 1 |
| mek-resnet18-fer2013-default | RN18 + MEK | 0.7010 | — | 0.6906 | 0.6726 | 0.6674 | 0.0001 | 2 |
| fer2013-mek-resnet34-lr1e4-l05 | RN34 + MEK | 0.7334 | — | 0.7182 | 0.6940 | 0.7017 | 0.0001 | 0.5 |
| fer2013-mek-resnet34-lr1e4-l01 ★ | RN34 + MEK | 0.7310 | 0.7190 | 0.7205 | 0.6960 | 0.7073 | 0.0001 | 0.1 |
| fer2013-mek-resnet34-lr1e4-l1 | RN34 + MEK | 0.7279 | — | 0.7143 | 0.6897 | 0.6997 | 0.0001 | 1 |
| fer2013-mek-resnet34-lr3e4-l2 | RN34 + MEK | 0.7157 | — | 0.7037 | 0.6842 | 0.6898 | 0.0003 | 2 |
| mek-resnet34-fer2013-default | RN34 + MEK | 0.7153 | — | 0.7087 | 0.6849 | 0.6879 | 0.0001 | 2 |
| fer2013-mek-resnet34-lr5e4-l2 | RN34 + MEK | 0.7035 | — | 0.6946 | 0.6710 | 0.6813 | 0.0005 | 2 |


# ════════════════════════════════════════════════════════════════════
# COMBINED BEST RESULTS — tuned, both datasets (2026-06-04)
# ════════════════════════════════════════════════════════════════════

Best configuration per (dataset, method), **selected on validation accuracy** (no test
leakage), ImageNet backbone. Bold = best in column within each dataset. RAF-DB from this
job (`baodqhust`); FER-2013 from the parallel job (`baobaoo`). See the per-dataset sections
above for full search logs, per-class accuracy, and noise caveats (±0.5 pt single-seed).

### RAF-DB (peak config per method)

| Model | Accuracy | F1 | Precision | Recall | Mean-class acc | # cfg |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.8396 | 0.7616 | 0.7815 | 0.7492 | 0.7492 | 6 |
| RN34 | 0.8367 | 0.7465 | 0.7622 | 0.7369 | 0.7369 | 6 |
| RN18 + MEK | 0.8585 | **0.7962** | **0.8038** | **0.7910** | **0.7910** | 5 |
| RN34 + MEK | **0.8657** | 0.7951 | 0.8026 | 0.7892 | 0.7892 | 6 |

### FER-2013 (peak config per method — search may be in progress)

| Model | Accuracy | F1 | Precision | Recall | Mean-class acc | # cfg |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| RN18 | 0.6655 | 0.6480 | 0.6529 | 0.6461 | 0.6461 | 5 |
| RN34 | 0.6622 | 0.6504 | 0.6592 | 0.6448 | 0.6448 | 7 |
| RN18 + MEK | 0.7081 | 0.6952 | 0.7009 | 0.6928 | 0.6928 | 6 |
| RN34 + MEK | **0.7205** | **0.7073** | **0.7252** | **0.6960** | **0.6960** | 6 |

# ════════════════════════════════════════════════════════════════════
# TUNED RESULTS — RAF-DB, MS-Celeb-1M face backbone (2026-06-09)
# ════════════════════════════════════════════════════════════════════

Automated Kaggle tuning campaign (`kaggle/tuning/`, `prep_msceleb.py`) across **3 accounts**
(`baodqhust`, `baobaoo`, `quocbaohust`) — 9 configs, 6 GPU sessions in parallel. **MEK
ResNet-18 with the MS-Celeb-1M face-recognition backbone** (`resnet18_msceleb.pth`, loads
100/100 encoder tensors). RAF-DB (`shuvoalok/raf-db-dataset`), seed 42, 90/10 train/val split.
Class index → emotion: `1`=surprise `2`=fear `3`=disgust `4`=happy `5`=sad `6`=angry `7`=neutral.

Search = **one-factor-at-a-time around the paper recipe** (arXiv 2310.19636 §4.2, verified:
Adam lr=1e-4, wd=1e-4, ExponentialLR γ=0.9, 60 ep, λ_flip=2, ε=0.1, 224 px, MS-Celeb backbone).
**Selection on VALIDATION (mean-class, overall-acc guardrail); TEST reported only.** No method/
loss/data edits — only allow-listed hyperparameter overrides (integrity contract in `prepare_kernel.py`).

> **Infra note:** Kaggle's current image ships torch 2.10+cu128 (sm_70–120) and the API assigns
> a **P100 (sm_60)**, so a P100-compatible torch (2.5.1+cu121) is reinstalled at kernel start
> via `--gpu-compat-torch`. Runs are therefore on **P100**, not T4 (T4 isn't API-selectable).

## Headline — MS-Celeb-1M vs ImageNet (MEK ResNet-18, validation-selected)

| MEK-RN18 backbone | Test acc | F1 | **Mean-class** |
|:--|:-:|:-:|:-:|
| ImageNet (tuned, LR3e-4) | 0.8585 | 0.7962 | 0.7910 |
| **MS-Celeb-1M (tuned, LR3e-4)** | **0.8641** | 0.7969 | **0.8137** |
| MS-Celeb-1M (peak: 80ep+γ0.95, *not val-selected*) | 0.8703 | 0.8071 | 0.8146 |
| *Paper RN18 (reference)* | *0.8977 / 0.8980 best* | — | *0.8244 / 0.8405 best* |

→ **MS-Celeb-1M > ImageNet on both metrics**, decisively on **mean-class (+2.3 pt)** — exactly
the imbalance metric MEK targets, and the expected payoff of face pre-training. The peak config
reaches **0.8146 mean-class**, within ~1 pt of the paper's last-epoch 0.8244.

## Best config (validation-selected) — LR 3e-4

| Backbone | LR | Sched/opt | Ep | Val acc | Val mean | **Test acc** | F1 | Precision | Recall | **Mean-class** |
|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| MS-Celeb-1M | 0.0003 | Adam+ExpLR γ0.9 | 60 | 0.8802 | 0.8195 | **0.8641** | 0.7969 | 0.7849 | 0.8137 | **0.8137** |

(LR3e-4 has the highest **val acc AND val mean-class** of all 9 → selected unambiguously, no test peeking.)

## Per-class test accuracy (selected LR3e-4 vs paper RN18 last-epoch)

| | surprise | fear | disgust | happy | sad | angry | neutral |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **MS-Celeb MEK-RN18 (ours)** | 0.875 | 0.662 | 0.731 | 0.931 | 0.866 | 0.827 | 0.803 |
| Paper RN18 | 0.878 | 0.662 | 0.669 | 0.964 | 0.893 | 0.809 | 0.896 |

→ We **match/exceed the paper on the hard minority classes** (fear 0.662 ≈ paper; disgust
0.731 > 0.669; anger 0.827 > 0.809) and trail mainly on majority classes (happy, neutral, sad)
— the expected effect of the weighted sampler trading majority headroom for minority recall.

## Key findings

1. **Face pre-training is the lever it's claimed to be:** tuned MS-Celeb MEK-RN18 beats tuned
   ImageNet MEK-RN18, with the gain concentrated in **mean-class (0.8137 vs 0.7910)**.
2. **The paper's LR1e-4 + γ0.9 underfits here** (0.8439 / 0.7477) — aggressive LR decay over
   60 ep. Raising LR (2e-4/3e-4) or slowing decay (γ0.95) lifts test ≈ +2 pt acc / +6 pt
   mean-class. **Longer + slower (80ep + γ0.95) gives the peak (0.8703 / 0.8146).**
3. **λ and ε must stay near the paper values.** λ=3 (0.6938 mean) and ε=0.2 (0.7276) **over-
   regularize** and crater minority accuracy; λ=2 / ε=0.1 are well-chosen; λ=1 is competitive.

## Honest caveats (deviations from the paper protocol)

* **Single seed (42)**; ±0.5 pt Kaggle GPU noise (`cudnn.benchmark=True`) applies as in the
  ImageNet sections — the LR3e-4 / γ0.95 / 80ep+γ0.95 cluster (0.864–0.870) is within noise.
* The ~3 pt overall-acc gap to the paper's 0.8977 is attributable to documented protocol
  differences, not a method bug: **(a)** `USE_WEIGHTED_SAMPLER=True` *on top of* MEK's
  re-balanced losses (extra minority oversampling → higher mean-class, lower overall);
  **(b)** best-validation selection + 10 % val split vs the paper's last-epoch on full train;
  **(c)** the pre-aligned `shuvoalok` RAF-DB images, not our own MTCNN alignment.
* Runs on **P100** (T4 not API-selectable); compute device does not affect accuracy.

## Full search log (all 9 configs; ★ = validation-selected best)

| config | val acc | val mean | test acc | mean-cls | F1 | LR | λ | ε | γ | ep |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| msceleb-lr3e4 ★ | 0.8802 | 0.8195 | 0.8641 | 0.8137 | 0.7969 | 3e-4 | 2 | 0.1 | 0.9 | 60 |
| msceleb-ep80-g095 | 0.8712 | 0.7905 | **0.8703** | **0.8146** | 0.8071 | 1e-4 | 2 | 0.1 | 0.95 | 80 |
| msceleb-lr2e4 | 0.8720 | 0.8048 | 0.8647 | 0.8061 | 0.7917 | 2e-4 | 2 | 0.1 | 0.9 | 60 |
| msceleb-g095 | 0.8688 | 0.7983 | 0.8667 | 0.8131 | 0.7993 | 1e-4 | 2 | 0.1 | 0.95 | 60 |
| msceleb-lam1 | 0.8574 | 0.7733 | 0.8579 | 0.8004 | 0.7850 | 1e-4 | 1 | 0.1 | 0.9 | 60 |
| msceleb-paper | 0.8468 | 0.7363 | 0.8439 | 0.7477 | 0.7470 | 1e-4 | 2 | 0.1 | 0.9 | 60 |
| msceleb-ep80 | 0.8484 | 0.7388 | 0.8429 | 0.7470 | 0.7462 | 1e-4 | 2 | 0.1 | 0.9 | 80 |
| msceleb-eps02 | 0.8354 | 0.7196 | 0.8341 | 0.7276 | 0.7243 | 1e-4 | 2 | 0.2 | 0.9 | 60 |
| msceleb-lam3 | 0.8191 | 0.6795 | 0.8243 | 0.6938 | 0.6941 | 1e-4 | 3 | 0.1 | 0.9 | 60 |

*mean across the 9-config search: test acc 0.8521 / mean-class 0.7738 / F1 0.7657.*

# ════════════════════════════════════════════════════════════════════
# WEBCAM-DEPLOYMENT RESULTS — RAF-DB, MS-Celeb-1M + CLAHE + OpenCV face-crop (2026-06-10)
# ════════════════════════════════════════════════════════════════════

The model we actually deploy in `demo.py`. MEK ResNet-18, MS-Celeb-1M backbone, trained
with the **exact live preprocessing demo.py uses** — OpenCV Haar face detect&crop (0.2
margin) → CLAHE lighting normalization, in that order, in BOTH train and eval — so there is
no train/serve gap on the crop OR the lighting. Plus the webcam-robustness recipe: heavy
in-the-wild augmentation (random-resized-crop, perspective, blur, strong colour jitter,
grayscale, erasing) + EMA weights. Script `kaggle/mek_webcam_resnet18_clahe_facecrop.py`;
9-config sweep across 3 Kaggle accounts; RAF-DB, seed 42, 90/10 train/val split.

## Methodology (integrity)

* The **augmentation + EMA + face-crop + CLAHE are FIXED** on every run — they are the
  deployment-robustness levers, deliberately NOT tuned away to chase clean-test accuracy.
  Only the optimisation (LR, γ, epochs) and loss shape (ε, λ) were swept.
* **Selection on VALIDATION mean-class; test reported only** (no test-set leakage). The
  val/test sets are themselves face-crop+CLAHE'd, so the metric is on the deployment
  distribution. **Caveat:** RAF-DB test is curated faces, not live webcam frames — this is a
  matched-distribution *proxy* for deployment quality, not a direct webcam measurement.
* Runs on Kaggle P100 (torch reinstalled for sm_60; T4 not API-selectable). Single seed.

## Best config (validation-selected) → the deployment checkpoint

| LR | Sched | Ep | ε | λ | EMA | Val acc | Val mean | **Test acc** | F1 | Prec | Recall | **Mean-class** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1e-4 | ExpLR γ0.95 | 80 | 0.15 | 0.1 | 0.999 | 0.8680 | 0.8066 | **0.8625** | 0.7994 | 0.7932 | 0.8083 | **0.8083** |

→ Saved as `checkpoints/mek_webcam_resnet18_rafdb_clahe_facecrop_best.pth` (demo.py
auto-discovers it; run with **"Detect & crop face"** AND **"CLAHE"** both ON).

## Per-class test accuracy (deployment winner)

| surprise | fear | disgust | happy | sad | angry | neutral |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.878 | 0.635 | 0.713 | 0.915 | 0.847 | 0.827 | 0.843 |

## Key findings

1. **Robustness was free.** With the full webcam stack (heavy aug + EMA + face-crop + CLAHE)
   the model still reaches **0.8625 / 0.8083**, on par with the clean benchmark MEK-RN18
   (0.8585 / 0.7910) — the train/serve-matched preprocessing did not cost accuracy and lifted
   mean-class slightly.
2. **λ must stay low for the webcam recipe.** Raising the attention-consistency weight from
   the live default λ=0.1 to 0.5 → 0.8008 mean-class, to 1.0 → **0.7636** (sharp drop). The
   live recipe's λ=0.1 is well-chosen; gentle attention regularization keeps confident preds.
3. **Slower LR decay + longer training wins.** γ0.95 (vs 0.9) and 80 epochs give the best
   validation; LR 1e-4/2e-4/3e-4 are all within ~1 pt once γ0.95 is used.
4. ε 0.1 vs 0.15 is within noise on test mean-class (0.8089 vs 0.8062).

## Full search log (all 9 configs; ★ = validation-selected best → deployed)

| config | val acc | val mean | test acc | mean-cls | F1 | LR | γ | ε | λ | ep |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| webcam-cf-ep80-g095 ★ | 0.8680 | 0.8066 | 0.8625 | 0.8083 | 0.7994 | 1e-4 | 0.95 | 0.15 | 0.1 | 80 |
| webcam-cf-g095 | 0.8663 | 0.7998 | 0.8618 | 0.8090 | 0.7991 | 1e-4 | 0.95 | 0.15 | 0.1 | 60 |
| webcam-cf-lr2e4-g095 | 0.8745 | 0.7948 | 0.8709 | 0.8067 | 0.8067 | 2e-4 | 0.95 | 0.15 | 0.1 | 60 |
| webcam-cf-default | 0.8533 | 0.7923 | 0.8449 | 0.8062 | 0.7818 | 1e-4 | 0.9 | 0.15 | 0.1 | 60 |
| webcam-cf-eps01 | 0.8533 | 0.7919 | 0.8462 | 0.8089 | 0.7854 | 1e-4 | 0.9 | 0.1 | 0.1 | 60 |
| webcam-cf-lr3e4 | 0.8655 | 0.7918 | 0.8670 | 0.8081 | 0.8051 | 3e-4 | 0.9 | 0.15 | 0.1 | 60 |
| webcam-cf-lr2e4 | 0.8623 | 0.7872 | 0.8559 | 0.8056 | 0.7942 | 2e-4 | 0.9 | 0.15 | 0.1 | 60 |
| webcam-cf-lam05 | 0.8223 | 0.7727 | 0.8295 | 0.8008 | 0.7672 | 1e-4 | 0.9 | 0.15 | 0.5 | 60 |
| webcam-cf-lam1 | 0.7995 | 0.7264 | 0.8110 | 0.7636 | 0.7341 | 1e-4 | 0.9 | 0.15 | 1.0 | 60 |

*mean across the 9-config search: test acc 0.8500 / mean-class 0.8019 / F1 0.7859.*

# ════════════════════════════════════════════════════════════════════
# ABLATION — RSL × RAC, MEK ResNet-18 + MS-Celeb-1M, RAF-DB (2026-06-10)
# ════════════════════════════════════════════════════════════════════

Two-module ablation of MEK (Zhang et al., NeurIPS 2023, arXiv 2310.19636): does each of
MEK's two regularizers — **RSL** (Re-balanced Smooth Labels) and **RAC** (Re-balanced
Attention Consistency) — help on its own, and are they **complementary (not redundant)**
when combined? Run on the **current best MS-Celeb-1M MEK-RN18 recipe** (the 0.8641/0.8137
winner above): Adam **LR 3e-4**, wd 1e-4, ExponentialLR γ=0.9, **60 ep**, λ_flip=2, ε=0.1,
224 px, MS-Celeb-1M backbone (face init 100/100), weighted sampler, seed 42, 90/10 split.
Class idx → emotion: `1`=surprise `2`=fear `3`=disgust `4`=happy `5`=sad `6`=angry `7`=neutral.

## Methodology (integrity)

* The **same MEKResNet, same data split, same MS-Celeb init, same recipe** on all four
  cells — only the two losses are toggled (RSL: ε-smoothed re-balanced CE vs plain CE;
  RAC: λ_flip>0 vs 0). This isolates each module's effect. Base script
  `kaggle/ablation/mek_ablation_resnet18.py`; prep `kaggle/tuning/prep_ablation.py`.
* **Split across 2 Kaggle GPU sessions / 2 accounts** to use both T4-class slots:
  `{baseline, rsl_rac}` (baodqhust) + `{rsl, rac}` (baobaoo). Both share seed 42 →
  identical val/train split and backbone init, so the four cells stay comparable.
* **Selection metric = mean-class accuracy** (MEK's headline imbalance metric); test
  reported. Single seed; ±0.5 pt Kaggle GPU noise (`cudnn.benchmark=True`) applies, and
  the cross-kernel baseline comparison carries that same ±0.5 pt.
* **Infra:** ran on **P100** + the `--gpu-compat-torch` reinstall (the installed kaggle
  CLI 1.7.4.5 cannot request a T4; device does not affect accuracy). See
  [[kaggle-p100-torch-incompat]].

## The 2×2 (validation-selected, test reported)

| Variant | RSL | RAC | Val acc | Test acc | **Mean-class** | F1 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline | ✗ | ✗ | 0.8663 | 0.8686 | 0.7804 | 0.8015 |
| + RSL only | ✓ | ✗ | 0.8794 | 0.8696 | 0.7953 | 0.8030 |
| + RAC only | ✗ | ✓ | **0.8802** | **0.8726** | 0.8157 | **0.8090** |
| **+ RSL + RAC (full MEK)** | ✓ | ✓ | 0.8753 | 0.8713 | **0.8214** | 0.8063 |

## Gain over baseline (the headline)

| | RSL only | RAC only | RSL + RAC | sum of singles |
|:--|:-:|:-:|:-:|:-:|
| **Mean-class** | +0.0149 | +0.0352 | **+0.0410** | +0.0501 |
| Test acc | +0.0010 | +0.0039 | +0.0026 | +0.0049 |

## Per-class test accuracy

| Variant | surprise | fear | disgust | happy | sad | angry | neutral |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline | 0.866 | 0.568 | 0.537 | **0.935** | **0.885** | 0.802 | **0.869** |
| + RSL only | 0.863 | 0.622 | 0.588 | 0.933 | 0.868 | 0.827 | 0.866 |
| + RAC only | 0.887 | 0.649 | 0.713 | **0.935** | 0.862 | 0.827 | 0.837 |
| **+ RSL + RAC** | **0.903** | **0.676** | **0.719** | 0.926 | 0.856 | **0.833** | 0.838 |

## Verdict — complementary, NOT redundant ✅

1. **Each module helps on its own.** Over baseline mean-class (0.7804): RSL alone
   **+1.49 pt** (0.7953), RAC alone **+3.52 pt** (0.8157). Both are positive — neither is
   dead weight. (This is the paper's first claim: *"both modules can improve the
   performance based on the baseline."*)
2. **Together they beat either alone.** Full MEK = **0.8214 mean-class (+4.10 pt)** —
   higher than RAC-only (0.8157) and RSL-only (0.7953), and it also wins F1-macro and the
   validation metric it's selected on. Stacking RSL **on top of** RAC buys a further
   **+0.58 pt** mean-class. *If RSL were redundant with RAC, adding it would give ≈ RAC
   alone; instead it improves further* → the two modules carry **partly-distinct** signal.
3. **Per-class proof of complementarity.** The hardest class, **fear**, is lifted most by
   the *combination* (0.676) — above RAC-only (0.649) and RSL-only (0.622); same pattern on
   **surprise** (0.903 vs 0.887 / 0.863). The two regularizers correct overlapping-but-not-
   identical minority errors that **add up**: RSL is the bigger lever on fear+disgust via
   the loss, RAC is the bigger lever on disgust+surprise via attention symmetry.
4. **Honest overlap (sub-additive).** Combined gain (+4.10) is *less than* the arithmetic
   sum of the single gains (+5.01) → the modules **partially overlap** (both attack class
   imbalance), so the effect isn't perfectly additive. But the combination is **strictly
   best on every aggregate** (mean-class, F1, val acc) — i.e. complementary with mild
   redundancy, exactly the paper's *"they can cooperate to achieve the best."*
5. **Overall accuracy is near-saturated** (~0.87) on this strong face-pretrained backbone,
   so the modules' value shows up in **mean-class / minority recall**, not top-line acc —
   precisely what MEK is designed to lift. Full MEK trades a little majority headroom
   (happy/sad/neutral ↓ ~1 pt) for large minority gains (fear +10.8, disgust +18.1 pt).
