# FER on a budget: ResNet baselines + MEK

7-class facial-expression recognition on **FER-2013** and **RAF-DB** with torchvision
ResNet-18/34. We re-implement **MEK** (*Mine Extra Knowledge*, Zhang et al., NeurIPS 2023,
[arXiv:2310.19636](https://arxiv.org/abs/2310.19636)) — two imbalance-aware regularisers —
alongside a tuned CNN baseline, the paper's MS-Celeb-1M backbone reproduction, a controlled
RSL×RAC ablation, a webcam-robust variant, and a live Gradio demo.

**Pretrained checkpoints:** the six released models are hosted on Hugging Face —
[`ToiTenBao/fer-mek-checkpoints`](https://huggingface.co/ToiTenBao/fer-mek-checkpoints).

The headline metric is **mean-class accuracy** (the unweighted mean of per-class recalls),
not overall accuracy: under severe class imbalance (~17× gap between *happy* and *fear*),
overall accuracy hides the collapse of the minority classes that MEK is built to fix.

> **Environment:** run everything in the **`baodq` conda env** (never `base`); install extra
> packages there. Datasets use the `train/<class>/`, `test/<class>/` `ImageFolder` layout, so
> any Kaggle FER/RAF mirror works (pass `--data-root`).

| Component | Entry point |
|---|---|
| CNN baseline | `train.py`, `kaggle/resnet*.py` |
| MEK (ImageNet) | `train_mek.py`, `kaggle/mek_resnet*.py` |
| **MEK + MS-Celeb-1M** (paper reproduction) | `train_mek.py --face-weights …`, `kaggle/mek_resnet18_MS-Celeb-1M.py` |
| Ablation (RSL × RAC) | `ablation/train.py`, `kaggle/ablation/` |
| Webcam-robust + live demo | `train_mek_webcam.py`, `demo.py` |
| Technical report (IEEE) | `docs/report/main.tex` → `docs/report/main.pdf` |

## Results (validation-selected, test reported)

Best config per method; **mC** = mean-class accuracy. MEK lifts mean-class by **+4–5 points**
on both datasets, with the gains concentrated on the rare classes and overall accuracy
preserved or improved.

| Dataset | Method | Acc | F1 | mC |
|---|---|:-:|:-:|:-:|
| RAF-DB | RN18 | 0.8396 | 0.7616 | 0.7492 |
| RAF-DB | RN18 + MEK | 0.8585 | 0.7962 | 0.7910 |
| RAF-DB | RN34 + MEK | 0.8657 | 0.7951 | 0.7892 |
| RAF-DB | **RN18 + MEK + MS-Celeb-1M** | **0.8641** | **0.7969** | **0.8137** |
| FER-2013 | RN18 | 0.6655 | 0.6480 | 0.6461 |
| FER-2013 | RN34 + MEK | 0.7205 | 0.7073 | 0.6960 |

**The backbone is the dominant lever.** Swapping the ImageNet ResNet-18 encoder for an
**MS-Celeb-1M face-pretrained** one lifts RAF-DB mean-class **0.7910 → 0.8137** (+2.3 pt) —
more than going from RN18 to RN34 — and matches the original paper on the hardest minority
classes (disgust **0.731** vs. paper 0.669; fear 0.662 = paper). Full numbers, training
curves, the ablation, and a grounded comparison to the paper are in
[`docs/report/main.pdf`](docs/report/main.pdf).

### Verified checkpoints + confusion structure

The two headline checkpoints (RAF-DB MS-Celeb MEK-RN18, FER-2013 MEK-RN34) were re-run on
the official test splits: the recomputed overall / mean-class / per-class accuracies reproduce
the reported tables **to four decimals**. The row-normalised confusion matrices show the
residual errors are structured, not random — e.g. on RAF-DB the top confusion is
**fear → surprise (13.5%)**; on FER-2013, **disgust → angry (19.8%)**.

## MEK in brief

- **RSL** — re-balanced smooth labels: the label-smoothing mass is spread over non-target
  classes by *inverse class frequency*, so rare classes get more soft probability.
- **RAC** — re-balanced attention consistency: per-class 7×7 CAM attention maps must agree
  between an image and its deterministic horizontal flip, weighted by inverse frequency.

`loss = RSL(logits, y) + λ · RAC(attn, attn_flip)`. The paired-flip dataloader yields
`(img, label, hflip(img))` and the train transform omits `RandomHorizontalFlip` (a random
flip would break RAC's assumption). Paper's ResNet recipe: Adam @ 1e-4, ExponentialLR γ=0.9,
ε=0.1, λ=2, 224×224.

## Training hyperparameters

Per-(dataset, arch) winners live in the `make_train_cfg` functions in `src/config.py`
(baseline) and `mek/config.py` (MEK).

| Recipe | Dataset | Arch | Optim. | LR | Ep | ε | λ |
|---|---|---|---|:-:|:-:|:-:|:-:|
| Baseline | RAF-DB | RN18 | SGD+Cosine | 3e-3 | 80 | 0.05 | — |
| Baseline | FER-2013 | RN18 | SGD+Cosine | 1.4e-2 | 100 | 0.10 | — |
| MEK | RAF-DB | RN18 | Adam+ExpLR γ0.9 | 3e-4 | 60 | 0.10 | 2.0 |
| MEK | RAF-DB | RN34 | Adam+ExpLR γ0.9 | 2e-4 | 60 | 0.20 | 2.0 |
| MEK | FER-2013 | RN34 | Adam+ExpLR γ0.9 | 1e-4 | 80 | 0.10 | 0.10 |
| MEK (webcam) | RAF-DB | RN18 | Adam+ExpLR γ0.95 | 1e-4 | 80 | 0.15 | 0.10 |

All runs: seed 42, 90/10 train/val split, weighted sampler; MEK adds grad-clip 5, AMP, and
early stopping (patience 25) with best-weight restoration.

## Pretrained checkpoints (Hugging Face)

The best checkpoints are published as a HF model repo: **`ToiTenBao/fer-mek-checkpoints`**
(see `scripts/push_to_hf.py`). Each is a `MEKResNet` `state_dict` (7 classes, 224×224).

```python
import torch
from huggingface_hub import hf_hub_download
from mek.model import MEKResNet

path = hf_hub_download("ToiTenBao/fer-mek-checkpoints",
                       "mek_resnet18_rafdb_msceleb_best.pth")
model = MEKResNet("resnet18", num_classes=7, pretrained=False)
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
model.eval()
```

To (re)publish: add `HUGGING_FACE_API_KEY=...` (write scope) to `.env`, then
`python scripts/push_to_hf.py --repo ToiTenBao/fer-mek-checkpoints`.

## The face backbone

MEK pre-trains its ResNet on **MS-Celeb-1M (faces), not ImageNet** — the biggest factor in
matching the paper. Get the checkpoint and pass it in:

```bash
pip install gdown
gdown 1EEx7qVCums-TM5fiblepgY70MDqIxbVz -O resnet18_msceleb.pth   # author's RUL repo
```

Loaders fall back to ImageNet if the file is missing — check the log for `Face init: filled
N/M` (the published checkpoint fills 100/100). A `.pth` loads via the safe `weights_only=True`
path; a `.pkl` is unpickled only against a source-pinned SHA-256 allow-list.

## Quick start

```bash
# CNN baseline
python train.py     --arch resnet18 --dataset fer2013 --data-root data/fer2013

# MEK (ImageNet) and MEK + MS-Celeb-1M (reproduces paper RN18)
python train_mek.py --arch resnet18 --dataset rafdb --data-root data/rafdb/DATASET
python train_mek.py --arch resnet18 --dataset rafdb --face-weights resnet18_msceleb.pth

# Ablation (baseline / rsl / rac / rsl_rac) and webcam-robust model
python ablation/train.py     --dataset rafdb --face-weights resnet18_msceleb.pth
python train_mek_webcam.py   --face-weights resnet18_msceleb.pth   # → demo.py checkpoint

# Live demo (Gradio): upload tab + continuous-prediction webcam tab
python demo.py
```

Add `--wandb` to any trainer to log to W&B (reads `WANDB_API_KEY` from `.env`).
`--resume CKPT --eval-only` re-runs test only. **Kaggle**: paste a `kaggle/*.py` script, set
the toggles at the top (`DATASET`, `USE_FACE_BACKBONE`, `FACE_WEIGHTS_PATH`), run.

## Building the report

```bash
conda run -n baodq python docs/report/figures/make_figures.py        # vector data figures
conda run -n baodq python docs/report/figures/make_extra_figures.py  # training curves (from logs) + extras
cd docs/report && conda run -n baodq tectonic main.tex               # → main.pdf
```

## References

- MEK paper — Zhang et al., *Leave No Stone Unturned*, NeurIPS 2023: [arXiv:2310.19636](https://arxiv.org/abs/2310.19636).
- MEK code (Swin-T): [zyh-uaiaaaa/Mine-Extra-Knowledge](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge).
- MS-Celeb-1M `resnet18_msceleb.pth`: [zyh-uaiaaaa/Relative-Uncertainty-Learning](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning).
