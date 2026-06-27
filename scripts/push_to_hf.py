"""Push the best FER/MEK checkpoints to a Hugging Face model repo.

Usage:
    python scripts/push_to_hf.py --repo <hf-username>/fer-mek-checkpoints
    python scripts/push_to_hf.py --repo <user>/<repo> --private

Auth: reads HUGGING_FACE_API_KEY (or HF_TOKEN) from the local .env (python-dotenv)
or the environment. No token is ever printed.

The curated set is the validation-selected best checkpoint per (dataset x arch x
method); metrics are the test numbers verified in docs/report (see VERIFIED note).
All files are MEKResNet state_dicts (load with mek.model.MEKResNet then
load_state_dict), 7 classes, 224x224 input.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (source path, dest filename, description, test_acc, mean_class, backbone)
CHECKPOINTS = [
    ("kaggle/tuning/runs/msceleb/baodqhust/rafdb-mek-rn18-msceleb-lr3e4/output/"
     "mek_resnet18_rafdb_msceleb_best.pth",
     "mek_resnet18_rafdb_msceleb_best.pth",
     "RAF-DB - MEK ResNet-18 - MS-Celeb-1M backbone (headline best, verified)",
     0.8641, 0.8137, "MS-Celeb-1M"),
    ("kaggle/tuning/runs/mek-rn18-lr3e4-mc/output/mek_resnet18_rafdb_best.pth",
     "mek_resnet18_rafdb_imagenet_best.pth",
     "RAF-DB - MEK ResNet-18 - ImageNet backbone",
     0.8585, 0.7910, "ImageNet"),
    ("kaggle/tuning/runs/mek-rn34-rafdb-lr2e4-eps2/output/mek_resnet34_rafdb_best.pth",
     "mek_resnet34_rafdb_imagenet_best.pth",
     "RAF-DB - MEK ResNet-34 - ImageNet backbone (best overall acc)",
     0.8657, 0.7892, "ImageNet"),
    ("kaggle/tuning/runs/fer2013-mek-resnet34-lr1e4-l01/output/mek_resnet34_fer2013_best.pth",
     "mek_resnet34_fer2013_imagenet_best.pth",
     "FER-2013 - MEK ResNet-34 - ImageNet backbone (best FER, verified)",
     0.7205, 0.6960, "ImageNet"),
    ("kaggle/tuning/runs/fer2013-mek-resnet18-lr1e4-l025/output/mek_resnet18_fer2013_best.pth",
     "mek_resnet18_fer2013_imagenet_best.pth",
     "FER-2013 - MEK ResNet-18 - ImageNet backbone",
     0.7081, 0.6928, "ImageNet"),
    ("checkpoints/mek_webcam_resnet18_rafdb_clahe_facecrop_best.pth",
     "mek_webcam_resnet18_rafdb_clahe_facecrop_best.pth",
     "RAF-DB - webcam-robust MEK ResNet-18 - MS-Celeb-1M + EMA + face-crop/CLAHE (demo.py)",
     0.8625, 0.8083, "MS-Celeb-1M"),
]


def load_token():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
    for k in ("HUGGING_FACE_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(k)
        if tok:
            return tok
    sys.exit("ERROR: no HF token found. Add HUGGING_FACE_API_KEY=... to .env "
             "(get one at https://huggingface.co/settings/tokens, 'write' scope).")


def model_card(repo):
    rows = "\n".join(
        f"| `{d}` | {desc} | {acc:.4f} | {mc:.4f} | {bb} |"
        for _, d, desc, acc, mc, bb in CHECKPOINTS)
    return f"""---
license: mit
tags:
- facial-expression-recognition
- image-classification
- pytorch
- resnet
- imbalanced-classification
library_name: pytorch
---

# FER / MEK checkpoints (ResNet-18/34)

7-class facial-expression recognition checkpoints for **MEK** (*Mine Extra Knowledge*,
Zhang et al., NeurIPS 2023, [arXiv:2310.19636](https://arxiv.org/abs/2310.19636))
re-implemented on torchvision ResNet-18/34. Trained and evaluated on **RAF-DB** and
**FER-2013** under a leakage-free validation protocol (config selected on a 10% val split;
test reported only). Headline metric: **mean-class accuracy** (unweighted mean of per-class
recalls), the right objective under severe class imbalance.

Code: <https://github.com/frichickens/CV-Project-HUST>.

## Checkpoints

| File | Description | Test acc | Mean-class | Backbone |
|---|---|:-:|:-:|:-:|
{rows}

All files are `MEKResNet` `state_dict`s (7 classes, 224x224 input). The `rafdb_msceleb`
and `fer_rn34` checkpoints are end-to-end **verified**: re-running them reproduces the
reported overall / mean-class / per-class accuracy to four decimals.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from mek.model import MEKResNet   # from the project repo

path = hf_hub_download("{repo}", "mek_resnet18_rafdb_msceleb_best.pth")
model = MEKResNet("resnet18", num_classes=7, pretrained=False)
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
model.eval()
logits, attn = model(images)            # images: [B,3,224,224], ImageNet-normalised
```

RAF-DB class index -> emotion: `0`=surprise `1`=fear `2`=disgust `3`=happy `4`=sad
`5`=angry `6`=neutral (folders `1..7` sorted as strings). FER-2013 classes are the folder
names in alphabetical order: angry, disgust, fear, happy, neutral, sad, surprise.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. user/fer-mek-checkpoints")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = load_token()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"Authenticated as: {who}")

    # verify all sources exist before touching the hub
    for src, *_ in CHECKPOINTS:
        if not os.path.isfile(os.path.join(ROOT, src)):
            sys.exit(f"ERROR missing source checkpoint: {src}")

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{args.repo}  (private={args.private})")

    # model card
    card = os.path.join(ROOT, "scripts", "_hf_README.md")
    with open(card, "w", encoding="utf-8") as f:
        f.write(model_card(args.repo))
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    os.remove(card)

    for src, dest, desc, *_ in CHECKPOINTS:
        print(f"uploading {dest} ...")
        api.upload_file(path_or_fileobj=os.path.join(ROOT, src),
                        path_in_repo=dest, repo_id=args.repo, repo_type="model",
                        commit_message=f"add {dest}: {desc}")
    print(f"\nDONE -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
