"""Interactive FER demo — load a trained checkpoint and predict the emotion of an
uploaded image or a live webcam frame.

This is the "real demo" for the capstone defense: it runs the actual best
checkpoint end-to-end. For MEK checkpoints it also renders the per-class
attention-map overlay that visually justifies the method.

Checkpoints live in ./checkpoints and are auto-discovered. The filename encodes
everything needed to rebuild the model:
    mek_<arch>_<dataset>_best.pth   -> MEK model      (returns logits + attention)
    <arch>_<dataset>_best.pth       -> CNN baseline   (returns logits)

Run:
    python demo.py
    python demo.py --checkpoint checkpoints/mek_resnet18_rafdb_best.pth
    python demo.py --share            # public Gradio link (handy on Kaggle/Colab)

Requires: gradio  (pip install gradio).  Optional: opencv-python for face crop.
"""
import argparse
import glob
import os
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import gradio as gr

from src.models import build_model
from src.config import DATASETS as BASELINE_DATASETS
from mek.model import MEKResNet
from mek.config import DATASETS as MEK_DATASETS


HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")

# Class-index -> human label. The index is the ImageFolder order (folders sorted
# alphabetically), which differs between the two datasets:
#   • FER2013 (msambare): named folders -> alphabetical order below.
#   • RAF-DB (shuvoalok): numeric folders "1".."7"; the official RAF-DB label
#     semantics for class id k (1-based) are mapped here in index (k-1) order.
CLASS_NAMES = {
    "fer2013": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "rafdb":   ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"],
}

_JET = plt.get_cmap("jet")


# ──────────────────────────────────────────────────────────────────────
# Checkpoint -> model
# ──────────────────────────────────────────────────────────────────────
def parse_ckpt_name(path: str) -> dict:
    """Parse method/arch/dataset from the filename. Handles:
        resnet18_fer2013_best.pth         -> baseline
        mek_resnet18_rafdb_best.pth       -> mek
        mek_webcam_resnet18_rafdb_best.pth-> mek (webcam-trained; same architecture)
    """
    stem = os.path.basename(path)
    for suffix in ("_best.pth", ".pth"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_")
    if parts[0] == "mek":
        rest = parts[1:]
        if rest and rest[0] == "webcam":      # webcam variant is still a MEKResNet
            rest = rest[1:]
        return {"method": "mek", "arch": rest[0], "dataset": rest[1]}
    return {"method": "baseline", "arch": parts[0], "dataset": parts[1]}


def _eval_cfg(method: str, dataset: str):
    """Reuse the training config so the demo preprocesses exactly like eval."""
    factory = (MEK_DATASETS if method == "mek" else BASELINE_DATASETS)[dataset]
    return factory()


@lru_cache(maxsize=None)
def load_model(ckpt_path: str, device_str: str):
    """Build the right architecture, load weights, cache by (path, device)."""
    device = torch.device(device_str)
    meta = parse_ckpt_name(ckpt_path)
    cfg = _eval_cfg(meta["method"], meta["dataset"])

    if meta["method"] == "mek":
        model = MEKResNet(meta["arch"], num_classes=cfg.num_classes, pretrained=False)
    else:
        model = build_model(meta["arch"], num_classes=cfg.num_classes, pretrained=False)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, meta, cfg


# ──────────────────────────────────────────────────────────────────────
# Pre/post-processing
# ──────────────────────────────────────────────────────────────────────
def _transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.CenterCrop(cfg.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])


def detect_face(pil_img: Image.Image) -> Image.Image:
    """Crop the largest detected face (with margin). No-op if OpenCV is missing
    or no face is found — webcam frames have a lot of background, the datasets
    are tight face crops, so this keeps the demo honest."""
    try:
        import cv2
    except ImportError:
        return pil_img
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return pil_img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.2 * max(w, h))
    h_img, w_img = rgb.shape[:2]
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(w_img, x + w + m), min(h_img, y + h + m)
    return Image.fromarray(rgb[y0:y1, x0:x1])


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """Normalize lighting via CLAHE on the L channel — webcam exposure varies a
    lot vs. the curated datasets. No-op if OpenCV is missing."""
    try:
        import cv2
    except ImportError:
        return pil_img
    rgb = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def attention_overlay(face_pil: Image.Image, hm_class: torch.Tensor, size: int) -> Image.Image:
    """Overlay one class's attention map (e.g. 7×7) on the face crop."""
    hm = hm_class.detach().float().cpu()
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    hm = F.interpolate(hm[None, None], size=(size, size),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = _JET(hm)[..., :3]                                  # RGB float 0..1
    base = np.asarray(face_pil.resize((size, size)).convert("RGB"), dtype=np.float32) / 255.0
    blended = (0.5 * base + 0.5 * heat) * 255.0
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────
def predict(image, ckpt_path, use_face_detect, use_clahe):
    if image is None or not ckpt_path:
        return {}, None, "Upload an image (or use the webcam) and pick a checkpoint."

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, cfg = load_model(ckpt_path, device_str)
    device = torch.device(device_str)

    face = detect_face(image) if use_face_detect else image.convert("RGB")
    if use_clahe:
        face = apply_clahe(face)
    # FER2013 was trained on grayscale (ImageFolder replicates it to 3 channels),
    # so match that distribution or a colour webcam frame is off-distribution.
    if meta["dataset"] == "fer2013":
        face = face.convert("L").convert("RGB")
    x = _transform(cfg)(face).unsqueeze(0).to(device)

    # Flip-TTA: average the prediction over the frame and its mirror. The MEK
    # model is trained for attention-flip consistency (AC loss), so the two views
    # agree by construction — averaging them is a free, robustness-neutral gain.
    x_flip = torch.flip(x, dims=[3])
    with torch.no_grad():
        if meta["method"] == "mek":
            logits, hm = model(x)
            logits_f, _ = model(x_flip)
        else:
            logits, hm = model(x), None
            logits_f = model(x_flip)
    probs = (0.5 * (logits.softmax(1) + logits_f.softmax(1)))[0].cpu().numpy()

    names = CLASS_NAMES.get(meta["dataset"], [str(i) for i in range(cfg.num_classes)])
    label_dict = {names[i]: float(probs[i]) for i in range(len(names))}

    overlay = None
    if hm is not None:
        pred = int(probs.argmax())
        overlay = attention_overlay(face, hm[0, pred], cfg.crop_size)

    info = (f"**{meta['method'].upper()}** · {meta['arch']} · trained on "
            f"**{meta['dataset']}** · device={device_str}")
    return label_dict, overlay, info


def predict_live(frame, ckpt_path, use_face_detect, use_clahe, every, state):
    """Throttled handler for the streaming webcam. Gradio fires this for every frame
    it grabs; we only run the (relatively expensive) model at most once per `every`
    seconds and replay the last result in between, so the live view stays smooth even
    on CPU. `state` (a gr.State) carries the last timestamp + outputs per session."""
    if state is None:
        state = {"t": 0.0, "out": ({}, None, "Waiting for webcam…")}
    now = time.monotonic()
    if frame is None or (now - state["t"]) < float(every):
        return (*state["out"], state)
    out = predict(frame, ckpt_path, use_face_detect, use_clahe)
    state = {"t": now, "out": out}
    return (*out, state)


# ──────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────
def build_interface(default_ckpt=None):
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {CKPT_DIR}. Place your *_best.pth files there."
        )

    # Best webcam default: prefer webcam-trained > MEK > RAF-DB (RGB, in-the-wild).
    def _score(path):
        n = os.path.basename(path)
        return (("webcam" in n) * 4) + (n.startswith("mek_") * 2) + (("rafdb" in n) * 1)

    if default_ckpt and default_ckpt in ckpts:
        default = default_ckpt
    else:
        default = max(ckpts, key=_score)

    with gr.Blocks(title="Facial Emotion Recognition — demo") as demo:
        gr.Markdown(
            "# Facial Emotion Recognition demo\n"
            "Pick a trained checkpoint, then use the **Live webcam** tab for continuous "
            "prediction or the **Upload image** tab for a single image. "
            "MEK checkpoints also show the predicted class's **attention heatmap**."
        )
        # Shared controls (apply to both tabs).
        with gr.Row():
            ckpt_dd = gr.Dropdown(choices=ckpts, value=default, label="Checkpoint", type="value")
            face_cb = gr.Checkbox(value=True, label="Detect & crop face (OpenCV, if installed)")
            clahe_cb = gr.Checkbox(value=True, label="CLAHE lighting normalization (recommended for webcam)")

        with gr.Row():
            with gr.Column():
                with gr.Tab("Live webcam"):
                    live_in = gr.Image(type="pil", sources=["webcam"], streaming=True,
                                       label="Live webcam (predicts continuously)")
                    every_sl = gr.Slider(1, 10, value=3, step=1,
                                         label="Predict every N seconds")
                with gr.Tab("Upload image"):
                    img_in = gr.Image(type="pil", sources=["upload", "webcam"], label="Input image")
                    btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label_out = gr.Label(num_top_classes=7, label="Predicted emotion")
                heat_out = gr.Image(label="MEK attention (predicted class)")
                info_out = gr.Markdown()

        # Single-shot (upload tab).
        btn.click(predict, [img_in, ckpt_dd, face_cb, clahe_cb], [label_out, heat_out, info_out])

        # Continuous live prediction (webcam tab). Gradio streams frames; predict_live
        # throttles to one inference per `every` seconds. gr.State keeps per-session
        # cadence + last result so in-between frames replay instantly.
        live_state = gr.State()
        live_in.stream(
            predict_live,
            inputs=[live_in, ckpt_dd, face_cb, clahe_cb, every_sl, live_state],
            outputs=[label_out, heat_out, info_out, live_state],
            concurrency_limit=1,
            show_progress="hidden",
        )

    return demo


def main():
    p = argparse.ArgumentParser(description="Interactive FER demo.")
    p.add_argument("--checkpoint", default=None,
                   help="Preselect a checkpoint path (default: first in ./checkpoints).")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    demo = build_interface(args.checkpoint)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
