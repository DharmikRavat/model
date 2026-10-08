"""Torch inference for the EfficientNet-B2 plant disease model.

  predict(image_path_or_PIL) -> dict   (quality / plant check / disease / confidence / top-k)
  gradcam(image)            -> (PIL.Image overlay, heatmap ndarray)

Used by api.py and the CLI:  python predict_b2.py --image leaf.jpg
"""
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

import config

MODEL_PATH = os.path.join(config.MODELS_DIR, "plant_disease_b2.pt")
CFG_PATH = os.path.join(config.MODELS_DIR, "b2_config.json")

_BUNDLE = None


# ---------------------------------------------------------------- loading
def load_bundle(force: bool = False):
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    import torch
    from PIL import Image  # noqa: F401
    from torchvision import models
    import torch.nn as nn

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"FATAL: {MODEL_PATH} missing. Run: python train_b2.py")
    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    model = models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.4),
                                     nn.Linear(model.classifier[1].in_features, len(cfg["class_names"])))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _BUNDLE = (model, cfg, device)
    return _BUNDLE


def _transform(cfg, train: bool = False):
    from torchvision import transforms

    size = cfg["image_size"]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def _cam_transform(cfg, size: int):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def _to_pil(image):
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8")).convert("RGB")
    raise TypeError(type(image))


# ---------------------------------------------------------------- quality gate
def check_image_quality(image) -> dict:
    """Reuses the shared thresholds in config.py (dark / bright / tiny / blurry)."""
    import cv2

    im = _to_pil(image)
    arr = np.asarray(im)
    w, h = im.size
    issues, details = [], {"size": f"{w}x{h}"}
    if w < config.MIN_IMAGE_SIDE or h < config.MIN_IMAGE_SIDE:
        issues.append(f"too small ({w}x{h}, minimum {config.MIN_IMAGE_SIDE}px)")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mean_luma = float(gray.mean())
    details["mean_luma"] = round(mean_luma, 1)
    if mean_luma < config.DARK_MEAN:
        issues.append(f"extremely dark (mean luma {mean_luma:.0f})")
    elif mean_luma > config.BRIGHT_MEAN:
        issues.append(f"extremely bright (mean luma {mean_luma:.0f})")
    g = cv2.resize(gray, (config.QUALITY_REPORT_SIDE, config.QUALITY_REPORT_SIDE))
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    details["laplacian_variance"] = round(lap, 1)
    if lap < config.BLUR_MIN_VARIANCE:
        issues.append(f"excessive blur (laplacian variance {lap:.1f})")
    return {"status": "POOR" if issues else "GOOD", "issues": issues, "details": details,
            "width": w, "height": h}


def green_fraction(image) -> float:
    import cv2

    im = _to_pil(image)
    im.thumbnail((256, 256))
    arr = np.asarray(im)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35)).astype(np.uint8)
    return float(mask.mean())


# ---------------------------------------------------------------- prediction
def predict(image, top_k: int = 3) -> dict:
    import torch

    started = time.time()
    model, cfg, device = load_bundle()
    classes = cfg["class_names"]
    res = {
        "image_name": os.path.basename(image) if isinstance(image, (str, os.PathLike)) else "uploaded",
        "quality": None, "quality_issues": [], "quality_details": {},
        "plant_detected": False, "plant": None,
        "prediction": None, "confidence": None, "top_predictions": [],
        "health_status": None, "status": None, "message": None, "elapsed_sec": None,
    }

    q = check_image_quality(image)
    res.update(quality=q["status"], quality_issues=q["issues"], quality_details=q["details"])
    if q["status"] == "POOR":
        res.update(status="IMAGE QUALITY: POOR",
                   message="Please upload a clearer, well-lit plant image.")
        res["elapsed_sec"] = round(time.time() - started, 3)
        return res

    gf = green_fraction(image)
    res["plant"] = {"method": "efficientnet_open_set", "green_fraction": round(gf, 4)}

    pil = _to_pil(image)
    tensor = _transform(cfg)(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor)
        probs = logits.float().softmax(1)[0].cpu().numpy()

    max_p = float(probs.max())
    order = np.argsort(probs)[::-1]
    best_index = int(order[0])
    best = classes[best_index]

    if best == config.OTHER_PLANT_CLASS:
        res.update(
            status="NO PLANT DETECTED",
            message=("This appears to be a plant outside the supported tomato, potato, "
                     "and pepper classes. Upload a supported plant leaf for diagnosis."),
            confidence=round(max_p, 6),
            plant={**res["plant"], "rejection": "unsupported_plant_class"},
        )
        res["elapsed_sec"] = round(time.time() - started, 3)
        return res

    plant_ok = gf >= 0.10 and max_p >= 0.35
    res["plant_detected"] = bool(plant_ok)
    if not plant_ok:
        res.update(status="NO PLANT DETECTED",
                   message=("This image does not look like a leaf of the trained plant species. "
                            "Upload a clear, close-up photo of a leaf."))
        res["confidence"] = round(max_p, 6)
        res["top_predictions"] = [
            {"rank": i + 1, "class": classes[int(j)], "confidence": round(float(probs[int(j)]), 6)}
            for i, j in enumerate(
                [j for j in order if classes[int(j)] != config.OTHER_PLANT_CLASS][:top_k])]
        res["elapsed_sec"] = round(time.time() - started, 3)
        return res

    res["prediction"] = best
    res["confidence"] = round(float(probs[best_index]), 6)
    res["top_predictions"] = [
        {"rank": i + 1, "class": classes[int(j)], "confidence": round(float(probs[int(j)]), 6)}
        for i, j in enumerate(
            [j for j in order if classes[int(j)] != config.OTHER_PLANT_CLASS][:top_k])]
    res["health_status"] = "HEALTHY" if "healthy" in best.lower() else "DISEASE SUSPECTED"
    res["status"] = config.confidence_label(res["confidence"])
    if res["confidence"] < config.MEDIUM_CONFIDENCE:
        res["status"] = "UNCERTAIN"
        res["message"] = config.UNCERTAIN_MESSAGE + " Upload a clearer, close-up leaf image."
    elif res["health_status"] == "HEALTHY":
        res["message"] = "No disease symptoms detected in the top prediction."
    else:
        res["message"] = "Disease predicted from leaf imagery."
    res["elapsed_sec"] = round(time.time() - started, 3)
    return res


# ---------------------------------------------------------------- Grad-CAM
def gradcam(image, class_index: Optional[int] = None, out_size: int = 384,
            cam_px: Optional[int] = None):
    """Returns (overlay PIL image, heatmap np.ndarray [0..1]).

    The CAM pass runs at a higher resolution than the classifier by default so the
    last feature map is 16x16 instead of 8x8 - small lesions then show up as their
    own blobs instead of one smeared average over the leaf. Falls back to the
    classifier resolution if the GPU cannot spare the memory."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, cfg, device = load_bundle()
    pil = _to_pil(image)
    base_px = int(cfg["image_size"])
    sizes = [cam_px] if cam_px else [max(base_px * 2, 512), base_px]

    target_layer = model.features[-1]          # last Conv2dNormActivation block
    activations, gradients = [], []

    def fwd_hook(_m, _i, o):
        activations.append(o)

    def bwd_hook(_m, _g, grad):
        gradients.append(grad)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    def _cam_at(px: int) -> np.ndarray:
        activations.clear()
        gradients.clear()
        model.zero_grad(set_to_none=True)
        tensor = _cam_transform(cfg, px)(pil).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor)
            probs = logits.float().softmax(1)
            if class_index is None:
                class_index_local = int(probs.argmax(1))
            else:
                class_index_local = class_index
            score = logits[0, class_index_local]
        score.backward()
        act = activations[0]
        grad = gradients[0][0]
        if act.dim() == 4:            # forward hook gives [1,C,H,W]
            act = act[0]
        if grad.dim() == 4:           # backward hook gives a tuple of [1,C,H,W]
            grad = grad[0]
        act = act.float()
        grad = grad.float()
        weights = grad.mean(dim=(1, 2))                       # [C]
        cam = (weights[:, None, None] * act).sum(0)           # [H,W]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(out_size, out_size),
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()

    try:
        heat = None
        last_err: Optional[RuntimeError] = None
        for px in sizes:
            try:
                heat = _cam_at(px)
                break
            except RuntimeError as e:                       # CUDA OOM / cuDNN workspace
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                last_err = e
        if heat is None:
            raise last_err if last_err else RuntimeError("Grad-CAM failed")
    finally:
        h1.remove()
        h2.remove()

    base = pil.resize((out_size, out_size), Image.BILINEAR)
    arr = np.asarray(base).astype(np.float32) / 255.0
    overlay = _colorize(heat) * 0.45 + arr * 0.55
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(overlay), heat


def _colorize(heat: np.ndarray) -> np.ndarray:
    import cv2

    h8 = (np.clip(heat, 0, 1) * 255).astype("uint8")
    colored = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def structured_payload(res: dict) -> dict:
    return {
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "top_predictions": [{"class": t["class"], "confidence": t["confidence"]}
                            for t in res["top_predictions"]],
        "health_status": res["health_status"],
        "image": res["image_name"],
        "image_quality": res["quality"],
        "plant_detected": res["plant_detected"],
        "plant": (res["plant"] or {}).get("species"),
        "status": res["status"],
        "message": res["message"],
    }


def format_report(res: dict) -> str:
    lines = ["=" * 40, "AI URBAN FARMING - EfficientNet-B2", "=" * 40,
             f"Image:        {res['image_name']}",
             f"Quality:      {res['quality']}" +
             ("" if not res["quality_issues"] else " -> " + "; ".join(res["quality_issues"])),
             f"Plant seen:   {'YES' if res['plant_detected'] else 'NO'}"]
    if res["plant_detected"] and res["prediction"]:
        lines += ["", f"Prediction:   {res['prediction']}",
                  f"Confidence:   {res['confidence'] * 100:.2f}%", "", "Top 3:"]
        lines += [f"  {t['rank']}. {t['class']} - {t['confidence'] * 100:.2f}%"
                  for t in res["top_predictions"]]
        lines += ["", f"Health:       {res['health_status']}", f"Status:       {res['status']}",
                  f"Message:      {res['message']}"]
    else:
        lines += ["", f"Status:       {res['status']}", f"Message:      {res['message']}"]
    lines += [f"Elapsed:      {res['elapsed_sec']}s", "=" * 40]
    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="EfficientNet-B2 plant disease prediction")
    ap.add_argument("--image", required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--gradcam", action="store_true", help="also save outputs/gradcam_<name>.png")
    a = ap.parse_args()

    r = predict(a.image)
    if a.json:
        print(json.dumps(structured_payload(r), indent=2))
    else:
        print(format_report(r))
    if a.gradcam and r["plant_detected"]:
        idx = None
        if r["prediction"]:
            _, cfg, _ = load_bundle()
            idx = cfg["class_names"].index(r["prediction"])
        img, _heat = gradcam(a.image, class_index=idx)
        out = os.path.join(config.OUTPUTS_DIR,
                           "gradcam_" + os.path.basename(a.image).rsplit(".", 1)[0] + ".png")
        config.ensure_dirs()
        img.save(out)
        print(f"Grad-CAM -> {out}")


if __name__ == "__main__":
    main()
