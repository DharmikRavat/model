"""Terminal inference: image quality check -> plant detection -> disease prediction.

Run:
  python predict.py --image "path/to/image.jpg"
  python predict.py --folder "test_images/"
  python predict.py --image "leaf.jpg" --json
  python predict.py --folder "test_images/" --csv outputs/predictions.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import config

_BUNDLE = None


# ---------------------------------------------------------------- model loading
def load_bundle(force: bool = False):
    """Load (disease_model, class_names, plant_model, plant_class_names) once."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    import keras

    if not os.path.isfile(config.MODEL_PATH):
        raise SystemExit(f"FATAL: model not found at {config.MODEL_PATH}. Run: python train.py")
    disease_model = keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        class_names = json.load(fh)

    plant_model, plant_classes = None, []
    if os.path.isfile(config.PLANT_MODEL_PATH):
        try:
            plant_model = keras.models.load_model(config.PLANT_MODEL_PATH)
            with open(config.PLANT_CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
                plant_classes = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] plant classifier could not be loaded ({e}); using colour heuristic instead")
    _BUNDLE = (disease_model, class_names, plant_model, plant_classes)
    return _BUNDLE


# ---------------------------------------------------------------- quality checks
def check_image_quality(path: str) -> dict:
    """Corrupt / too small / too dark / too bright / blurry -> POOR."""
    issues = []
    details = {}
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            rgb = np.asarray(im)
    except Exception as e:  # noqa: BLE001
        return {"status": "POOR", "issues": [f"corrupted/unreadable image ({type(e).__name__})"],
                "details": {}, "width": 0, "height": 0}

    details["size"] = f"{w}x{h}"
    if w < config.MIN_IMAGE_SIDE or h < config.MIN_IMAGE_SIDE:
        issues.append(f"too small ({w}x{h}, minimum {config.MIN_IMAGE_SIDE}px)")

    from PIL import Image as PILImage

    gray = np.asarray(PILImage.fromarray(rgb).convert("L"))
    mean_luma = float(gray.mean())
    details["mean_luma"] = round(mean_luma, 1)
    if mean_luma < config.DARK_MEAN:
        issues.append(f"extremely dark (mean luma {mean_luma:.0f} < {config.DARK_MEAN:.0f})")
    elif mean_luma > config.BRIGHT_MEAN:
        issues.append(f"extremely bright (mean luma {mean_luma:.0f} > {config.BRIGHT_MEAN:.0f})")

    try:
        import cv2

        side = config.QUALITY_REPORT_SIDE
        g = cv2.cvtColor(cv2.resize(rgb, (side, side)), cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
        details["laplacian_variance"] = round(lap_var, 1)
        if lap_var < config.BLUR_MIN_VARIANCE:
            issues.append(f"excessive blur (laplacian variance {lap_var:.1f} < {config.BLUR_MIN_VARIANCE:.1f})")
    except Exception as e:  # noqa: BLE001
        details["blur_check"] = f"skipped ({type(e).__name__})"

    return {
        "status": "POOR" if issues else "GOOD",
        "issues": issues,
        "details": details,
        "width": w,
        "height": h,
    }


def green_fraction(path: str) -> float:
    try:
        import cv2

        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            arr = np.asarray(im)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 40)).astype(np.uint8)
        return float(mask.mean())
    except Exception:
        return 0.0


# ---------------------------------------------------------------- plant detection
def detect_plant(path: str) -> dict:
    """Plant detection / validation.

    Primary: species classifier confidence (trained on the labelled species).
    Fallback (model missing): green-pixel fraction heuristic.
    """
    _, _, plant_model, plant_classes = load_bundle()
    if plant_model is not None:
        arr = config.load_image_rgb(path)[None, ...]
        probs = plant_model.predict(arr, verbose=0)[0]
        i = int(np.argmax(probs))
        conf = float(probs[i])
        return {
            "detected": conf >= config.PLANT_CONF_THRESHOLD,
            "method": "species_classifier",
            "species": plant_classes[i],
            "confidence": conf,
            "all": {plant_classes[j]: round(float(probs[j]), 4) for j in range(len(plant_classes))},
        }
    gf = green_fraction(path)
    return {
        "detected": gf >= 0.05,
        "method": "green_fraction_heuristic",
        "species": None,
        "confidence": round(gf, 4),
        "all": {"green_fraction": round(gf, 4)},
    }


# ---------------------------------------------------------------- core prediction
def predict_array(model, arr: np.ndarray, class_names):
    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1][: config.TOP_K]
    return probs, [(class_names[int(i)], float(probs[int(i)])) for i in order]


def predict_image(path: str, top_k: int = config.TOP_K) -> dict:
    """Full pipeline for one image -> structured prediction dict (spec section 25)."""
    started = time.time()
    result = {
        "image": path,
        "image_name": os.path.basename(path),
        "quality": None,
        "quality_issues": [],
        "quality_details": {},
        "plant_detected": False,
        "plant": None,
        "prediction": None,
        "confidence": None,
        "top_predictions": [],
        "health_status": None,
        "severity": None,          # dataset has no severity labels
        "status": None,
        "message": None,
        "elapsed_sec": None,
    }

    quality = check_image_quality(path)
    result["quality"] = quality["status"]
    result["quality_issues"] = quality["issues"]
    result["quality_details"] = quality["details"]
    if quality["status"] == "POOR":
        result["status"] = "IMAGE QUALITY: POOR"
        result["message"] = "Please upload a clearer, well-lit plant image."
        result["elapsed_sec"] = round(time.time() - started, 3)
        return result

    plant = detect_plant(path)
    result["plant_detected"] = bool(plant["detected"])
    result["plant"] = plant
    if not plant["detected"]:
        result["status"] = "NO PLANT DETECTED"
        result["message"] = ("This image does not look like one of the trained plant species. "
                             "Please upload a clear photo of a plant or leaf.")
        result["elapsed_sec"] = round(time.time() - started, 3)
        return result

    disease_model, class_names, _, _ = load_bundle()
    arr = config.load_image_rgb(path)[None, ...]
    probs, top = predict_array(disease_model, arr, class_names, )
    top = top[:top_k]
    best_cls, best_conf = top[0]

    result["prediction"] = best_cls
    result["confidence"] = round(best_conf, 6)
    result["top_predictions"] = [
        {"rank": i + 1, "class": c, "confidence": round(p, 6)} for i, (c, p) in enumerate(top)
    ]
    result["health_status"] = "HEALTHY" if "healthy" in best_cls.lower() else "DISEASE SUSPECTED"
    result["severity"] = None
    result["status"] = config.confidence_label(best_conf)
    if best_conf < config.MEDIUM_CONFIDENCE:
        result["status"] = "UNCERTAIN"
        result["message"] = config.UNCERTAIN_MESSAGE + " Upload a clearer image or close-up leaf image."
    elif result["health_status"] == "HEALTHY":
        result["message"] = "No disease symptoms detected in the top prediction."
    else:
        result["message"] = "Disease predicted from leaf imagery (see limitations in README)."
    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


# ---------------------------------------------------------------- reporting
def format_report(res: dict, show_header: bool = True) -> str:
    lines = []
    if show_header:
        lines += ["=" * 37,
                  "AI URBAN FARMING ML PREDICTION",
                  "=" * 37]
    lines.append(f"Image:            {res['image_name']}")
    if res["quality"] == "POOR":
        lines.append("Image quality:    POOR")
        for issue in res["quality_issues"]:
            lines.append(f"  - {issue}")
        lines.append("")
        lines.append("Recommendation:")
        lines.append(f"  {res['message']}")
        lines += ["=" * 37]
        return "\n".join(lines)

    lines.append("Image quality:    GOOD")
    lines.append(f"Plant detected:   {'YES' if res['plant_detected'] else 'NO'}")
    if res["plant"] and res["plant"].get("species"):
        lines.append(f"Plant identified: {res['plant']['species']} "
                     f"({res['plant']['confidence'] * 100:.2f}%)")
    if not res["plant_detected"]:
        lines.append("")
        lines.append(res["message"])
        lines += ["=" * 37]
        return "\n".join(lines)

    lines.append("")
    if res["prediction"]:
        lines.append(f"Prediction:       {res['prediction']}")
        lines.append(f"Confidence:       {res['confidence'] * 100:.2f}%")
        lines.append("")
        lines.append("Top 3:")
        for t in res["top_predictions"]:
            lines.append(f"  {t['rank']}. {t['class']} — {t['confidence'] * 100:.2f}%")
        lines.append("")
        lines.append(f"Health status:    {res['health_status']}")
        lines.append(f"Severity:         Not available")
        lines.append("")
        lines.append("Status:")
        lines.append(f"  {res['status']}")
        if res["message"]:
            lines.append(f"  {res['message']}")
    lines += ["=" * 37]
    return "\n".join(lines)


def structured_payload(res: dict) -> dict:
    """Exactly the structured payload described in the problem statement (section 25)."""
    return {
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "top_predictions": [
            {"class": t["class"], "confidence": t["confidence"]} for t in res["top_predictions"]
        ],
        "health_status": res["health_status"],
        "severity": None,
        "image": res["image_name"],
        "image_quality": res["quality"],
        "plant_detected": res["plant_detected"],
        "plant": res["plant"]["species"] if res.get("plant") else None,
        "status": res["status"],
        "message": res["message"],
    }


# ---------------------------------------------------------------- folder mode
def infer_actual_class(path: str, class_names) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent in class_names:
        return parent
    stem = os.path.splitext(os.path.basename(path))[0]
    for c in class_names:
        if c.lower() in stem.lower() or stem.lower() in c.lower():
            return c
    return ""


def run_folder(folder: str, csv_path: str = None) -> str:
    import csv

    class_names = load_bundle()[1]
    paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(config.SUPPORTED_EXTS):
                paths.append(os.path.join(root, f))
    paths.sort()
    if not paths:
        raise SystemExit(f"No supported images found in {folder}")

    csv_path = csv_path or config.CSV_OUTPUT_PATH
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for i, p in enumerate(paths, 1):
        res = predict_image(p)
        actual = infer_actual_class(p, class_names)
        pred = res["prediction"] or ""
        conf = res["confidence"]
        accepted = bool(res["plant_detected"]) and res["quality"] == "GOOD" and conf is not None
        correct = "" if (not actual or not accepted) else ("YES" if actual == pred else "NO")
        rows.append({
            "filename": os.path.relpath(p, folder),
            "actual_class": actual,
            "predicted_class": pred if accepted else f"REJECTED ({res['status']})",
            "confidence": "" if conf is None else f"{conf:.4f}",
            "correct": correct,
        })
        print(f"[{i}/{len(paths)}] {os.path.basename(p)} -> "
              f"{rows[-1]['predicted_class']} ({rows[-1]['confidence']}) correct={correct or '-'}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "actual_class", "predicted_class",
                                           "confidence", "correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} predictions -> {csv_path}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="AI Urban Farming ML prediction")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="path to a single plant image")
    group.add_argument("--folder", help="folder of images (batch prediction -> CSV)")
    ap.add_argument("--json", action="store_true", help="print structured JSON only")
    ap.add_argument("--csv", default=None, help="CSV output path for --folder")
    ap.add_argument("--top", type=int, default=config.TOP_K, help="number of top predictions")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.folder:
        run_folder(args.folder, args.csv)
        return

    if not os.path.isfile(args.image):
        raise SystemExit(f"FATAL: image not found: {args.image}")

    load_bundle()
    res = predict_image(args.image, top_k=args.top)
    if args.json:
        print(json.dumps(structured_payload(res), indent=2))
    else:
        print(format_report(res))


if __name__ == "__main__":
    main()
