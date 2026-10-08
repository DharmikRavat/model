"""Visual verification of predictions on held-out test images.

Run:  python visualize_predictions.py [--n 12]
Saves: outputs/sample_predictions.png
"""
import argparse
import json
import os

import numpy as np

import config


def load_test_items(limit_per_class=None):
    manifest = os.path.join(config.MODELS_DIR, "split_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        items = [(config.project_path(x["path"]), x["class"]) for x in m["test"]]
    else:
        split = config.build_split()
        items = [(p, c) for p, c in split["test"]]
    if limit_per_class:
        seen = {}
        keep = []
        for p, c in items:
            if seen.get(c, 0) < limit_per_class:
                keep.append((p, c))
                seen[c] = seen.get(c, 0) + 1
        items = keep
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of test images to display")
    args = ap.parse_args()

    config.ensure_dirs()
    import keras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isfile(config.MODEL_PATH):
        raise SystemExit(f"FATAL: model not found: {config.MODEL_PATH}. Run: python train.py")

    model = keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        class_names = json.load(fh)

    items = load_test_items(limit_per_class=2)  # ensure broad class coverage
    results = []
    for path, actual in items:
        try:
            x = config.load_image_rgb(path)[None, ...]
            probs = model.predict(x, verbose=0)[0]
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {path}: {e}")
            continue
        idx = int(np.argmax(probs))
        pred = class_names[idx]
        conf = float(probs[idx])
        results.append({"path": path, "actual": actual, "pred": pred, "conf": conf,
                        "correct": actual == pred})

    if not results:
        raise SystemExit("FATAL: no test images could be scored.")

    # prefer a mix: fill with incorrect examples first (so errors are visible), then correct
    results.sort(key=lambda r: (r["correct"], -r["conf"]))
    wrong = [r for r in results if not r["correct"]]
    right = [r for r in results if r["correct"]]
    selected = (wrong + right)[: args.n]
    selected.sort(key=lambda r: (r["actual"], r["pred"]))

    n_correct = sum(1 for r in results if r["correct"])
    print("=" * 37)
    print("VISUAL PREDICTION VERIFICATION")
    print("=" * 37)
    print(f"Scored {len(results)} test images | correct={n_correct} "
          f"({100 * n_correct / len(results):.1f}%)")
    print(f"Displayed {len(selected)} "
          f"(incorrect first so mistakes are visible)\n")

    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 5.1))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, selected):
        try:
            from PIL import Image

            with Image.open(r["path"]) as im:
                w, h = im.size
                im = im.convert("RGB")
                im.thumbnail((256, 256))
                ax.imshow(np.asarray(im))
        except Exception:
            ax.set_title("UNREADABLE", color="red")
            continue
        color = "#1b5e20" if r["correct"] else "#b71c1c"
        title = (f"Actual:    {r['actual']}\n"
                 f"Predicted: {r['pred']}\n"
                 f"Confidence: {r['conf'] * 100:.1f}%\n"
                 f"Correct:   {'YES' if r['correct'] else 'NO'}")
        ax.set_title(title, fontsize=8.5, color=color)
        ax.axis("off")

    fig.suptitle(f"Sample test predictions  ({n_correct}/{len(results)} correct "
                 f"on the full test split)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dest = os.path.join(config.OUTPUTS_DIR, "sample_predictions.png")
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    print(f"Saved -> {os.path.relpath(dest, config.PROJECT_DIR)}")

    print("\nDisplayed samples:")
    for r in selected:
        print(f"  {'OK ' if r['correct'] else 'ERR'} {os.path.basename(r['path'])}: "
              f"{r['actual']} -> {r['pred']} ({r['conf'] * 100:.1f}%)")
    print("=" * 37)


if __name__ == "__main__":
    main()
