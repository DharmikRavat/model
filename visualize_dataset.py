"""Visual dataset inspection: per-class sample grids + class distribution.

Run:  python visualize_dataset.py
Outputs:
  outputs/dataset_samples/<class>.png
  outputs/dataset_distribution.png
Also prints an explicit limitation report when the data is leaf/close-up imagery.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config

SAMPLES_PER_CLASS = 12
GRID_COLS = 4
BORDER = 12  # px border used for background-uniformity heuristic


def background_uniformity(path: str) -> float:
    """Low std of the image border => plain background => likely a single leaf shot."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            arr = np.asarray(im, dtype=np.float32)
    except Exception:
        return 255.0
    b = BORDER
    border = np.concatenate(
        [arr[:b, :, :].reshape(-1, 3), arr[-b:, :, :].reshape(-1, 3),
         arr[:, :b, :].reshape(-1, 3), arr[:, -b:, :].reshape(-1, 3)]
    )
    return float(border.std())


def leaf_dataset_heuristic(classes) -> dict:
    """Decide whether the dataset is mostly isolated-leaf shots on plain backgrounds."""
    stats = []
    for cls, paths in classes.items():
        sample = paths[: min(30, len(paths))]
        vals = [background_uniformity(p) for p in sample]
        stats.append((cls, float(np.median(vals)), len(sample)))
    medians = [s[1] for s in stats]
    leaf_like = sum(1 for m in medians if m < 45)
    return {
        "per_class_median_border_std": {c: round(m, 1) for c, m, _ in stats},
        "leaf_like_classes": leaf_like,
        "total_classes": len(stats),
        "is_primarily_leaf": leaf_like >= max(1, int(0.7 * len(stats))),
    }


def save_class_grid(cls: str, paths, out_dir: str) -> str:
    n = min(SAMPLES_PER_CLASS, len(paths))
    cols = GRID_COLS
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        ax = axes[i]
        try:
            with Image.open(paths[i]) as im:
                w, h = im.size
                im = im.convert("RGB")
                im.thumbnail((224, 224))
                ax.imshow(np.asarray(im))
        except Exception:
            ax.set_title("UNREADABLE", fontsize=8, color="red")
            continue
        ax.axis("off")
        ax.set_title(f"{os.path.basename(paths[i])[:26]}\n{w}x{h}", fontsize=7)
    fig.suptitle(f"{cls}  (n={len(paths)})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{cls.replace(',', '')}.png")
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def save_distribution(classes) -> str:
    names = sorted(classes)
    counts = [len(classes[n]) for n in names]
    order = np.argsort(counts)[::-1]
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2e7d32" if "healthy" in n.lower() else "#c62828" for n in names]
    ax.bar(range(len(names)), counts, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("number of images")
    ax.set_title(f"Class distribution (total = {sum(counts)} images, {len(names)} classes)")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    dest = os.path.join(config.OUTPUTS_DIR, "dataset_distribution.png")
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def main():
    config.ensure_dirs()
    dataset_dir = config.resolve_dataset_dir()
    classes = config.discover_classes(dataset_dir)
    if not classes:
        raise SystemExit(f"FATAL: no classes found under {dataset_dir}")

    print("=" * 60)
    print("DATASET VISUAL INSPECTION")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")

    samples_dir = os.path.join(config.OUTPUTS_DIR, "dataset_samples")
    for cls in sorted(classes):
        dest = save_class_grid(cls, classes[cls], samples_dir)
        print(f"  grid -> {os.path.relpath(dest, config.PROJECT_DIR)}  (n={len(classes[cls])})")

    dist = save_distribution(classes)
    print(f"  distribution -> {os.path.relpath(dist, config.PROJECT_DIR)}")

    report = leaf_dataset_heuristic(classes)
    print("")
    print("Background / framing heuristic (border pixel std, lower = plain background):")
    for cls, v in report["per_class_median_border_std"].items():
        print(f"  {cls}: {v}")
    print("")
    if report["is_primarily_leaf"]:
        print("DATASET LIMITATION: This dataset contains primarily leaf images "
              "and may not generalize to complete-plant photographs.")
        print("The disease classifier is trained primarily on leaf imagery and may have "
              "reduced performance on complete-plant photographs.")
    else:
        print("Framing check: dataset is not predominantly plain-background leaf shots "
              "(still verify samples visually).")
    print("=" * 60)


if __name__ == "__main__":
    main()
