"""Dataset inspection: counts, classes, corruption, sizes, imbalance, duplicates.

Run:  python inspect_dataset.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

import config

SUPPORTED = config.SUPPORTED_EXTS
REPORT_PATH = os.path.join(config.OUTPUTS_DIR, "dataset_report.txt")


def fmt_size(w, h):
    return f"{w}x{h}"


def inspect(dataset_dir=None):
    dataset_dir = dataset_dir or config.resolve_dataset_dir()
    lines = []
    out = lines.append

    out("DATASET REPORT")
    out("-" * 60)
    out(f"Dataset directory : {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        out("ERROR: dataset directory does not exist.")
        return "\n".join(lines)

    # ---- raw file scan
    class_dirs = sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)))
    unsupported = []
    all_files = []
    for d in class_dirs:
        for f in sorted(os.listdir(os.path.join(dataset_dir, d))):
            p = os.path.join(dataset_dir, d, f)
            if os.path.isfile(p):
                all_files.append(p)
                if not f.lower().endswith(SUPPORTED):
                    unsupported.append(os.path.join(d, f))

    # files directly in the dataset root (not inside a class folder)
    root_files = [
        f for f in sorted(os.listdir(dataset_dir))
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]

    total = len(all_files) + len(root_files)
    out(f"Total image files : {total}")
    out(f"Class folders     : {len(class_dirs)}")
    if root_files:
        out(f"Loose files in root (ignored): {len(root_files)} -> {root_files[:5]}")
    out("")

    # ---- per class scan
    corrupted, too_small, dims_counter = [], [], Counter()
    class_counts = {}
    unsupported_by_class = {}
    records = []  # (path, class, w, h, mean, std)

    for d in class_dirs:
        folder = os.path.join(dataset_dir, d)
        n_ok = 0
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if not os.path.isfile(p):
                continue
            if not f.lower().endswith(SUPPORTED):
                unsupported_by_class.setdefault(d, []).append(f)
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    arr = np.asarray(im)
            except Exception as e:  # noqa: BLE001
                corrupted.append(os.path.join(d, f) + f" ({type(e).__name__})")
                continue
            dims_counter[(w, h)] += 1
            if w < config.MIN_IMAGE_SIDE or h < config.MIN_IMAGE_SIDE:
                too_small.append(os.path.join(d, f) + f" ({fmt_size(w, h)})")
            records.append((p, d, w, h, float(arr.mean()), float(arr.std())))
            n_ok += 1
        class_counts[d] = n_ok

    out("Classes:")
    for k in sorted(class_counts):
        out(f"  {k}: {class_counts[k]}")
    out("")

    out(f"Corrupted         : {len(corrupted)}")
    for c in corrupted[:10]:
        out(f"  - {c}")
    out(f"Unsupported ext   : {len(unsupported) + sum(len(v) for v in unsupported_by_class.values())}")
    for u in (unsupported + [f"{k}/{x}" for k, v in unsupported_by_class.items() for x in v])[:10]:
        out(f"  - {u}")
    out(f"Very small (<{config.MIN_IMAGE_SIDE}px): {len(too_small)}")
    for s in too_small[:10]:
        out(f"  - {s}")
    out("")

    # ---- dimensions
    out("Image dimensions (top 8):")
    for (w, h), n in dims_counter.most_common(8):
        out(f"  {fmt_size(w, h)}: {n}")
    if records:
        sizes = np.array([[r[2], r[3]] for r in records])
        out(f"  min={sizes.min(axis=0).tolist()} max={sizes.max(axis=0).tolist()} "
            f"mean={sizes.mean(axis=0).round(1).tolist()}")
    out("")

    # ---- class imbalance
    counts = np.array(list(class_counts.values())) if class_counts else np.array([0])
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    imb = ratio > config.IMBALANCE_RATIO_FLAG
    out(f"Class imbalance   : {'detected' if imb else 'not detected'} "
        f"(max/min ratio = {ratio:.2f}, flag threshold = {config.IMBALANCE_RATIO_FLAG})")
    if imb:
        out(f"  largest class : {max(class_counts, key=class_counts.get)} ({counts.max()})")
        out(f"  smallest class: {min(class_counts, key=class_counts.get)} ({counts.min()})")
    out("")

    # ---- brightness sanity
    dark = [os.path.relpath(r[0], dataset_dir) for r in records if r[4] < config.DARK_MEAN]
    bright = [os.path.relpath(r[0], dataset_dir) for r in records if r[4] > config.BRIGHT_MEAN]
    out(f"Extremely dark    : {len(dark)}")
    out(f"Extremely bright  : {len(bright)}")
    out("")

    # ---- duplicates (exact + near)
    exact, seen = [], {}
    for p, *_ in records:
        try:
            import hashlib

            with open(p, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen:
                exact.append((seen[h], os.path.relpath(p, dataset_dir)))
            else:
                seen[h] = os.path.relpath(p, dataset_dir)
        except Exception:
            pass
    out(f"Exact duplicates  : {len(exact)}")
    for a, b in exact[:5]:
        out(f"  - {a} == {b}")

    near_total = 0
    near_examples = []
    for cls in sorted(class_counts):
        cls_paths = [r[0] for r in records if r[1] == cls]
        try:
            groups = config.group_near_duplicates(cls_paths)
        except Exception:
            groups = [[p] for p in cls_paths]
        for g in groups:
            if len(g) > 1:
                near_total += len(g) - 1
                if len(near_examples) < 5:
                    near_examples.append([os.path.relpath(x, dataset_dir) for x in g[:3]])
    out(f"Near-duplicates   : {near_total} (hamming<={config.DUP_HAMMING_THRESHOLD} on 64-bit dHash)")
    for g in near_examples:
        out(f"  - {g}")
    out("")

    usable = len(records)
    out(f"Usable images     : {usable}")
    out(f"Recommendation    : "
        + ("OK to train." if len(class_dirs) >= 2 and len(records) >= 100
           else "DATASET INVALID - do not continue (too few classes/images)."))
    out("-" * 60)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(config.MODELS_DIR, "dataset_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_dir": config.relative_project_path(dataset_dir),
                "total_files": total,
                "classes": class_counts,
                "corrupted": corrupted,
                "unsupported": unsupported + [f"{k}/{x}" for k, v in unsupported_by_class.items() for x in v],
                "too_small": too_small,
                "dimensions": {f"{w}x{h}": n for (w, h), n in dims_counter.items()},
                "class_imbalance_detected": bool(imb),
                "imbalance_ratio": ratio,
                "exact_duplicates": len(exact),
                "near_duplicates": near_total,
            },
            fh,
            indent=2,
        )

    if len(class_dirs) < 2 or len(records) < 100:
        print("\n".join(lines))
        sys.exit("FATAL: dataset invalid - refusing to continue.")
    return "\n".join(lines)


if __name__ == "__main__":
    config.ensure_dirs()
    print(inspect())
