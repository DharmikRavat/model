"""Central configuration + shared helpers for the plant ML pipeline.

All tunable thresholds live here (never hard-code them in other files).
"""
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------- paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
TEST_IMAGES_DIR = os.path.join(PROJECT_DIR, "test_images")

FALLBACK_DATASET_DIRS = []

MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
PLANT_MODEL_PATH = os.path.join(MODELS_DIR, "plant_classifier.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
PLANT_CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "plant_class_names.json")
MODEL_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
OTHER_PLANT_CLASS = "__other_plant__"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- image
IMAGE_SIZE = 224
CHANNELS = 3
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ---------------------------------------------------------------- data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# near-duplicate grouping (data-leakage prevention): max hamming distance of 64-bit dHash
DUP_HAMMING_THRESHOLD = 8

# ---------------------------------------------------------------- model / training
BACKBONE = "EfficientNetB0"          # primary
FALLBACK_BACKBONE = "MobileNetV2"    # used only if EfficientNet weights are unavailable
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15                   # backbone frozen
EPOCHS_STAGE2 = 15                   # upper layers fine-tuned
LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5
OPTIMIZER = "Adam"
FINE_TUNE_LAYERS = 30                # number of top backbone layers unfrozen in stage 2
DROPOUT_1 = 0.30
DROPOUT_2 = 0.30
DENSE_UNITS = 256

# callbacks
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# class imbalance
USE_CLASS_WEIGHTS = True
IMBALANCE_RATIO_FLAG = 1.5           # max/min class count above this => "imbalance detected"

# ---------------------------------------------------------------- confidence thresholds
HIGH_CONFIDENCE = 0.80               # >= 0.80  -> HIGH CONFIDENCE
MEDIUM_CONFIDENCE = 0.60             # 0.60-0.79 -> MEDIUM CONFIDENCE
                                     # <  0.60  -> LOW CONFIDENCE / UNCERTAIN
PLANT_CONF_THRESHOLD = 0.50          # plant-detector confidence needed to accept an image
UNCERTAIN_MESSAGE = "UNCERTAIN - PLEASE UPLOAD A CLEARER OR CLOSE-UP IMAGE."

# ---------------------------------------------------------------- image quality
MIN_IMAGE_SIDE = 64                  # smaller than this = too small
DARK_MEAN = 30.0                     # mean luma below this = extremely dark
BRIGHT_MEAN = 230.0                  # mean luma above this = extremely bright
BLUR_MIN_VARIANCE = 40.0             # variance of Laplacian below this = excessive blur
QUALITY_REPORT_SIDE = 256            # quality metrics computed on a 256px rescale

# ---------------------------------------------------------------- misc
TOP_K = 3
CSV_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "predictions.csv")


# ================================================================ helpers
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dirs() -> None:
    for d in (DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR):
        os.makedirs(d, exist_ok=True)


def resolve_dataset_dir() -> str:
    """Return the first directory that actually contains class sub-folders."""
    candidates = [DATASET_DIR]
    candidates += [d for d in FALLBACK_DATASET_DIRS if d not in candidates]
    for c in candidates:
        if os.path.isdir(c) and discover_classes(c):
            return c
    return DATASET_DIR


def is_supported_image(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_EXTS)


def discover_classes(dataset_dir: str) -> Dict[str, List[str]]:
    """Dynamically discover classes = sub-directories holding supported images.

    Never assumes class names. Returns {class_name: [sorted image paths]}.
    """
    classes: Dict[str, List[str]] = {}
    if not os.path.isdir(dataset_dir):
        return classes
    for entry in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, entry)
        if not os.path.isdir(full):
            continue
        files = sorted(
            os.path.join(full, f)
            for f in os.listdir(full)
            if is_supported_image(f) and os.path.isfile(os.path.join(full, f))
        )
        if files:
            classes[entry] = files
    return classes


def species_from_class(class_name: str) -> str:
    """Tomato___Early_blight -> Tomato ; Pepper,_bell___healthy -> Pepper,_bell."""
    return class_name.split("___")[0]


def dhash(path: str, hash_size: int = 8) -> int:
    from PIL import Image

    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(g, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = np.packbits(diff.astype(np.uint8).flatten())
    return int("".join(format(b, "08b") for b in bits), 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_near_duplicates(paths: List[str], threshold: int = DUP_HAMMING_THRESHOLD) -> List[List[str]]:
    """Union-find grouping of near-identical images (practical duplicate detection)."""
    hashes = []
    for p in paths:
        try:
            hashes.append(dhash(p))
        except Exception:
            hashes.append(hash(p) & ((1 << 64) - 1))  # unreadable -> unique group
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if _hamming(hashes[i], hashes[j]) <= threshold:
                union(i, j)
    groups: Dict[int, List[str]] = {}
    for i, p in enumerate(paths):
        groups.setdefault(find(i), []).append(p)
    return list(groups.values())


def build_split(dataset_dir: str = None, seed: int = SEED) -> Dict[str, object]:
    """Deterministic, stratified, duplicate-group-aware 70/15/15 split.

    Related (near-duplicate) images always land in the same split so the test
    set can never leak into training.
    """
    dataset_dir = dataset_dir or resolve_dataset_dir()
    classes = discover_classes(dataset_dir)
    if not classes:
        raise FileNotFoundError(f"No class folders with images found under: {dataset_dir}")

    rng = random.Random(seed)
    train, val, test = [], [], []
    per_class_stats = {}

    for cls in sorted(classes):
        paths = classes[cls]
        clusters = group_near_duplicates(paths)
        rng.shuffle(clusters)
        n = len(paths)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        # deterministic cluster order: shuffled then largest-first
        order = sorted(clusters, key=len, reverse=True)
        c_train: List[str] = []
        c_val: List[str] = []
        c_test: List[str] = []
        for c in order:
            if len(c_train) < n_train:
                c_train.extend(c)
            elif len(c_val) < n_val:
                c_val.extend(c)
            else:
                c_test.extend(c)
        # guarantee non-empty val/test whenever enough clusters exist
        if len(order) >= 3 and not c_test:
            move = c_train[-len(order[-1]):]
            c_train = c_train[: len(c_train) - len(move)]
            c_test = move
        if len(order) >= 2 and not c_val:
            move = c_train[-len(order[-1]):]
            c_train = c_train[: len(c_train) - len(move)]
            c_val = move
        for p in c_train:
            train.append((p, cls))
        for p in c_val:
            val.append((p, cls))
        for p in c_test:
            test.append((p, cls))
        per_class_stats[cls] = {
            "total": n,
            "train": len(c_train),
            "val": len(c_val),
            "test": len(c_test),
            "duplicate_groups": len(clusters),
        }

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {
        "dataset_dir": dataset_dir,
        "class_names": sorted(classes),
        "train": train,
        "val": val,
        "test": test,
        "per_class": per_class_stats,
    }


def load_split_json(path: str = None) -> Dict[str, object]:
    path = path or os.path.join(MODELS_DIR, "split_manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def project_path(path: str) -> str:
    """Resolve a manifest path against this checkout while accepting old absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def relative_project_path(path: str) -> str:
    """Store paths relative to the project so manifests work after cloning."""
    return os.path.relpath(os.path.abspath(path), PROJECT_DIR).replace(os.sep, "/")


def save_split_json(split: Dict[str, object], path: str = None) -> str:
    path = path or os.path.join(MODELS_DIR, "split_manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "dataset_dir": relative_project_path(split["dataset_dir"]),
        "class_names": split["class_names"],
        "train": [{"path": relative_project_path(p), "class": c} for p, c in split["train"]],
        "val": [{"path": relative_project_path(p), "class": c} for p, c in split["val"]],
        "test": [{"path": relative_project_path(p), "class": c} for p, c in split["test"]],
        "per_class": split["per_class"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_image_rgb(path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load image as RGB float32 array of shape (size, size, 3), values 0..255."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr


def confidence_label(conf: float) -> str:
    if conf >= HIGH_CONFIDENCE:
        return "HIGH CONFIDENCE"
    if conf >= MEDIUM_CONFIDENCE:
        return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"
