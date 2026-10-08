"""Datasets for EfficientNet-B2 training.

Sources
  1. PlantVillage subset in ./dataset            -> existing 70/15/15 split (models/split_manifest.json)
  2. PlantDoc train (real photos)                -> appended to the training set only
  3. PlantDoc test  (real photos, never trained) -> real_world test set
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config

PLANTDOC_DIR = os.path.join(config.PROJECT_DIR, "dataset_plantdoc")
IMAGE_SIZE = 260
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- domain randomisation
COMPOSITE_P = 0.55          # probability of pasting the leaf onto a real photo background
_BG_POOL: List[str] = []


def background_pool() -> List[str]:
    """Real photos from PlantDoc-train (never the held-out test split) used as paste targets."""
    global _BG_POOL
    if not _BG_POOL and os.path.isdir(PLANTDOC_DIR):
        root = os.path.join(PLANTDOC_DIR, "train")
        for r, _, files in os.walk(root):
            for f in files:
                if config.is_supported_image(f):
                    _BG_POOL.append(os.path.join(r, f))
    return _BG_POOL


def _leaf_mask(arr: np.ndarray) -> Optional[np.ndarray]:
    """PlantVillage frames sit on a flat grey background -> colour distance from the border."""
    border = np.concatenate([arr[0:3].reshape(-1, 3), arr[-3:].reshape(-1, 3),
                             arr[:, 0:3].reshape(-1, 3), arr[:, -3:].reshape(-1, 3)])
    bg = np.median(border, 0).astype(np.float32)
    a = arr if arr.dtype == np.float32 else arr.astype(np.float32)
    dist = np.sqrt(((a - bg) ** 2).sum(-1))
    return dist > 28.0


def composite_background(img):
    """Paste the segmented leaf onto a random real photo (kills the flat-background shortcut)."""
    from PIL import Image, ImageFilter

    pool = background_pool()
    if not pool or random.random() > COMPOSITE_P:
        return img
    # PlantDoc originals can be 2592x3872; a 1024-px canvas is plenty before the 260 crop
    big = max(img.size)
    if big > 1024:
        r = 1024.0 / big
        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.BILINEAR)
    arr = np.asarray(img)
    mask = _leaf_mask(arr)
    if not mask.any() or mask.mean() > 0.97 or mask.mean() < 0.20:
        return img

    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    leaf = img.crop((x0, y0, x1, y1))
    mask_im = Image.fromarray((mask * 255).astype(np.uint8)).crop((x0, y0, x1, y1))
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(1.2))

    w, h = img.size
    try:
        bg = Image.open(pool[random.randrange(len(pool))]).convert("RGB")
    except Exception:  # noqa: BLE001
        return img
    # cover-crop the background to the canvas
    scale = max(w / bg.width, h / bg.height)
    bg = bg.resize((max(w, int(bg.width * scale + 1)), max(h, int(bg.height * scale + 1))),
                   Image.BILINEAR)
    ox = random.randint(0, bg.width - w)
    oy = random.randint(0, bg.height - h)
    bg = bg.crop((ox, oy, ox + w, oy + h))

    s = random.uniform(0.72, 1.12)
    lw, lh = max(4, int(leaf.width * s)), max(4, int(leaf.height * s))
    leaf = leaf.resize((lw, lh), Image.BILINEAR)
    mask_im = mask_im.resize((lw, lh), Image.BILINEAR)
    px = random.randint(-int(lw * 0.15), w - int(lw * 0.85))
    py = random.randint(-int(lh * 0.15), h - int(lh * 0.85))
    bg.paste(leaf, (px, py), mask_im)
    return bg


class ChannelStyleJitter:
    """Camera white-balance / exposure shift applied on the normalised tensor."""

    def __init__(self, gain=(0.85, 1.15), bias=(-0.06, 0.06), p=0.5):
        self.gain, self.bias, self.p = gain, bias, p

    def __call__(self, t):
        import torch

        if random.random() > self.p:
            return t
        g = t.new_empty(3, 1, 1).uniform_(*self.gain)
        b = t.new_empty(3, 1, 1).uniform_(*self.bias)
        return t * g + b


def class_names() -> List[str]:
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        return [*json.load(fh), config.OTHER_PLANT_CLASS]


def load_split() -> Dict[str, List[Tuple[str, str]]]:
    """PlantVillage split straight from the manifest."""
    with open(os.path.join(config.MODELS_DIR, "split_manifest.json"), "r", encoding="utf-8") as fh:
        m = json.load(fh)
    out = {"train": [], "val": [], "test": []}
    for key in out:
        for e in m[key]:
            path = config.project_path(e["path"])
            if os.path.isfile(path):
                out[key].append((path, e["class"]))
    return out


def plantdoc(split: str, allowed: List[str]) -> List[Tuple[str, str]]:
    root = os.path.join(PLANTDOC_DIR, split)
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return out
    for cls in sorted(os.listdir(root)):
        if cls not in allowed:
            continue
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if config.is_supported_image(f):
                out.append((os.path.join(d, f), cls))
    return out


def plantdoc_other(split: str) -> List[Tuple[str, str]]:
    """PlantDoc crops outside the supported disease classes, used as reject examples."""
    root = os.path.join(PLANTDOC_DIR, split, "other_plant")
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return out
    for r, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            if config.is_supported_image(name):
                out.append((os.path.join(r, name), config.OTHER_PLANT_CLASS))
    return out


def build_sets(oversample_real: float = 2.0) -> Dict[str, List[Tuple[str, str]]]:
    """train = PlantVillage-train + PlantDoc-train + PlantDoc other-plant negatives ;
    val        = PlantVillage-val  (clean images)
    val_real   = PlantDoc-train slice (real photos, used only for model selection)
    test       = PlantVillage-test
    val_ood    = PlantDoc-train other-plant slice (used only for model selection)
    real_world_test = PlantDoc-test plus other-plant test (never trained)"""
    allowed = class_names()
    pv = load_split()
    real_train = plantdoc("train", allowed)
    # deterministic 1/6 of the real photos held out for validation
    real_val = [x for i, x in enumerate(real_train) if i % 6 == 0]
    real_train = [x for i, x in enumerate(real_train) if i % 6 != 0]
    real_train = real_train * max(1, int(oversample_real))     # real photos are rare -> up-weight
    other_images = plantdoc_other("train")
    other_val = [x for i, x in enumerate(other_images) if i % 6 == 0]
    other_train = [x for i, x in enumerate(other_images) if i % 6 != 0]
    sets = {
        "train": pv["train"] + real_train + other_train,
        "val": pv["val"],
        "val_real": real_val,
        "val_ood": other_val,
        "test": pv["test"],
        "real_world_test": plantdoc("test", allowed) + plantdoc_other("test"),
    }
    return sets


# ---------------------------------------------------------------- transforms
def train_transform():
    return transforms.Compose([
        transforms.Lambda(composite_background),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.75, 1.35)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=8),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6))], p=0.25),
        transforms.RandomGrayscale(p=0.06),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ChannelStyleJitter(),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.15)),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_transform():
    return eval_transform()


# ---------------------------------------------------------------- dataset
class LeafDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], class_to_idx: Dict[str, int], tfm):
        self.items = items
        self.c2i = class_to_idx
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        path, cls = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.tfm(img), self.c2i[cls], path


def class_to_idx() -> Dict[str, int]:
    return {c: i for i, c in enumerate(class_names())}


def make_loaders(batch_size: int = 32, num_workers: int = None):
    sets = build_sets()
    c2i = class_to_idx()
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = int(os.environ.get("LOADER_WORKERS", "4"))
    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)

    train_ds = LeafDataset(sets["train"], c2i, train_transform())
    val_ds = LeafDataset(sets["val"], c2i, eval_transform())
    val_real_ds = LeafDataset(sets["val_real"], c2i, eval_transform())
    val_ood_ds = LeafDataset(sets["val_ood"], c2i, eval_transform())
    test_ds = LeafDataset(sets["test"], c2i, eval_transform())
    real_ds = LeafDataset(sets["real_world_test"], c2i, eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    val_real_loader = DataLoader(val_real_ds, batch_size=batch_size, shuffle=False, **kw)
    val_ood_loader = DataLoader(val_ood_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, val_real_loader, val_ood_loader, test_loader, real_loader, sets


def label_counts(items: List[Tuple[str, str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, c in items:
        out[c] = out.get(c, 0) + 1
    return out


if __name__ == "__main__":
    sets = build_sets()
    for k, v in sets.items():
        print(f"{k:16s} {len(v):5d}")
    print("\ntrain class counts:")
    for c, n in sorted(label_counts(sets["train"]).items()):
        print(f"  {c:35s} {n}")
