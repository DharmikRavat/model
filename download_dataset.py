"""Fetch a real PlantVillage subset from Hugging Face into ./dataset/<Class>/.

This is a data-acquisition helper (not part of the trained model).
The training pipeline discovers classes dynamically from ./dataset/.
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

REPO = "leoho36/plant_village_dataset"
API = "https://huggingface.co/api/datasets/" + REPO + "/tree/main/"
RAW = "https://huggingface.co/datasets/" + REPO + "/resolve/main/"
HDR = {"User-Agent": "Mozilla/5.0 (compatible; plant-ml/1.0)"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
SEED = 42
CAP_PER_CLASS = 500
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
]

UNKNOWN_IMAGES = {
    "unknown_dog.jpg": "Dog",
    "unknown_car.jpg": "Car",
    "unknown_building.jpg": "Building",
    "unknown_person.jpg": "Person",
    "unknown_object.jpg": "Rhinoceros_beetle",
}


def get(url, tries=4, timeout=90):
    last = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url} :: {last}")


def list_class(cls):
    url = API + urllib.parse.quote("color/" + cls, safe="/")
    data = json.loads(get(url))
    return [f["path"] for f in data if f.get("type") == "file" and f["path"].lower().endswith(EXTS)]


def download_one(path, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    try:
        blob = get(RAW + urllib.parse.quote(path, safe="/"))
    except Exception as e:  # noqa: BLE001
        return f"fail {e}"
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(blob)
    os.replace(tmp, dest)
    return "ok"


def fetch_classes():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for cls in CLASSES:
        try:
            files = list_class(cls)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {cls}: cannot list ({e})", flush=True)
            continue
        files.sort()
        if len(files) > CAP_PER_CLASS:
            files = sorted(rng.sample(files, CAP_PER_CLASS))
        target_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(target_dir, exist_ok=True)
        jobs = []
        for p in files:
            name = p.split("/")[-1]
            ext = os.path.splitext(name)[1]
            if ext not in EXTS:
                ext = ".jpg"
            jobs.append((p, os.path.join(target_dir, name)))
        t0 = time.time()
        ok = fail = 0
        workers = int(os.environ.get("DL_WORKERS", "24"))
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(download_one, p, d) for p, d in jobs]
            for i, f in enumerate(cf.as_completed(futs), 1):
                r = f.result()
                if r == "ok" or r == "skip":
                    ok += 1
                else:
                    fail += 1
                    print(f"  {r}", flush=True)
                if i % 200 == 0:
                    print(f"  {cls}: {i}/{len(jobs)} ({i / (time.time() - t0):.1f}/s)", flush=True)
        total += ok
        print(f"[DONE] {cls}: {ok} files, {fail} failed, {time.time() - t0:.1f}s", flush=True)
    print(f"TOTAL downloaded/verified: {total}", flush=True)


def fetch_unknown_images():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(out, exist_ok=True)
    for fname, article in UNKNOWN_IMAGES.items():
        dest = os.path.join(out, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[SKIP] {fname}", flush=True)
            continue
        try:
            meta = json.loads(get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}", timeout=60))
            url = (meta.get("originalimage") or meta.get("thumbnail") or {}).get("source")
            if not url:
                print(f"[FAIL] {fname}: no image url", flush=True)
                continue
            blob = get(url, timeout=120)
            with open(dest, "wb") as fh:
                fh.write(blob)
            print(f"[OK] {fname} <- {url[:90]} ({len(blob)} bytes)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {fname}: {e}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "dataset"):
        fetch_classes()
    if what in ("all", "unknown"):
        fetch_unknown_images()
    print("download finished", flush=True)
