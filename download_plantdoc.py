"""Download the PlantDoc real-world leaf dataset (GitHub: pratikkayal/PlantDoc-Dataset).

Outputs (kept separate from the PlantVillage ./dataset):
  dataset_plantdoc/train/<mapped_class>/*.jpg   -> mixed into training
  dataset_plantdoc/test/<mapped_class>/*.jpg    -> held-out REAL-WORLD test set
  dataset_plantdoc/other_plant/<class>/*.jpg    -> plant species we do not classify

Run:  python download_plantdoc.py
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.parse
import urllib.request

REPO = "pratikkayal/PlantDoc-Dataset"
BRANCH = "master"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
HDR = {"User-Agent": "plant-ml/1.0"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset_plantdoc")

# PlantDoc folder name -> our training class name ("" = keep as other_plant)
CLASS_MAP = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "OTHER_Tomato_bacterial_spot",
    "Tomato leaf mosaic virus": "OTHER_Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "OTHER_Tomato_yellow_virus",
    "Tomato two spotted spider mites leaf": "OTHER_Tomato_spider_mites",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
}
EXTS = (".jpg", ".jpeg", ".png")


def get(url, tries=4, timeout=90):
    last = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 + i)
    raise RuntimeError(f"failed after {tries} tries: {url} :: {last}")


def tree_paths():
    data = json.loads(get(TREE))
    if data.get("truncated"):
        print("[WARN] github tree truncated", flush=True)
    out = []
    for x in data["tree"]:
        p = x["path"]
        if x["type"] == "blob" and p.lower().endswith(EXTS) and p.startswith(("train/", "test/")):
            out.append(p)
    return out


def sanitize(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad or ord(ch) < 32 else ch for ch in name)
    return out[:150] or "image.jpg"


def download_one(path, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return "skip"
    try:
        blob = get(RAW + urllib.parse.quote(urllib.parse.unquote(path), safe="/"), timeout=120)
    except Exception as e:  # noqa: BLE001
        return f"fail {e}"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, dest)
    except Exception as e:  # noqa: BLE001
        return f"fail write {e}"
    return "ok"


def main():
    paths = tree_paths()
    jobs = []
    skipped = 0
    for p in paths:
        parts = p.split("/")
        if len(parts) != 3:
            skipped += 1
            continue
        split, cls, fname = parts
        cls = sanitize(urllib.parse.unquote(cls))
        fname = sanitize(urllib.parse.unquote(fname))
        mapped = CLASS_MAP.get(cls)
        if mapped and mapped.startswith("OTHER_"):
            target = os.path.join(OUT, split, "other_plant", cls, fname)
        elif mapped:
            target = os.path.join(OUT, split, mapped, fname)
        elif cls.startswith(("Apple", "Corn", "grape", "Peach", "Cherry",
                             "Blueberry", "Soyabean", "Raspberry", "Strawberry", "Squash")):
            target = os.path.join(OUT, split, "other_plant", cls, fname)
        else:
            skipped += 1
            continue
        jobs.append((p, target))

    print(f"PlantDoc: {len(paths)} images found, {len(jobs)} to fetch, {skipped} unmapped skipped", flush=True)
    t0 = time.time()
    ok = fail = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(cf.as_completed([ex.submit(download_one, p, d) for p, d in jobs]), 1):
            res = r.result()
            if res in ("ok", "skip"):
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  {res}", flush=True)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} ok={ok} fail={fail} ({i / (time.time() - t0):.1f}/s)", flush=True)
    print(f"DONE ok={ok} fail={fail} in {time.time() - t0:.0f}s -> {OUT}", flush=True)

    # summary
    summary = {}
    for root, _, files in os.walk(OUT):
        n = len([f for f in files if f.lower().endswith(EXTS)])
        if n:
            summary[os.path.relpath(root, OUT)] = n
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    sys.exit(main())
