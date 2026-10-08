"""Model persistence test: save -> clear session -> reload -> predict -> compare.

Run:  python test_reload.py
The prediction before and after reload must match within numerical tolerance.
"""
import json
import os

import numpy as np

import config

TOLERANCE = 1e-5


def sample_images(n: int = 4):
    manifest = os.path.join(config.MODELS_DIR, "split_manifest.json")
    paths = []
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        paths = [config.project_path(x["path"]) for x in m["test"][:n]]
    if not paths:
        classes = config.discover_classes(config.resolve_dataset_dir())
        paths = [v[0] for v in list(classes.values())[:n]]
    # include one non-plant image if available
    if os.path.isdir(config.TEST_IMAGES_DIR):
        for f in sorted(os.listdir(config.TEST_IMAGES_DIR)):
            if f.lower().endswith(config.SUPPORTED_EXTS):
                paths.append(os.path.join(config.TEST_IMAGES_DIR, f))
                break
    return paths[: max(n, 4)]


def main():
    import keras

    config.ensure_dirs()
    if not os.path.isfile(config.MODEL_PATH):
        raise SystemExit(f"FATAL: model not found: {config.MODEL_PATH}. Run: python train.py")

    print("=" * 37)
    print("MODEL RELOAD TEST")
    print("=" * 37)

    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        class_names = json.load(fh)

    paths = sample_images()
    print(f"Test images: {len(paths)}")
    for p in paths:
        print(f"  - {os.path.basename(p)}")

    # ---- step 1: model already saved by train.py; re-save a copy from the in-memory model
    model = keras.models.load_model(config.MODEL_PATH)
    reload_copy = os.path.join(config.MODELS_DIR, "reload_check.keras")
    model.save(reload_copy)
    print(f"\n[1] Saved in-memory model -> {os.path.relpath(reload_copy, config.PROJECT_DIR)}")

    batch = np.stack([config.load_image_rgb(p) for p in paths])
    probs_before = model.predict(batch, verbose=0)
    print("[2] Predictions computed with the in-memory model")

    # ---- step 3: clear / restart model object, then reload from disk
    del model
    keras.backend.clear_session()
    print("[3] Session cleared (model object destroyed)")

    model_a = keras.models.load_model(reload_copy)   # the freshly saved copy
    probs_copy = model_a.predict(batch, verbose=0)

    del model_a
    keras.backend.clear_session()
    model_b = keras.models.load_model(config.MODEL_PATH)  # the original artefact
    probs_orig = model_b.predict(batch, verbose=0)
    print("[4] Model reloaded from disk (twice: copy + original)")

    # ---- step 5: compare
    max_diff_copy = float(np.max(np.abs(probs_before - probs_copy)))
    max_diff_orig = float(np.max(np.abs(probs_before - probs_orig)))
    argmax_ok = bool(np.array_equal(probs_before.argmax(1), probs_copy.argmax(1)) and
                     np.array_equal(probs_before.argmax(1), probs_orig.argmax(1)))

    print("")
    print(f"{'image':40s} {'before':>10s} {'after(copy)':>12s} {'after(orig)':>12s}  match")
    for i, p in enumerate(paths):
        b = int(probs_before[i].argmax())
        c = int(probs_copy[i].argmax())
        o = int(probs_orig[i].argmax())
        match = (b == c == o)
        print(f"{os.path.basename(p)[:40]:40s} {class_names[b][:10]:>10s} "
              f"{class_names[c][:12]:>12s} {class_names[o][:12]:>12s}  {'YES' if match else 'NO'}")

    print("")
    print(f"Max |diff| (saved copy)   : {max_diff_copy:.3e}")
    print(f"Max |diff| (original file): {max_diff_orig:.3e}")
    print(f"Tolerance                 : {TOLERANCE:.0e}")
    print(f"Argmax identical          : {'YES' if argmax_ok else 'NO'}")

    passed = max_diff_copy <= TOLERANCE and max_diff_orig <= TOLERANCE and argmax_ok
    print("")
    print("RESULT: " + ("PASS - reload reproduces the original prediction"
                        if passed else "FAIL - reload differs beyond tolerance"))
    print("=" * 37)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
