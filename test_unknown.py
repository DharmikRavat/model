"""Evaluate open-set rejection on non-plants and held-out unsupported PlantDoc leaves.

Run: python test_unknown.py
"""
import os
from collections import Counter

import config
from predict_b2 import predict

PLANTDOC_OTHER_DIR = os.path.join(config.PROJECT_DIR, "dataset_plantdoc", "test", "other_plant")


def collect_images(directory: str) -> list[str]:
    paths = []
    if not os.path.isdir(directory):
        return paths
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        paths.extend(
            os.path.join(root, name)
            for name in sorted(files)
            if config.is_supported_image(name)
        )
    return paths


def evaluate_group(label: str, paths: list[str]) -> tuple[int, int]:
    rejected = 0
    reasons = Counter()
    accepted = []
    for path in paths:
        result = predict(path)
        if result["plant_detected"]:
            accepted.append((os.path.basename(path), result["prediction"], result["confidence"]))
        else:
            rejected += 1
            reasons[result["status"]] += 1

    print(f"\n{label}: rejected {rejected}/{len(paths)}")
    print(f"  rejection reasons: {dict(reasons)}")
    for name, prediction, confidence in accepted[:15]:
        print(f"  false accept: {name} -> {prediction} ({confidence:.3f})")
    return rejected, len(paths)


def main() -> None:
    config.ensure_dirs()
    unknown = collect_images(config.TEST_IMAGES_DIR)
    unsupported = collect_images(PLANTDOC_OTHER_DIR)
    if not unknown and not unsupported:
        raise SystemExit("No non-plant or held-out unsupported-plant images found.")

    print("B2 OPEN-SET REJECTION CHECK")
    print("=" * 37)
    evaluate_group("Non-plant examples", unknown)
    evaluate_group("PlantDoc unsupported plants (held out)", unsupported)
    print("\nOpen-set detection is a learned screening signal, not a guarantee against every "
          "unseen image. Review false accepts before deployment.")


if __name__ == "__main__":
    main()
