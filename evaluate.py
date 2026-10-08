"""Evaluate the trained model on the held-out test split (never used for training).

Run:  python evaluate.py
Outputs:
  outputs/metrics.json
  outputs/classification_report.txt
  outputs/confusion_matrix.png
"""
import json
import os
import time

import numpy as np

import config


def load_test_split():
    manifest = os.path.join(config.MODELS_DIR, "split_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        items = [(config.project_path(x["path"]), x["class"]) for x in m["test"]]
        if items:
            dataset_dir = m.get("dataset_dir")
            return (items, m["class_names"],
                    config.project_path(dataset_dir) if dataset_dir else None)
    split = config.build_split()
    return [(p, c) for p, c in split["test"]], split["class_names"], split["dataset_dir"]


def predict_in_batches(model, paths, batch_size=32):
    preds, probs = [], []
    for i in range(0, len(paths), batch_size):
        batch = np.stack([config.load_image_rgb(p) for p in paths[i:i + batch_size]])
        p = model.predict(batch, verbose=0)
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
    return (np.concatenate(preds) if preds else np.array([]),
            np.concatenate(probs) if probs else np.empty((0, 0)))


def species_of(class_name: str) -> str:
    return config.species_from_class(class_name)


def main():
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    import keras

    config.ensure_dirs()
    started = time.time()
    if not os.path.isfile(config.MODEL_PATH):
        raise SystemExit(f"FATAL: model not found at {config.MODEL_PATH}. Run: python train.py")

    model = keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
        class_names = json.load(fh)
    items, split_classes, dataset_dir = load_test_split()
    if split_classes != class_names:
        print("[WARN] class names in split manifest differ from class_names.json")

    paths = [p for p, _ in items]
    y_true_names = [c for _, c in items]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([class_to_idx[c] for c in y_true_names])

    print("=" * 37)
    print("MODEL EVALUATION (held-out test set)")
    print("=" * 37)
    print(f"Model        : {config.MODEL_PATH}")
    print(f"Test images  : {len(paths)}  (from {dataset_dir})")
    print(f"Classes      : {len(class_names)}")
    t0 = time.time()
    y_pred_idx, probs = predict_in_batches(model, paths)
    print(f"Inference    : {time.time() - t0:.1f}s")

    y_pred = np.array([class_names[int(i)] for i in y_pred_idx])

    # ------------------------------------------------ metrics
    acc = float(accuracy_score(y_true_names, y_pred))
    precision_macro = float(precision_score(y_true_names, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true_names, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_names, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_names, y_pred, average="weighted", zero_division=0))
    precision_weighted = float(precision_score(y_true_names, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true_names, y_pred, average="weighted", zero_division=0))

    report = classification_report(y_true_names, y_pred, labels=class_names, zero_division=0)
    cm = confusion_matrix(y_true_names, y_pred, labels=class_names)

    per_class = {}
    for i, c in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted_as_i = int(cm[:, i].sum())
        per_class[c] = {
            "precision": round(float(cm[i, i] / predicted_as_i) if predicted_as_i else 0.0, 4),
            "recall": round(float(cm[i, i] / support) if support else 0.0, 4),
            "f1": round(float(2 * cm[i, i] / (predicted_as_i + support))
                        if (predicted_as_i + support) else 0.0, 4),
            "support": support,
        }

    # species (plant) level accuracy, derived from the disease model output
    true_species = [species_of(c) for c in y_true_names]
    pred_species = [species_of(c) for c in y_pred]
    species_acc = float(np.mean([a == b for a, b in zip(true_species, pred_species)])) if paths else 0.0

    # top misclassifications
    mismatches = []
    for i, j in np.ndindex(cm.shape):
        if i != j and cm[i, j] > 0:
            mismatches.append((int(cm[i, j]), class_names[i], class_names[j]))
    mismatches.sort(reverse=True)

    # mean confidence + accuracy-vs-confidence buckets
    conf = probs.max(axis=1) if len(probs) else np.array([])
    correct = (y_pred == np.array(y_true_names))

    # ------------------------------------------------ print
    print("")
    print(f"Accuracy                 : {acc:.4f}")
    print(f"Precision (macro)        : {precision_macro:.4f}")
    print(f"Recall (macro)           : {recall_macro:.4f}")
    print(f"F1 score (macro)         : {f1_macro:.4f}")
    print(f"F1 score (weighted)      : {f1_weighted:.4f}")
    print(f"Precision (weighted)     : {precision_weighted:.4f}")
    print(f"Recall (weighted)        : {recall_weighted:.4f}")
    print(f"Species-level accuracy   : {species_acc:.4f}  (plant identification from disease model)")
    if len(conf):
        print(f"Mean confidence          : {conf.mean():.4f}")
        print(f"Accuracy @ conf>=0.80    : "
              f"{float(correct[conf >= config.HIGH_CONFIDENCE].mean()) if (conf >= config.HIGH_CONFIDENCE).any() else float('nan'):.4f} "
              f"({int((conf >= config.HIGH_CONFIDENCE).sum())} images)")
        low = conf < config.MEDIUM_CONFIDENCE
        print(f"Accuracy @ conf<0.60     : "
              f"{float(correct[low].mean()) if low.any() else float('nan'):.4f} ({int(low.sum())} images)")

    print("\nPer-class metrics:")
    print(f"  {'class':40s} {'precision':>9s} {'recall':>9s} {'f1':>9s} {'support':>8s}")
    for c in class_names:
        m = per_class[c]
        print(f"  {c:40s} {m['precision']:9.4f} {m['recall']:9.4f} {m['f1']:9.4f} {m['support']:8d}")

    print("\nImportant misclassifications (actual -> predicted):")
    if not mismatches:
        print("  none - all test predictions correct")
    for count, actual, predicted in mismatches[:15]:
        print(f"  Actual:    {actual}\n  Predicted: {predicted}\n  Count:     {count}\n")

    print("LIMITATION: the classifier is trained primarily on leaf imagery and may have "
          "reduced performance on complete-plant photographs.")

    # ------------------------------------------------ confusion matrix plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar_kws={"label": "count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - test set ({len(paths)} images, accuracy {acc:.3f})")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=140)
    plt.close(fig)

    # ------------------------------------------------ files
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(classification_report(y_true_names, y_pred, labels=class_names, digits=4,
                                       zero_division=0))
        fh.write("\nOverall accuracy: %.4f\n" % acc)
        fh.write("Macro F1        : %.4f\n" % f1_macro)
        fh.write("Weighted F1     : %.4f\n" % f1_weighted)
        fh.write("Species acc     : %.4f\n" % species_acc)
        fh.write("\nConfusion matrix rows=actual, cols=predicted, labels order:\n")
        for i, c in enumerate(class_names):
            fh.write(f"  {i}: {c}\n")
        fh.write("\nConfusion matrix:\n")
        fh.write(np.array2string(cm, separator=" "))
        fh.write("\n")

    metrics = {
        "model": config.MODEL_PATH,
        "test_images": len(paths),
        "dataset_dir": dataset_dir,
        "class_names": class_names,
        "accuracy": round(acc, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "species_accuracy": round(species_acc, 4),
        "mean_confidence": round(float(conf.mean()), 4) if len(conf) else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_misclassifications": [
            {"actual": a, "predicted": p, "count": c} for c, a, p in mismatches[:25]
        ],
        "limitations": [
            "Training data consists of isolated-leaf images (PlantVillage-style); "
            "complete-plant photograph performance is not guaranteed.",
            "No severity labels in the dataset => severity not predicted.",
            "Out-of-distribution rejection is confidence-based only (no true OOD detector).",
        ],
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_seconds": round(time.time() - started, 1),
    }
    metrics_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("")
    print(f"Saved -> {os.path.relpath(metrics_path, config.PROJECT_DIR)}")
    print(f"Saved -> {os.path.relpath(report_path, config.PROJECT_DIR)}")
    print(f"Saved -> {os.path.relpath(cm_path, config.PROJECT_DIR)}")
    print("=" * 37)


if __name__ == "__main__":
    main()
