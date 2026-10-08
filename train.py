"""Train the plant disease model (EfficientNet-B0 transfer learning, 2 stages).

Run:  python train.py                (full training)
      python train.py --smoke        (1 epoch/stage on a small subset - pipeline test)

Saves:
  models/plant_disease_model.keras   (10-class plant+disease classifier)
  models/plant_classifier.keras      (species classifier used for plant detection)
  models/class_names.json
  models/plant_class_names.json
  models/model_config.json
  models/split_manifest.json         (exact train/val/test membership - reused by evaluate)
  outputs/training_accuracy.png
  outputs/training_loss.png
  outputs/augmentation_examples.png
"""
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch/stage, 400-image subset")
    ap.add_argument("--skip-plant-model", action="store_true", help="skip the species classifier")
    return ap.parse_args()


def header_info(split, n_classes, device, batch, lr1, lr2, e1, e2):
    import tensorflow as tf

    print("=" * 37)
    print("MODEL TRAINING")
    print("=" * 37)
    print(f"Python version     : {platform.python_version()} ({sys.executable})")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Device             : {device}")
    print(f"Dataset size       : {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test   (total {len(split['train']) + len(split['val']) + len(split['test'])})")
    print(f"Number of classes  : {n_classes}")
    print(f"Image size         : {config.IMAGE_SIZE}x{config.IMAGE_SIZE}x{config.CHANNELS}")
    print(f"Batch size         : {batch}")
    print(f"Learning rate      : stage1={lr1:g} (backbone frozen)  stage2={lr2:g} (upper layers fine-tuned)")
    print(f"Epoch count        : max {e1} (stage1) + {e2} (stage2), early stopping patience "
          f"{config.EARLY_STOPPING_PATIENCE}")
    print(f"Optimizer          : {config.OPTIMIZER}")
    print(f"Backbone           : {config.BACKBONE} (ImageNet pretrained), fallback {config.FALLBACK_BACKBONE}")
    print("=" * 37)


def detect_device() -> str:
    import tensorflow as tf

    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def build_backbone(name: str):
    """Returns (keras.Model backbone, preprocess_layer_or_None, name_used)."""
    from keras import applications, layers

    if name.upper() == "EFFICIENTNETB0":
        try:
            base = applications.EfficientNetB0(
                include_top=False, weights="imagenet",
                input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
            )
            return base, None, "EfficientNetB0"  # EfficientNet rescales 1/255 internally
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] EfficientNetB0 weights unavailable ({e}); falling back to {config.FALLBACK_BACKBONE}")
            name = config.FALLBACK_BACKBONE

    if name.upper() == "MOBILENETV2":
        base = applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        pre = layers.Rescaling(2.0 / 255.0, offset=-1.0, name="mobilenet_preprocess")
        return base, pre, "MobileNetV2"

    raise ValueError(f"Unknown backbone: {name}")


def build_model(num_classes: int, backbone_name: str = config.BACKBONE):
    """Input -> Augmentation -> Backbone -> GAP -> BN -> Dropout -> Dense256 -> Dropout -> Softmax."""
    from keras import layers

    inp = keras_input = layers.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="input_image"
    )
    # ---- data augmentation (training only, applied on raw 0..255 pixels)
    x = layers.RandomFlip("horizontal", name="aug_flip")(inp)
    x = layers.RandomRotation(0.06, name="aug_rotate")(x)
    x = layers.RandomZoom(0.10, name="aug_zoom")(x)
    x = layers.RandomTranslation(0.08, 0.08, name="aug_translate")(x)
    x = layers.RandomBrightness(0.10, value_range=(0, 255), name="aug_brightness")(x)
    x = layers.RandomContrast(0.10, value_range=(0, 255), name="aug_contrast")(x)

    backbone, pre, used = build_backbone(backbone_name)
    if pre is not None:
        x = pre(x)
    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(config.DROPOUT_1, name="head_dropout1")(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="head_dense")(x)
    x = layers.Dropout(config.DROPOUT_2, name="head_dropout2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    import keras

    model = keras.Model(keras_input, out, name=f"plant_{used.lower()}")
    return model, backbone, used


def make_dataset(paths, labels, training: bool, batch: int, cache_decode=False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((np.array(paths, dtype=object), np.array(labels, dtype=np.int32)))
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=config.SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), method="bilinear")
        img = tf.cast(img, tf.float32)  # 0..255, EfficientNet rescales internally
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(y_train, n_classes: int) -> dict:
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(y_train)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return {int(i): float(w) for i, w in enumerate(weights)}


def callbacks_for(model_path: str):
    import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.REDUCE_LR_MIN, verbose=1,
        ),
    ]


def train_one_model(model, backbone, train_ds, val_ds, class_weights, epochs1, epochs2,
                    lr1, lr2, model_path, tag=""):
    """Stage 1 (frozen backbone) + stage 2 (fine-tune upper layers). Returns history dicts."""
    import keras

    print(f"\n--- [{tag}] STAGE 1: backbone frozen | lr={lr1:g} | max epochs={epochs1} ---")
    # NOTE: in Keras 3 `parent.trainable = x` propagates to every child, therefore the
    # model must be unfrozen FIRST and only the backbone frozen afterwards - otherwise
    # the trainable head would be frozen as well and stage 1 would train nothing.
    model.trainable = True
    backbone.trainable = False
    trainable_head = sum(1 for l in model.layers if l.trainable)
    print(f"Trainable layers: head={trainable_head}, backbone=0/{len(backbone.layers)}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs1,
        class_weight=class_weights if config.USE_CLASS_WEIGHTS else None,
        callbacks=callbacks_for(model_path), verbose=1,
        shuffle=False,  # x is a tf.data.Dataset - already shuffled; avoids Keras UserWarning
    )

    print(f"\n--- [{tag}] STAGE 2: fine-tune upper {config.FINE_TUNE_LAYERS} backbone layers | "
          f"lr={lr2:g} | max epochs={epochs2} ---")
    layers_list = backbone.layers
    n_layers = len(layers_list)
    start = max(0, n_layers - config.FINE_TUNE_LAYERS)
    model.trainable = True   # propagates True to every sub-layer first...
    for i, l in enumerate(layers_list):
        # ...then freeze everything except the top of the backbone
        # (BatchNorm stays frozen: running stats are stable)
        l.trainable = (i >= start) and (l.__class__.__name__ != "BatchNormalization")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"Trainable backbone layers: {trainable}/{len(backbone.layers)}")
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs2,
        class_weight=class_weights if config.USE_CLASS_WEIGHTS else None,
        callbacks=callbacks_for(model_path), verbose=1,
        shuffle=False,  # x is a tf.data.Dataset - already shuffled; avoids Keras UserWarning
    )
    return {
        "stage1": {k: [float(x) for x in v] for k, v in h1.history.items()},
        "stage2": {k: [float(x) for x in v] for k, v in h2.history.items()},
    }


def plot_history(history, accuracy_path, loss_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1, s2 = history["stage1"], history["stage2"]
    epochs1, epochs2 = len(s1.get("accuracy", [])), len(s2.get("accuracy", []))
    total = epochs1 + epochs2
    x = np.arange(1, total + 1)

    acc = s1.get("accuracy", []) + s2.get("accuracy", [])
    vacc = s1.get("val_accuracy", []) + s2.get("val_accuracy", [])
    loss = s1.get("loss", []) + s2.get("loss", [])
    vloss = s1.get("val_loss", []) + s2.get("val_loss", [])

    def styled(ax, y1, y2, label1, label2, ylabel):
        ax.plot(x, y1, "o-", color="#1565c0", label=label1, linewidth=1.6, markersize=4)
        ax.plot(x, y2, "s--", color="#c62828", label=label2, linewidth=1.6, markersize=4)
        if epochs1:
            ax.axvline(epochs1 + 0.5, color="gray", linestyle=":", linewidth=1.2)
            ax.text(epochs1 + 0.5, ax.get_ylim()[0], " stage2 ", fontsize=8, color="gray")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(title)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    styled(ax, acc, vacc, "training accuracy", "validation accuracy", "accuracy")
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    styled(ax, loss, vloss, "training loss", "validation loss", "loss")
    fig.tight_layout()
    fig.savefig(loss_path, dpi=130)
    plt.close(fig)


def save_augmentation_examples():
    """outputs/augmentation_examples.png : original + 4 augmented versions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keras import layers
    import keras

    classes = config.discover_classes(config.resolve_dataset_dir())
    first = sorted(classes)[0]
    sample_path = classes[first][0]
    img = config.load_image_rgb(sample_path)

    aug = keras.Sequential(
        [
            layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomBrightness(0.10, value_range=(0, 255)),
            layers.RandomContrast(0.10, value_range=(0, 255)),
        ],
        name="augmentation_pipeline",
    )
    x = np.expand_dims(img, 0)
    outs = [np.clip(aug(x, training=True)[0].numpy(), 0, 255).astype(np.uint8) for _ in range(4)]

    titles = ["Original"] + [f"Augmented {i}" for i in range(1, 5)]
    panels = [img.astype(np.uint8)] + outs
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, panel, t in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Augmentation examples (source: {first})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    dest = os.path.join(config.OUTPUTS_DIR, "augmentation_examples.png")
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def main():
    args = parse_args()
    config.ensure_dirs()
    config.set_global_seed(config.SEED)

    import tensorflow as tf
    import keras

    device = detect_device()

    batch = config.BATCH_SIZE
    lr1, lr2 = config.LR_STAGE1, config.LR_STAGE2
    e1, e2 = config.EPOCHS_STAGE1, config.EPOCHS_STAGE2

    # ---------------- split
    split = config.build_split()
    class_names = split["class_names"]
    if len(class_names) < 2:
        raise SystemExit("FATAL: fewer than 2 classes discovered - check dataset/.")
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    config.save_split_json(split)

    species = sorted({config.species_from_class(c) for c in class_names})
    species_to_idx = {s: i for i, s in enumerate(species)}

    train_y = [class_to_idx[c] for _, c in split["train"]]
    val_y = [class_to_idx[c] for _, c in split["val"]]
    test_y = [class_to_idx[c] for _, c in split["test"]]

    header_info(split, len(class_names), device, batch, lr1, lr2, e1, e2)
    print(f"Class names        : {class_names}")
    print(f"Species (plants)   : {species}")
    print("")

    if args.smoke:
        e1 = e2 = 1
        split["train"] = split["train"][:400]
        split["val"] = split["val"][:80]
        train_y = train_y[:400]
        val_y = val_y[:80]
        print(f"[SMOKE MODE] epochs={e1}, train={len(split['train'])}, val={len(split['val'])}")
        print("")

    # ---------------- class weights
    class_weights = compute_class_weights(train_y, len(class_names))
    print("Class weights (balanced, training split):")
    for i, c in enumerate(class_names):
        print(f"  {c}: {class_weights[i]:.4f}")
    print("")

    # ---------------- datasets
    train_ds = make_dataset([p for p, _ in split["train"]], train_y, True, batch)
    val_ds = make_dataset([p for p, _ in split["val"]], val_y, False, batch)

    # ---------------- augmentation visual check
    aug_path = save_augmentation_examples()
    print(f"Augmentation examples -> {os.path.relpath(aug_path, config.PROJECT_DIR)}\n")

    # ---------------- primary model: plant + disease
    print("=" * 37)
    print("TRAINING PRIMARY MODEL (plant + disease)")
    print("=" * 37)
    t0 = time.time()
    model, backbone, used_backbone = build_model(len(class_names), config.BACKBONE)
    model.summary(print_fn=lambda s: print("  " + s), line_length=100)
    history = train_one_model(model, backbone, train_ds, val_ds, class_weights,
                              e1, e2, lr1, lr2, config.MODEL_PATH, tag="disease")
    train_time = time.time() - t0

    # ModelCheckpoint already stored the best weights in model_path; reload to be safe
    model = keras.models.load_model(config.MODEL_PATH)
    print(f"\nBest model saved -> {os.path.relpath(config.MODEL_PATH, config.PROJECT_DIR)}")

    plot_history(history,
                 os.path.join(config.OUTPUTS_DIR, "training_accuracy.png"),
                 os.path.join(config.OUTPUTS_DIR, "training_loss.png"),
                 "Primary model - plant+disease")
    print("Graphs -> outputs/training_accuracy.png, outputs/training_loss.png")

    # ---------------- species (plant) classifier: plant detection / identification
    plant_history = None
    if not args.skip_plant_model:
        print("\n" + "=" * 37)
        print("TRAINING PLANT SPECIES CLASSIFIER")
        print("=" * 37)
        sp_train_y = [species_to_idx[config.species_from_class(c)] for _, c in split["train"]]
        sp_val_y = [species_to_idx[config.species_from_class(c)] for _, c in split["val"]]
        sp_weights = compute_class_weights(sp_train_y, len(species))
        print("Species weights:", {species[k]: round(v, 3) for k, v in sp_weights.items()})
        sp_train_ds = make_dataset([p for p, _ in split["train"]], sp_train_y, True, batch)
        sp_val_ds = make_dataset([p for p, _ in split["val"]], sp_val_y, False, batch)
        pm, pbackbone, _ = build_model(len(species), config.BACKBONE)
        plant_history = train_one_model(pm, pbackbone, sp_train_ds, sp_val_ds, sp_weights,
                                        e1, e2, lr1, lr2, config.PLANT_MODEL_PATH, tag="plant")
        pm = keras.models.load_model(config.PLANT_MODEL_PATH)
        print(f"\nBest plant model saved -> {os.path.relpath(config.PLANT_MODEL_PATH, config.PROJECT_DIR)}")
        plot_history(plant_history,
                     os.path.join(config.OUTPUTS_DIR, "plant_training_accuracy.png"),
                     os.path.join(config.OUTPUTS_DIR, "plant_training_loss.png"),
                     "Plant species classifier")

    # ---------------- metadata
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    with open(config.PLANT_CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(species, fh, indent=2)

    best1 = history["stage1"].get("val_accuracy") or [None]
    best2 = history["stage2"].get("val_accuracy") or [None]
    best1, best2 = best1[-1], best2[-1]
    cfg = {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "device": device,
        "dataset_dir": config.relative_project_path(split["dataset_dir"]),
        "dataset_size": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "class_names": class_names,
        "plant_class_names": species,
        "image_size": config.IMAGE_SIZE,
        "channels": config.CHANNELS,
        "batch_size": batch,
        "optimizer": config.OPTIMIZER,
        "learning_rates": {"stage1": lr1, "stage2": lr2},
        "epochs": {
            "stage1_max": e1, "stage2_max": e2,
            "stage1_ran": len(history["stage1"].get("accuracy", [])),
            "stage2_ran": len(history["stage2"].get("accuracy", [])),
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        },
        "fine_tune_layers": config.FINE_TUNE_LAYERS,
        "backbone": used_backbone,
        "architecture": (
            "Input(224x224x3) -> Augmentation(horizontal flip, rotate 0.06, zoom 0.10, "
            "translate 0.08, brightness 0.10, contrast 0.10) -> "
            f"{used_backbone} (ImageNet) -> GlobalAveragePooling2D -> BatchNormalization -> "
            f"Dropout({config.DROPOUT_1}) -> Dense({config.DENSE_UNITS}, relu) -> "
            f"Dropout({config.DROPOUT_2}) -> Softmax"
        ),
        "class_weights": {class_names[k]: round(v, 4) for k, v in class_weights.items()},
        "use_class_weights": config.USE_CLASS_WEIGHTS,
        "seed": config.SEED,
        "train_seconds": round(train_time, 1),
        "final_val_accuracy_stage1": best1,
        "final_val_accuracy_stage2": best2,
        "history": history,
        "plant_model_history": plant_history,
        "limitations": [
            "Training data is isolated-leaf imagery (PlantVillage); performance on "
            "complete-plant photographs is not guaranteed.",
            "Disease severity labels are not present in this dataset => severity is not predicted.",
        ],
    }
    with open(config.MODEL_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print("")
    print("=" * 37)
    print("TRAINING COMPLETE")
    print("=" * 37)
    print(f"Model      : {os.path.relpath(config.MODEL_PATH, config.PROJECT_DIR)}")
    print(f"Classes    : {os.path.relpath(config.CLASS_NAMES_PATH, config.PROJECT_DIR)}")
    print(f"Config     : {os.path.relpath(config.MODEL_CONFIG_PATH, config.PROJECT_DIR)}")
    print(f"Val acc    : stage1={best1 if best1 is None else round(best1, 4)}  "
          f"stage2={best2 if best2 is None else round(best2, 4)}")
    print(f"Wall time  : {train_time:.1f}s (+ plant classifier)")
    print("Next: python evaluate.py")
    print("Severity  : Not available (dataset has no severity labels)")


if __name__ == "__main__":
    main()
