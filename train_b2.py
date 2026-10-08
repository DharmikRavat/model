"""Train EfficientNet-B2 on GPU (PlantVillage + PlantDoc-train).

PlantDoc's ``other_plant`` images form an explicit reject class; their held-out
test split is included in the real-world report and never used for training.

Run:
  python train_b2.py                 # full run
  python train_b2.py --epochs2 12    # shorter fine-tune stage
  python train_b2.py --quick         # smoke test (300 train imgs, 1 epoch each stage)

Saves:
  models/plant_disease_b2.pt   weights
  models/b2_config.json        class names, image size, metrics
  outputs/b2_report.txt/.json  accuracy, precision/recall/F1, confusion matrix, per-class recall
  outputs/b2_confusion_matrix.png, outputs/b2_training_curves.png
"""
import argparse
import json
import os
import random
import time

import numpy as np

import b2_data as D
import config

SEED = 42


# ---------------------------------------------------------------- helpers
def seed_everything(s=SEED):
    random.seed(s)
    np.random.seed(s)
    import torch

    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(n_classes: int):
    import torch.nn as nn
    from torchvision import models

    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = models.efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, n_classes))
    return model


def freeze_backbone(model, freeze: bool):
    for p in model.features.parameters():
        p.requires_grad = not freeze
    for p in model.classifier.parameters():
        p.requires_grad = True


def class_weights(counts, class_names):
    import torch

    total = sum(counts.values())
    w = []
    for c in class_names:
        n = max(counts.get(c, 1), 1)
        w.append(total / (len(class_names) * n))
    t = torch.tensor(w, dtype=torch.float32)
    return t / t.mean()


# ---------------------------------------------------------------- evaluation
def evaluate(model, loader, device, criterion=None):
    import torch

    model.eval()
    y_true, y_pred, y_conf = [], [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(xb)
            logits = logits.float()
            if criterion is not None:
                loss_sum += float(criterion(logits, yb)) * len(yb)
            pred = logits.float().softmax(1).argmax(1)
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            n += len(yb)
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if n else 0.0
    return {"acc": acc, "loss": (loss_sum / n) if n and criterion is not None else None,
            "y_true": y_true, "y_pred": y_pred}


def full_report(y_true, y_pred, class_names, title=""):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, digits=4, zero_division=0)
    lines = [title, "=" * 60, rep,
             f"accuracy    : {float((y_true == y_pred).mean()):.4f}",
             f"macro F1    : {f1_score(y_true, y_pred, average='macro'):.4f}",
             f"weighted F1 : {f1_score(y_true, y_pred, average='weighted'):.4f}"]
    per_class = {}
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))),
                                                 zero_division=0)
    for i, c in enumerate(class_names):
        per_class[c] = {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                        "f1": round(float(f[i]), 4), "support": int(s[i])}
    txt = "\n".join(lines)
    return txt, cm, per_class, {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }


def save_confusion(cm, class_names, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="Actual", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_curves(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["acc"], label="train")
    axes[0].plot(history["val_acc"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- training
def train(args):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    seed_everything()
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"gpu         : {torch.cuda.get_device_name(0)}")

    bs = args.batch_size
    (train_loader, val_loader, val_real_loader, val_ood_loader,
     test_loader, real_loader, sets) = D.make_loaders(batch_size=bs)
    classes = D.class_names()
    n_classes = len(classes)
    print(f"train/val/val_real/val_ood/test/real = {len(sets['train'])}/{len(sets['val'])}/"
          f"{len(sets['val_real'])}/{len(sets['val_ood'])}/{len(sets['test'])}/"
          f"{len(sets['real_world_test'])}")
    if args.quick:
        per_class = {}
        for item in train_loader.dataset.items:
            per_class.setdefault(item[1], []).append(item)
        train_loader.dataset.items = [item for group in per_class.values() for item in group[:25]]
        print(f"[quick] train capped to {len(train_loader.dataset.items)}")

    counts = D.label_counts(train_loader.dataset.items)
    cw = class_weights(counts, classes).to(device)
    print("class weights:", {c: round(float(cw[i]), 3) for i, c in enumerate(classes)})

    model = build_model(n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    def run_stage(name, epochs, lr, freeze, stage_offset):
        if epochs <= 0:
            return
        freeze_backbone(model, freeze)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=1e-4)
        warm = LinearLR(opt, start_factor=0.05, total_iters=max(1, int(epochs * 0.1)))
        sched = CosineAnnealingLR(opt, T_max=max(1, epochs - int(epochs * 0.1)), eta_min=lr * 0.05)
        lr_sch = SequentialLR(opt, [warm, sched], milestones=[max(1, int(epochs * 0.1))])
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        best_acc, best_epoch, best_state, patience = -1.0, -1, None, 0

        for ep in range(1, epochs + 1):
            model.train()
            freeze_backbone(model, freeze)
            t0 = time.time()
            tot, correct, loss_sum = 0, 0, 0.0
            for xb, yb, _ in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                loss_sum += float(loss.detach()) * len(yb)
                correct += int((logits.float().argmax(1) == yb).sum())
                tot += len(yb)
            lr_sch.step()
            v = evaluate(model, val_loader, device, criterion)
            vr = evaluate(model, val_real_loader, device) if len(val_real_loader.dataset) else None
            vo = evaluate(model, val_ood_loader, device) if len(val_ood_loader.dataset) else None
            scores = [(0.4, v["acc"])]
            if vr is not None:
                scores.append((0.3, vr["acc"]))
            if vo is not None:
                scores.append((0.3, vo["acc"]))
            score = sum(weight * value for weight, value in scores) / sum(
                weight for weight, _ in scores)
            tr_acc = correct / max(tot, 1)
            history["acc"].append(tr_acc)
            history["loss"].append(loss_sum / max(tot, 1))
            history["val_acc"].append(v["acc"])
            history["val_loss"].append(v["loss"])
            history.setdefault("val_real_acc", []).append(vr["acc"] if vr else 0.0)
            history.setdefault("val_ood_acc", []).append(vo["acc"] if vo else 0.0)
            mark = ""
            if score > best_acc + 1e-4:
                best_acc, best_epoch, patience = score, stage_offset + ep, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                mark = " *"
            else:
                patience += 1
            real_txt = f" real_val {vr['acc']:.4f}" if vr is not None else ""
            ood_txt = f" ood_val {vo['acc']:.4f}" if vo is not None else ""
            print(f"[{name}] epoch {ep:2d}/{epochs} "
                  f"train {tr_acc:.4f} loss {loss_sum / max(tot, 1):.4f} "
                  f"val {v['acc']:.4f} val_loss {v['loss']:.4f}{real_txt}{ood_txt} "
                  f"score {score:.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s{mark}",
                  flush=True)
            if args.early_stop and patience >= args.patience:
                print(f"[{name}] early stop at epoch {ep} (best {best_epoch})")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return best_acc

    history = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
    t_start = time.time()
    print("\n--- stage 1: frozen backbone (head warm-up) ---")
    run_stage("S1", args.epochs1, 1e-3, freeze=True, stage_offset=0)
    print("\n--- stage 2: full fine-tune ---")
    run_stage("S2", args.epochs2, 1e-4, freeze=False, stage_offset=args.epochs1)
    train_time = time.time() - t_start

    # ---------------------------------------------------------------- evaluate
    print("\n=== evaluation ===")
    tag = "smoke_" if args.quick else ""
    reports = {}
    for name, loader in (("plantvillage_test", test_loader), ("plantdoc_real_world", real_loader)):
        if len(loader.dataset) == 0:
            print(f"{name}: empty, skipped")
            continue
        r = evaluate(model, loader, device)
        txt, cm, per_class, agg = full_report(r["y_true"], r["y_pred"], classes, title=name)
        print(txt)
        reports[name] = {"aggregate": agg, "per_class": per_class, "confusion_matrix": cm.tolist()}
        out_txt = os.path.join(config.OUTPUTS_DIR, f"{tag}b2_report_{name}.txt")
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n\nConfusion matrix (rows=actual, cols=predicted):\n" +
                     np.array2string(cm))
        save_confusion(cm, classes,
                       os.path.join(config.OUTPUTS_DIR, f"{tag}b2_confusion_matrix_{name}.png"),
                       f"EfficientNet-B2 / {name}")

    save_curves(history, os.path.join(config.OUTPUTS_DIR, f"{tag}b2_training_curves.png"))

    # ---------------------------------------------------------------- save
    model_name = "_smoke_b2.pt" if args.quick else "plant_disease_b2.pt"
    torch.save({k: t for k, t in model.state_dict().items()},
               os.path.join(config.MODELS_DIR, model_name))
    meta = {
        "backbone": "EfficientNet-B2",
        "image_size": D.IMAGE_SIZE,
        "mean": D.IMAGENET_MEAN,
        "std": D.IMAGENET_STD,
        "class_names": classes,
        "batch_size": bs,
        "epochs_stage1": args.epochs1,
        "epochs_stage2": args.epochs2,
        "train_size": len(sets["train"]),
        "train_time_sec": round(train_time, 1),
        "metrics": reports,
    }
    with open(os.path.join(config.MODELS_DIR, f"{tag}b2_config.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(config.OUTPUTS_DIR, f"{tag}b2_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"\nmodel  -> models/{model_name}")
    print(f"report -> outputs/{tag}b2_metrics.json")
    print(f"train time {train_time:.0f}s")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs1", type=int, default=5)
    ap.add_argument("--epochs2", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--early_stop", action="store_true", default=True)
    ap.add_argument("--no_early_stop", dest="early_stop", action="store_false")
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import torch  # noqa: F401
    train(parse())
