"""evaluate.py — Evaluación en test: top-1 global y accuracy POR CLASE.

PAPER: top-1 accuracy global + por clase. Reportamos también las clases con pocas
imágenes (las de 4 imágenes) sin esconderlas detrás del promedio. Opcional: ms/imagen.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

import config
from src.dataset import load_split, make_dataloader
from src.models import build_model
from src.transforms import build_transforms
from src.utils import get_device, load_json


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str,
             num_classes: int = config.NUM_CLASSES) -> dict:
    """Devuelve accuracy global, accuracy por clase y timing (ms/imagen)."""
    model.eval()
    per_correct = torch.zeros(num_classes)
    per_total = torch.zeros(num_classes)
    correct, total = 0, 0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        ok = (preds == labels)
        correct += ok.sum().item()
        total += labels.size(0)
        for lbl, hit in zip(labels.cpu(), ok.cpu()):
            per_total[lbl] += 1
            per_correct[lbl] += hit.item()
    elapsed = time.time() - t0

    present = per_total > 0
    per_class_acc = torch.where(present, per_correct / per_total.clamp(min=1),
                                torch.full_like(per_total, float("nan")))
    # accuracy media POR CLASE (balanced) además de la global
    balanced = per_class_acc[present].mean().item()
    return {
        "global_acc": correct / max(total, 1),
        "balanced_acc": balanced,
        "n_test": total,
        "ms_per_image": round(1000 * elapsed / max(total, 1), 2),
        "per_class_acc": per_class_acc.tolist(),
        "per_class_total": per_total.int().tolist(),
        "per_class_correct": per_correct.int().tolist(),
    }


def evaluate_checkpoint(checkpoint: str | Path, device: str | None = None,
                        max_test: int | None = None,
                        save_csv: Path | None = None) -> dict:
    """Carga un checkpoint, evalúa en test y (opcional) guarda CSV por clase."""
    device = device or get_device()
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"],
                        freeze_backbone=True, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    eval_tf = build_transforms(train=False)
    test_e = load_split("test")
    if max_test is not None:
        test_e = test_e[:max_test]
    loader = make_dataloader(test_e, eval_tf, shuffle=False,
                             batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    res = evaluate(model, loader, device)

    if save_csv is not None:
        _save_per_class_csv(res, save_csv)
    return res


def _save_per_class_csv(res: dict, path: Path) -> None:
    """Guarda accuracy por clase (label, nombre, n_test, correct, acc)."""
    import csv

    label_map = load_json(config.SPLITS_DIR / "label_map.json")  # nombre -> idx
    idx_to_name = {v: k for k, v in label_map.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "class_name", "n_test", "correct", "accuracy"])
        for i, (n, c, a) in enumerate(zip(res["per_class_total"],
                                          res["per_class_correct"],
                                          res["per_class_acc"])):
            w.writerow([i, idx_to_name.get(i, f"label_{i}"), n, c,
                        "" if a != a else round(a, 4)])  # a!=a → NaN
