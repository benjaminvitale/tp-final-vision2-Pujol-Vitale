"""train.py — Loop de entrenamiento + validación de UNA corrida (1 modelo, 1 receta).

Receta del paper (ver plan.md / config.py):
- SGD(momentum=0.9), lr=0.001, StepLR(step_size=7, gamma=0.1), 50 épocas.
- Solo se optimizan los parámetros entrenables (con freeze_backbone, solo las FC).
- Se trackea val accuracy por época y se guarda el MEJOR checkpoint por val acc.

`train_one_run` es la unidad reusada por los scripts de Fase 3/4. Soporta subsetear
(max_train/max_val) y bajar épocas para smoke-tests rápidos del pipeline.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

import config
from src.dataset import load_split, make_dataloader
from src.losses import build_loss
from src.models import build_model, count_parameters, trainable_parameters
from src.transforms import build_transforms
from src.utils import get_device, get_logger, save_json, set_seed


@dataclass
class RunConfig:
    """Configuración de una corrida (se loguea y se guarda con el checkpoint)."""
    model_name: str
    loss_kind: str = "ce"          # 'ce' | 'wce'
    use_aug: bool = False          # data augmentation en train
    seed: int = 0
    freeze_backbone: bool = config.FREEZE_BACKBONE
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH_SIZE
    num_workers: int = config.NUM_WORKERS
    lr: float = config.LR
    momentum: float = config.MOMENTUM
    lr_step_size: int = config.LR_STEP_SIZE
    lr_gamma: float = config.LR_GAMMA
    pretrained: bool = True
    use_imagenet_norm: bool = config.USE_IMAGENET_NORM
    tag: str = ""
    # subset para smoke-tests (None = dataset completo)
    max_train: int | None = None
    max_val: int | None = None


def _maybe_subset(entries: list[dict], n: int | None) -> list[dict]:
    return entries if n is None else entries[:n]


def run_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    """Una época. Si optimizer es None → modo evaluación (sin grad). Devuelve (loss, acc)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_run(rc: RunConfig, device: str | None = None,
                  ckpt_dir: Path = config.CHECKPOINTS_DIR) -> dict:
    """Entrena una corrida y devuelve métricas + ruta del mejor checkpoint."""
    device = device or get_device()
    set_seed(rc.seed)
    tag = rc.tag or f"{rc.model_name}_{rc.loss_kind}_{'aug' if rc.use_aug else 'noaug'}_s{rc.seed}"
    log = get_logger(f"train.{tag}", logfile=config.LOGS_DIR / f"{tag}.log")
    config.ensure_output_dirs()

    log.info(f"=== RUN {tag} | device={device} ===")
    log.info(f"config: {asdict(rc)}")

    # ---- Datos ----
    train_tf = build_transforms(train=rc.use_aug, use_imagenet_norm=rc.use_imagenet_norm)
    eval_tf = build_transforms(train=False, use_imagenet_norm=rc.use_imagenet_norm)
    train_e = _maybe_subset(load_split("train"), rc.max_train)
    val_e = _maybe_subset(load_split("val"), rc.max_val)
    train_loader = make_dataloader(train_e, train_tf, shuffle=True,
                                   batch_size=rc.batch_size, num_workers=rc.num_workers)
    val_loader = make_dataloader(val_e, eval_tf, shuffle=False,
                                 batch_size=rc.batch_size, num_workers=rc.num_workers)

    # ---- Modelo / loss / optimizador ----
    model = build_model(rc.model_name, num_classes=config.NUM_CLASSES,
                        freeze_backbone=rc.freeze_backbone, pretrained=rc.pretrained).to(device)
    tr, tot = count_parameters(model)
    log.info(f"params entrenables: {tr:,}/{tot:,} ({100*tr/tot:.1f}%)")

    criterion = build_loss(rc.loss_kind, train_entries=train_e, device=device)
    optimizer = torch.optim.SGD(trainable_parameters(model), lr=rc.lr, momentum=rc.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=rc.lr_step_size, gamma=rc.lr_gamma)

    # ---- Loop ----
    best_val, best_epoch = -1.0, -1
    history = []
    ckpt_path = Path(ckpt_dir) / f"{tag}_best.pt"
    t0 = time.time()
    for epoch in range(1, rc.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr_now})
        log.info(f"ep {epoch:02d}/{rc.epochs} | train acc {tr_acc:.4f} loss {tr_loss:.4f} "
                 f"| val acc {va_acc:.4f} loss {va_loss:.4f} | lr {lr_now:.1e}")
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save({
                "model_state": model.state_dict(),
                "model_name": rc.model_name,
                "num_classes": config.NUM_CLASSES,
                "epoch": epoch,
                "val_acc": va_acc,
                "run_config": asdict(rc),
            }, ckpt_path)

    elapsed = time.time() - t0
    log.info(f"FIN {tag}: best val acc {best_val:.4f} @ ep {best_epoch} | {elapsed:.0f}s")
    return {
        "tag": tag,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(elapsed, 1),
        "history": history,
        "run_config": asdict(rc),
    }
