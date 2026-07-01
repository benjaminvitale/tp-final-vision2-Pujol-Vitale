"""train.py — Loop de entrenamiento + validación de UNA corrida (1 modelo, 1 receta).

Receta del paper (ver plan.md / config.py):
- SGD(momentum=0.9), lr=0.001, StepLR(step_size=7, gamma=0.1), 50 épocas.
- Solo se optimizan los parámetros entrenables (con freeze_backbone, solo las FC).
- Se trackea val accuracy por época y se guarda el MEJOR checkpoint por val acc.

`train_one_run` es la unidad reusada por los scripts de Fase 3/4/5. Soporta subsetear
(max_train/max_val) y bajar épocas para smoke-tests rápidos del pipeline.

Etapa 2 (aditivo, backward-compatible): RunConfig acepta `data_dir`, `splits_dir`,
`num_classes`, `image_size` y `use_precomputed_aug`. Con todos en None/default el
comportamiento es EXACTAMENTE el de la Etapa 1 (dataset del paper). Pasándolos se puede
entrenar sobre otro dataset (p.ej. CMPD300) reusando este mismo loop, sin reimplementarlo.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

import config
from src.dataset import load_split, make_dataloader, make_train_loader
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
    # --- Etapa 2 (None = defaults de la Etapa 1 / dataset del paper) ---
    data_dir: str | None = None          # raíz de imágenes (p.ej. str(config.CMPD300_DIR))
    splits_dir: str | None = None        # carpeta de los JSON de split
    num_classes: int | None = None       # nº de clases del dataset (CMPD300: del label_map)
    image_size: int | None = None        # lado de entrada (Etapa 2: 224)
    use_precomputed_aug: bool = True      # False para datasets sin cache de aug (CMPD300)
    init_from: str | None = None          # warm-start del backbone desde un checkpoint propio


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

    # --- Resolución de fuentes (Etapa 2 si se pasan; si no, defaults Etapa 1) ---
    data_dir = Path(rc.data_dir) if rc.data_dir else config.DATA_DIR
    splits_dir = Path(rc.splits_dir) if rc.splits_dir else config.SPLITS_DIR
    num_classes = rc.num_classes if rc.num_classes is not None else config.NUM_CLASSES
    image_size = rc.image_size if rc.image_size is not None else config.IMAGE_SIZE

    log.info(f"=== RUN {tag} | device={device} ===")
    log.info(f"config: {asdict(rc)}")
    log.info(f"data_dir={data_dir} | splits_dir={splits_dir} | "
             f"num_classes={num_classes} | image_size={image_size}")

    # ---- Datos ----
    # PAPER: la augmentation CREA imágenes sintéticas y AGRANDA el dataset (mantiene los
    # originales a brillo real), no reemplaza cada imagen. Originales → transform limpio;
    # copias sintéticas → augmentation. Ver dataset.make_train_loader / DEVIATIONS D4.
    clean_tf = build_transforms(train=False, image_size=image_size,
                                use_imagenet_norm=rc.use_imagenet_norm)
    aug_tf = build_transforms(train=True, image_size=image_size,
                              use_imagenet_norm=rc.use_imagenet_norm)
    train_e = _maybe_subset(load_split("train", splits_dir=splits_dir), rc.max_train)
    val_e = _maybe_subset(load_split("val", splits_dir=splits_dir), rc.max_val)
    train_loader = make_train_loader(train_e, use_aug=rc.use_aug, clean_tf=clean_tf,
                                     aug_tf=aug_tf, seed=rc.seed,
                                     batch_size=rc.batch_size, num_workers=rc.num_workers,
                                     data_dir=data_dir,
                                     # con subset (smoke) o datasets sin cache NO usar el
                                     # cache precomputado (es del dataset del paper).
                                     use_precomputed=(rc.max_train is None and rc.use_precomputed_aug))
    val_loader = make_dataloader(val_e, clean_tf, shuffle=False, data_dir=data_dir,
                                 batch_size=rc.batch_size, num_workers=rc.num_workers)
    log.info(f"train imgs: {len(train_loader.dataset):,} (use_aug={rc.use_aug}) | "
             f"val imgs: {len(val_e):,}")

    # ---- Modelo / loss / optimizador ----
    model = build_model(rc.model_name, num_classes=num_classes,
                        freeze_backbone=rc.freeze_backbone, pretrained=rc.pretrained,
                        init_from=rc.init_from).to(device)
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
                "num_classes": num_classes,
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
