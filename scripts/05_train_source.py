"""05_train_source.py — Baseline SOURCE (hocico) sobre CMPD300 (Fase 5, Etapa 2).

Entrena el encoder de hocico que después se transfiere al dominio de caras (Ahmed).
Reusa el loop de la Etapa 1 (`src/train.train_one_run`) apuntándolo a CMPD300 vía los
campos nuevos de RunConfig (data_dir / splits_dir / num_classes / image_size). NO se
reimplementa el loop ni la evaluación (test usa `run_epoch` en modo eval).

Preprocesamiento (decisión Etapa 2, ver config): 224 + norm ImageNet. ResNet-50.
El mejor checkpoint por val acc se copia a outputs/checkpoints/cmpd300_source.pt
→ ese es el encoder que tomará la Fase 6 (embeddings + gallery/probe sobre caras).

Requiere correr antes:  python scripts/00_inspect_cmpd300.py   (genera splits + label_map)

Uso:
    python scripts/05_train_source.py                 # freeze (solo FC), 50 épocas, seed 0
    python scripts/05_train_source.py --no-freeze     # fine-tune completo del backbone
    python scripts/05_train_source.py --aug           # + data augmentation online en train
    python scripts/05_train_source.py --smoke         # pipeline rápido en CPU (subset, 2 épocas)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# permitir correr desde cualquier cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

import config
from src.dataset import load_split, make_dataloader
from src.models import build_model
from src.transforms import build_transforms
from src.train import RunConfig, run_epoch, train_one_run
from src.utils import get_device, get_logger, load_json, save_json


def _num_classes() -> int:
    """Lee el nº de clases del label_map generado por 00_inspect_cmpd300.py."""
    lm = config.CMPD300_SPLITS_DIR / "label_map.json"
    if not lm.is_file():
        raise FileNotFoundError(
            f"No existe {lm}. Corré primero: python scripts/00_inspect_cmpd300.py"
        )
    return len(load_json(lm))


def evaluate_test(model_name: str, ckpt_path: Path, num_classes: int,
                  rc: RunConfig, device: str, log) -> float:
    """Carga el mejor checkpoint y devuelve top-1 accuracy en el split test de CMPD300."""
    test_e = load_split("test", splits_dir=config.CMPD300_SPLITS_DIR)
    clean_tf = build_transforms(train=False, image_size=rc.image_size,
                                use_imagenet_norm=rc.use_imagenet_norm)
    test_loader = make_dataloader(test_e, clean_tf, shuffle=False,
                                  data_dir=Path(rc.data_dir),
                                  batch_size=rc.batch_size, num_workers=rc.num_workers)
    model = build_model(model_name, num_classes=num_classes,
                        freeze_backbone=rc.freeze_backbone, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)["model_state"]
    model.load_state_dict(state)
    _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), None, device)
    log.info(f"TEST top-1 acc (CMPD300, {len(test_e)} imgs): {test_acc:.4f}")
    return test_acc


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline source (hocico) en CMPD300.")
    ap.add_argument("--no-freeze", action="store_true",
                    help="fine-tune completo del backbone (default: freeze, solo FC).")
    ap.add_argument("--aug", action="store_true", help="data augmentation online en train.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline rápido: subset + 2 épocas + sin ImageNet (CPU ok).")
    args = ap.parse_args()

    log = get_logger("train.source.cmpd300")
    device = get_device()
    config.ensure_output_dirs()
    num_classes = _num_classes()
    log.info(f"CMPD300 num_classes={num_classes} | device={device}")

    rc = RunConfig(
        model_name="resnet50",
        loss_kind="ce",
        use_aug=args.aug,
        seed=args.seed,
        freeze_backbone=not args.no_freeze,     # default: freeze (solo FC)
        epochs=2 if args.smoke else args.epochs,
        pretrained=not args.smoke,              # smoke: sin pesos (no requiere internet)
        use_imagenet_norm=False if args.smoke else config.USE_IMAGENET_NORM_S2,
        image_size=config.IMAGE_SIZE_S2,        # 224 (Etapa 2)
        tag=f"cmpd300_resnet50_{'ft' if args.no_freeze else 'freeze'}"
            f"{'_aug' if args.aug else ''}_s{args.seed}{'_smoke' if args.smoke else ''}",
        # --- apuntar el loop a CMPD300 ---
        data_dir=str(config.CMPD300_DIR),
        splits_dir=str(config.CMPD300_SPLITS_DIR),
        num_classes=num_classes,
        use_precomputed_aug=False,              # CMPD300 no tiene cache de aug → aug online
        max_train=64 if args.smoke else None,
        max_val=64 if args.smoke else None,
    )

    result = train_one_run(rc, device=device)
    best_ckpt = Path(result["checkpoint"])

    # Evaluación en test (closed-set top-1), reusando run_epoch.
    test_acc = None
    if not args.smoke:
        test_acc = evaluate_test("resnet50", best_ckpt, num_classes, rc, device, log)

    # Checkpoint estable para la Fase 6 (transferencia a caras).
    stable = config.CHECKPOINTS_DIR / "cmpd300_source.pt"
    shutil.copy(best_ckpt, stable)
    log.info(f"encoder source copiado a {stable}")

    summary = {
        "tag": result["tag"],
        "num_classes": num_classes,
        "best_val_acc": result["best_val_acc"],
        "best_epoch": result["best_epoch"],
        "test_acc": test_acc,
        "checkpoint": str(stable),
        "image_size": rc.image_size,
        "use_imagenet_norm": rc.use_imagenet_norm,
        "freeze_backbone": rc.freeze_backbone,
        "use_aug": rc.use_aug,
        "elapsed_sec": result["elapsed_sec"],
    }
    save_json(summary, config.RESULTS_DIR / "05_source_summary.json")
    log.info(f"resumen: {summary}")


if __name__ == "__main__":
    main()
