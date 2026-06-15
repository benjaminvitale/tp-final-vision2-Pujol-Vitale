"""01_make_splits.py — Fase 1: genera y guarda los splits 65/15/20 por imagen.

PAPER: split aleatorio por imagen, 65% train / 15% val / 20% test, con TODAS las
268 clases presentes en cada split. Acá lo hacemos estratificado POR CLASE con una
asignación per-clase que garantiza ≥1 imagen de cada clase en cada split (necesario
para las clases con solo 4 imágenes; ver DEVIATIONS.md). Semilla fija (config.SPLIT_SEED).

Genera en outputs/splits/:
  - label_map.json   : {nombre_carpeta: índice 0..267}
  - train.json / val.json / test.json : listas de {"path": rel_a_DATA_DIR, "label": int}
  - splits_meta.json : provenance (seed, fracciones, conteos, verificación)

Las rutas se guardan RELATIVAS a DATA_DIR para portabilidad (Kaggle/local).
Correr una sola vez; el entrenamiento reusa estos archivos (no re-splitea).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_dirs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def list_images(class_dir: Path) -> list[Path]:
    return sorted([f for f in class_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS])


def split_counts(n: int, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """Asigna (n_train, n_val, n_test) para una clase con n imágenes.

    Garantiza ≥1 en val y test si n lo permite (n>=3) y ≥1 en train siempre,
    de modo que cada clase esté presente en los tres splits.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1  # train + test (sin val); caso no presente en este dataset
    # n >= 3: garantizar al menos 1 en val y 1 en test.
    n_val = max(1, round(val_frac * n))
    n_test = max(1, round(test_frac * n))
    # asegurar que quede al menos 1 para train
    while n - n_val - n_test < 1:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def main() -> int:
    log = get_logger("make_splits")
    data_dir = config.DATA_DIR
    log.info(f"DATA_DIR = {data_dir}")
    if not data_dir.is_dir():
        log.error(f"No existe DATA_DIR: {data_dir}")
        return 1

    config.ensure_output_dirs()
    rng = random.Random(config.SPLIT_SEED)

    class_dirs = list_class_dirs(data_dir)
    label_map = {d.name: i for i, d in enumerate(class_dirs)}

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    per_class_report: dict[str, dict] = {}

    for d in class_dirs:
        label = label_map[d.name]
        imgs = list_images(d)
        # path relativo a DATA_DIR para portabilidad
        rels = [str(f.relative_to(data_dir)) for f in imgs]
        rng.shuffle(rels)  # reshuffle por clase con semilla fija

        n = len(rels)
        n_tr, n_va, n_te = split_counts(n, config.VAL_FRAC, config.TEST_FRAC)
        tr = rels[:n_tr]
        va = rels[n_tr:n_tr + n_va]
        te = rels[n_tr + n_va:]

        train += [{"path": p, "label": label} for p in tr]
        val += [{"path": p, "label": label} for p in va]
        test += [{"path": p, "label": label} for p in te]
        per_class_report[d.name] = {"n": n, "train": len(tr), "val": len(va), "test": len(te)}

    # Mezclar el orden global (no afecta reproducibilidad: misma semilla).
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # ---- Verificaciones ----
    total = len(train) + len(val) + len(test)
    classes_in = lambda split: {e["label"] for e in split}  # noqa: E731
    all_present = (len(classes_in(train)) == len(classes_in(val))
                   == len(classes_in(test)) == config.NUM_CLASSES)

    # ---- Guardar ----
    save_json(label_map, config.SPLITS_DIR / "label_map.json")
    save_json(train, config.SPLITS_DIR / "train.json")
    save_json(val, config.SPLITS_DIR / "val.json")
    save_json(test, config.SPLITS_DIR / "test.json")

    meta = {
        "seed": config.SPLIT_SEED,
        "fracs": {"train": config.TRAIN_FRAC, "val": config.VAL_FRAC, "test": config.TEST_FRAC},
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": total},
        "frac_real": {
            "train": round(len(train) / total, 4),
            "val": round(len(val) / total, 4),
            "test": round(len(test) / total, 4),
        },
        "num_classes": config.NUM_CLASSES,
        "all_classes_in_each_split": all_present,
        "per_class": per_class_report,
    }
    save_json(meta, config.SPLITS_DIR / "splits_meta.json")

    # ---- Reporte ----
    print("\n" + "=" * 60)
    print("SPLITS GENERADOS — Fase 1")
    print("=" * 60)
    print(f"Total imágenes      : {total} (esperado {config.EXPECTED_IMAGES})")
    print(f"  train             : {len(train)} ({meta['frac_real']['train']:.1%})")
    print(f"  val               : {len(val)} ({meta['frac_real']['val']:.1%})")
    print(f"  test              : {len(test)} ({meta['frac_real']['test']:.1%})")
    print(f"Clases en train/val/test: "
          f"{len(classes_in(train))}/{len(classes_in(val))}/{len(classes_in(test))}")
    print(f"[{'OK' if all_present else 'FALLO'}] las 268 clases están en los 3 splits")
    print(f"[{'OK' if total == config.EXPECTED_IMAGES else 'FALLO'}] "
          f"total imágenes = {config.EXPECTED_IMAGES}")
    print(f"\nGuardado en: {config.SPLITS_DIR}")
    print("=" * 60)
    return 0 if (all_present and total == config.EXPECTED_IMAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
