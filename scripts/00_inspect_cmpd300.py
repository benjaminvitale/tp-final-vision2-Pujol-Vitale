"""00_inspect_cmpd300.py — Inspección de CMPD300 + generación de splits (Fase 5).

CMPD300 ya viene splitteado en carpetas (train/val/test → subcarpeta por ID), así que
NO se re-splittea (no hace falta 01_make_splits para este dataset): este script

  1. recorre <CMPD300_DIR>/{train,val,test}/<ID>/*.JPG,
  2. reporta nº de clases, imágenes por split, min/max/media por clase, IDs faltantes
     en val/test y (opcional) imágenes corruptas,
  3. construye el label_map (carpeta→entero, por carpetas PRESENTES, no asumiendo 1..N),
  4. escribe outputs/splits_cmpd300/{train,val,test}.json + label_map.json,
     en el MISMO formato {"path","label"} que lee src/dataset.py (rutas RELATIVAS a
     config.CMPD300_DIR → el mismo split sirve en local y en Kaggle).

Uso:
    python scripts/00_inspect_cmpd300.py                  # reporte + genera splits
    python scripts/00_inspect_cmpd300.py --check-corrupt  # además verifica ilegibles
    python scripts/00_inspect_cmpd300.py --no-write        # solo reporta, no escribe
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# permitir correr desde cualquier cwd (root del proyecto en sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils import get_logger, save_json

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # case-insensitive (CMPD300 usa .JPG)


def list_images(d: Path) -> list[Path]:
    """Imágenes (por extensión, case-insensitive) dentro de una carpeta."""
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def list_class_dirs(split_dir: Path) -> list[str]:
    """Nombres de subcarpetas (IDs) presentes en un split."""
    if not split_dir.is_dir():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def scan(cmpd_dir: Path) -> dict:
    """Recorre los tres splits y devuelve, por split, {ID: [paths relativas]}."""
    per_split: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS:
        split_dir = cmpd_dir / split
        by_class: dict[str, list[str]] = {}
        for cls in list_class_dirs(split_dir):
            imgs = list_images(split_dir / cls)
            by_class[cls] = [p.relative_to(cmpd_dir).as_posix() for p in imgs]
        per_split[split] = by_class
    return per_split


def stats(by_class: dict[str, list[str]]) -> dict:
    counts = [len(v) for v in by_class.values()]
    n_imgs = sum(counts)
    return {
        "n_classes": len(by_class),
        "n_images": n_imgs,
        "min_per_class": min(counts) if counts else 0,
        "max_per_class": max(counts) if counts else 0,
        "mean_per_class": round(n_imgs / len(by_class), 2) if by_class else 0.0,
    }


def check_corrupt(cmpd_dir: Path, per_split: dict, log) -> list[str]:
    """Intenta abrir cada imagen; devuelve las rutas ilegibles."""
    from PIL import Image
    bad: list[str] = []
    total = 0
    for split in SPLITS:
        for rels in per_split[split].values():
            for rel in rels:
                total += 1
                try:
                    with Image.open(cmpd_dir / rel) as im:
                        im.convert("RGB").load()
                except Exception as e:  # noqa: BLE001
                    bad.append(rel)
                    log.warning(f"corrupta: {rel} ({e})")
    log.info(f"chequeadas {total} imágenes | corruptas: {len(bad)}")
    return bad


def build_label_map(per_split: dict, log) -> dict[str, int]:
    """carpeta→entero 0..N-1, por clases PRESENTES en train (canónico)."""
    train_ids = set(per_split["train"].keys())
    all_ids = set().union(*(set(per_split[s].keys()) for s in SPLITS))

    # toda clase de val/test debería estar en train (si no, no se puede aprender)
    not_in_train = sorted(all_ids - train_ids)
    if not_in_train:
        log.warning(f"⚠ {len(not_in_train)} clases en val/test NO están en train "
                    f"(no se podrían aprender): {not_in_train}")

    classes = sorted(train_ids)  # numeración por carpetas presentes en train
    return {cls: i for i, cls in enumerate(classes)}


def to_entries(by_class: dict[str, list[str]], label_map: dict[str, int]) -> list[dict]:
    """{ID: [paths]} → [{'path','label'}], saltando clases sin label (no en train)."""
    entries: list[dict] = []
    for cls, rels in sorted(by_class.items()):
        if cls not in label_map:
            continue
        lbl = label_map[cls]
        entries += [{"path": rel, "label": lbl} for rel in rels]
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspección + splits de CMPD300.")
    ap.add_argument("--check-corrupt", action="store_true",
                    help="verificar que cada imagen se pueda abrir (más lento).")
    ap.add_argument("--no-write", action="store_true",
                    help="solo reportar, no escribir JSON.")
    args = ap.parse_args()

    log = get_logger("inspect.cmpd300")
    cmpd_dir = config.CMPD300_DIR
    log.info(f"CMPD300_DIR: {cmpd_dir}  (existe: {cmpd_dir.is_dir()})")
    if not (cmpd_dir / "train").is_dir():
        log.error("No encuentro <CMPD300_DIR>/train. Revisá config.CMPD300_DIR "
                  "o seteá CMPD300_DATA_DIR.")
        sys.exit(1)

    per_split = scan(cmpd_dir)

    # ---- Reporte por split ----
    report: dict = {"cmpd300_dir": str(cmpd_dir), "splits": {}}
    for split in SPLITS:
        s = stats(per_split[split])
        report["splits"][split] = s
        log.info(f"[{split:5s}] clases={s['n_classes']:4d} imgs={s['n_images']:5d} "
                 f"min={s['min_per_class']} max={s['max_per_class']} "
                 f"media={s['mean_per_class']}")

    # ---- Clases faltantes entre splits ----
    train_ids = set(per_split["train"].keys())
    for split in ("val", "test"):
        missing = sorted(train_ids - set(per_split[split].keys()))
        report["splits"][split]["missing_vs_train"] = missing
        if missing:
            log.warning(f"[{split}] le faltan {len(missing)} IDs que sí están en train: "
                        f"{missing}")

    # ---- label_map + num_classes ----
    label_map = build_label_map(per_split, log)
    num_classes = len(label_map)
    report["num_classes"] = num_classes
    total_imgs = sum(report["splits"][s]["n_images"] for s in SPLITS)
    report["total_images"] = total_imgs
    log.info(f"=> num_classes (carpetas en train) = {num_classes} | "
             f"total imágenes = {total_imgs}")

    # ---- Corruptas (opcional) ----
    if args.check_corrupt:
        report["corrupt"] = check_corrupt(cmpd_dir, per_split, log)

    # ---- Escribir splits + label_map + reporte ----
    if not args.no_write:
        config.CMPD300_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            entries = to_entries(per_split[split], label_map)
            save_json(entries, config.CMPD300_SPLITS_DIR / f"{split}.json")
            log.info(f"escrito {split}.json ({len(entries)} entradas)")
        save_json(label_map, config.CMPD300_SPLITS_DIR / "label_map.json")
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(report, config.RESULTS_DIR / "00_inspect_cmpd300.json")
        log.info(f"label_map.json + reporte escritos en {config.CMPD300_SPLITS_DIR} "
                 f"y {config.RESULTS_DIR}")
    else:
        log.info("--no-write: no se escribió nada.")

    log.info("OK.")


if __name__ == "__main__":
    main()
