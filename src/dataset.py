"""dataset.py — Dataset de PyTorch + DataLoader leyendo desde los splits JSON.

Los splits se generan una sola vez con `scripts/01_make_splits.py` y se guardan en
`outputs/splits/`. Acá solo se leen (no se re-splitea por corrida → reproducibilidad).

Las rutas en los JSON son RELATIVAS a `config.DATA_DIR`, así el mismo split funciona
en Kaggle y en local sin reescribir paths.

Etapa 2: `make_dataloader` y `make_train_loader` aceptan `data_dir` (default
`config.DATA_DIR`) para poder apuntar a otro dataset —p.ej. `config.CMPD300_DIR`—
reusando el mismo Dataset y los mismos splits JSON.

Diseño preparado para la fase futura de embeddings: `MuzzleDataset` devuelve
(imagen, label) y opcionalmente el path, suficiente para gallery/probe más adelante.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config
from src.utils import load_json


class MuzzleDataset(Dataset):
    """Dataset de imágenes de hocico. Lee entradas {"path", "label"} de un split."""

    def __init__(self, entries: list[dict], transform=None,
                 data_dir: Path = config.DATA_DIR, return_path: bool = False):
        self.entries = entries
        self.transform = transform
        self.data_dir = Path(data_dir)
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        path = self.data_dir / e["path"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = e["label"]
        if self.return_path:
            return img, label, str(e["path"])
        return img, label


def load_split(split_name: str, splits_dir: Path = config.SPLITS_DIR) -> list[dict]:
    """Carga un split ('train' | 'val' | 'test') desde su JSON."""
    path = Path(splits_dir) / f"{split_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Generá los splits primero: "
            f"python scripts/01_make_splits.py"
        )
    return load_json(path)


def _make_loader(ds, *, shuffle: bool, batch_size: int, num_workers: int) -> DataLoader:
    """DataLoader sobre un Dataset ya construido (helper compartido)."""
    # persistent_workers: con 50 épocas evita recrear los workers en cada época
    # (solo aplica si num_workers > 0). No cambia datos ni resultados, solo velocidad.
    extra = {}
    if num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 4
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **extra,
    )


def make_dataloader(entries: list[dict], transform, *, shuffle: bool,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    data_dir: Path = config.DATA_DIR,
                    return_path: bool = False) -> DataLoader:
    """Construye un DataLoader sobre un split ya cargado (un solo transform para todo).

    `data_dir` (Etapa 2): raíz desde la que se resuelven las rutas de las entradas.
    Default = config.DATA_DIR (dataset del paper); pasar config.CMPD300_DIR para CMPD300.
    """
    ds = MuzzleDataset(entries, transform=transform, data_dir=data_dir,
                       return_path=return_path)
    return _make_loader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


def build_augmented_entries(entries: list[dict],
                            cap: int = config.AUG_TARGET_CAP,
                            factor: int = config.AUG_FACTOR,
                            seed: int = 0) -> list[dict]:
    """Entradas EXTRA a sintetizar por clase para agrandar el dataset.

    PAPER: la augmentation "crea imágenes sintéticas y agranda el dataset", sobre todo para
    las clases con pocas imágenes, MANTENIENDO los originales. Por clase se expande hasta
    `min(cap, factor * N_i)`; devuelve solo las copias EXTRA (los originales se agregan
    aparte con transform limpio). El muestreo es determinista por `seed` (reproducibilidad).
    """
    by_label: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_label[e["label"]].append(e)
    rng = random.Random(seed)
    extra: list[dict] = []
    for label, items in sorted(by_label.items()):
        n = len(items)
        n_add = max(0, min(cap, factor * n) - n)
        for _ in range(n_add):
            extra.append(rng.choice(items))
    return extra


def load_augmented_manifest(cache_dir: Path = config.AUG_CACHE_DIR) -> list[dict] | None:
    """Devuelve el manifest de imágenes sintéticas precomputadas, o None si no existe."""
    mp = Path(cache_dir) / "aug_manifest.json"
    return load_json(mp) if mp.is_file() else None


def make_train_loader(entries: list[dict], *, use_aug: bool, clean_tf, aug_tf,
                      seed: int = 0,
                      batch_size: int = config.BATCH_SIZE,
                      num_workers: int = config.NUM_WORKERS,
                      cap: int = config.AUG_TARGET_CAP,
                      factor: int = config.AUG_FACTOR,
                      use_precomputed: bool = True,
                      aug_cache_dir: Path = config.AUG_CACHE_DIR,
                      data_dir: Path = config.DATA_DIR) -> DataLoader:
    """Loader de train.

    - `use_aug=False`: originales con transform limpio (variantes ce / wce).
    - `use_aug=True` (PAPER): originales limpios + copias sintéticas aumentadas → el dataset
      se AGRANDA, no se reemplaza. Así el modelo sigue viendo las imágenes a brillo real
      (evita el mismatch train/test) y gana variedad en las clases con pocas imágenes.

    Para las copias sintéticas, si `use_precomputed` y existe el manifest de
    `scripts/04_precompute_aug.py`, se usan las imágenes FIJAS de disco (transform limpio,
    ya están aumentadas). Si no, se generan online con `aug_tf` (fallback; p.ej. en smoke o
    para datasets sin cache como CMPD300 → pasar use_precomputed=False).

    `data_dir` (Etapa 2): raíz de las rutas de las entradas originales y de las copias
    online. (El cache precomputado sigue resolviéndose contra `aug_cache_dir`.)
    """
    clean_ds = MuzzleDataset(entries, transform=clean_tf, data_dir=data_dir)
    if not use_aug:
        return _make_loader(clean_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    manifest = load_augmented_manifest(aug_cache_dir) if use_precomputed else None
    if manifest is not None:
        # Precomputado: imágenes ya aumentadas en disco → solo resize + ToTensor.
        aug_ds = MuzzleDataset(manifest, transform=clean_tf, data_dir=aug_cache_dir)
    else:
        # Fallback online: aumentar al vuelo (mismo criterio de expansión).
        extra = build_augmented_entries(entries, cap=cap, factor=factor, seed=seed)
        aug_ds = MuzzleDataset(extra, transform=aug_tf, data_dir=data_dir)
    train_ds = ConcatDataset([clean_ds, aug_ds])
    return _make_loader(train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
