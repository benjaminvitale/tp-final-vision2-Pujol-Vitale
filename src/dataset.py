"""dataset.py — Dataset de PyTorch + DataLoader leyendo desde los splits JSON.

Los splits se generan una sola vez con `scripts/01_make_splits.py` y se guardan en
`outputs/splits/`. Acá solo se leen (no se re-splitea por corrida → reproducibilidad).

Las rutas en los JSON son RELATIVAS a `config.DATA_DIR`, así el mismo split funciona
en Kaggle y en local sin reescribir paths.

Diseño preparado para la fase futura de embeddings: `MuzzleDataset` devuelve
(imagen, label) y opcionalmente el path, suficiente para gallery/probe más adelante.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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


def make_dataloader(entries: list[dict], transform, *, shuffle: bool,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    return_path: bool = False) -> DataLoader:
    """Construye un DataLoader sobre un split ya cargado."""
    ds = MuzzleDataset(entries, transform=transform, return_path=return_path)
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
