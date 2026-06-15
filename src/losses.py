"""losses.py — Cross-Entropy y Weighted Cross-Entropy.

PAPER (optimización de desbalance):
- Weighted CE: peso por clase w_i = N_max / N_i, con N_i = nº de imágenes de la
  clase i en train y N_max = máximo de esos conteos. El paper asume N_max=70; el
  máximo real del dataset también es 70, así que el empírico coincide (ver
  DEVIATIONS.md). Calculamos N_max EMPÍRICAMENTE desde el split de train.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config


def compute_class_counts(train_entries: list[dict],
                         num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """Conteo de imágenes por clase (N_i) en el split de train."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for e in train_entries:
        counts[e["label"]] += 1
    return counts


def compute_class_weights(train_entries: list[dict],
                          num_classes: int = config.NUM_CLASSES,
                          nmax_override: int | None = config.WCE_NMAX_OVERRIDE) -> torch.Tensor:
    """Pesos de Weighted CE: w_i = N_max / N_i (float32)."""
    counts = compute_class_counts(train_entries, num_classes).float()
    if (counts == 0).any():
        # No debería pasar: el split garantiza ≥1 por clase en train.
        missing = (counts == 0).nonzero().flatten().tolist()
        raise ValueError(f"Clases sin imágenes en train: {missing}. Revisar splits.")
    n_max = float(nmax_override) if nmax_override is not None else float(counts.max())
    return n_max / counts


def build_loss(kind: str,
               train_entries: list[dict] | None = None,
               num_classes: int = config.NUM_CLASSES,
               device: str = "cpu") -> nn.Module:
    """Construye la loss.

    Args:
        kind: 'ce' (Cross-Entropy) | 'wce' (Weighted Cross-Entropy).
        train_entries: requerido para 'wce' (para calcular los pesos por clase).
        device: dónde poner el tensor de pesos.
    """
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "wce":
        if train_entries is None:
            raise ValueError("Weighted CE necesita train_entries para los pesos.")
        weights = compute_class_weights(train_entries, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Loss no soportada: {kind}. Usar 'ce' o 'wce'.")
