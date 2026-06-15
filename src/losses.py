"""losses.py — Cross-Entropy y Weighted Cross-Entropy.

PAPER (optimización de desbalance):
- Weighted CE: peso por clase w_i = N_max / N_i, con N_i = nº de imágenes de la
  clase i en train y N_max = máximo de esos conteos. El paper asume N_max=70; el
  máximo real del dataset también es 70, así que el empírico coincide (ver
  DEVIATIONS.md). Calculamos N_max EMPÍRICAMENTE desde el split de train.
"""
from __future__ import annotations

import warnings

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
    """Pesos de Weighted CE: w_i = N_max / N_i (float32).

    En el run real las 268 clases están en train (garantizado por los splits). Si
    alguna clase tiene 0 imágenes (p.ej. al subsetear en un smoke-test), se le asigna
    peso 0 —no aparece como target, así que no afecta la loss— y se emite un warning.
    """
    counts = compute_class_counts(train_entries, num_classes).float()
    nonzero = counts > 0
    if not bool(nonzero.all()):
        n_missing = int((~nonzero).sum())
        warnings.warn(f"{n_missing}/{num_classes} clases sin imágenes en train "
                      f"(peso 0). En el run completo esto NO debería pasar.")
    n_max = float(nmax_override) if nmax_override is not None else float(counts[nonzero].max())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    weights[nonzero] = n_max / counts[nonzero]
    return weights


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
