"""models.py — Builders de VGG16_BN y ResNet-50.

PAPER:
- Transfer learning desde ImageNet.
- Se fine-tunean SOLO las capas fully-connected (backbone convolucional congelado).
- VGG16_BN es el modelo que replica el 98.7%. ResNet-50 es nuestro backbone propio
  (para reutilizar en domain adaptation), y además se prueba con full fine-tune.

Carga de pesos preentrenados sin depender de internet (Kaggle): si `config.PRETRAINED_DIR`
tiene el .pth correspondiente, se carga de ahí (`weights=None` + load_state_dict);
si no, se baja con la API de torchvision.

Diseño para la fase futura de embeddings: `build_model(..., head=False)` (o
`strip_classifier`) devuelve el backbone sin la capa de clasificación, exponiendo
embeddings. No se usa en esta etapa, pero queda preparado.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tvm

import config


def _load_pretrained_state(model_name: str) -> dict:
    """Devuelve el state_dict de ImageNet: local si está, si no lo baja torchvision."""
    fname = config.PRETRAINED_FILES[model_name]
    if config.PRETRAINED_DIR is not None:
        local = Path(config.PRETRAINED_DIR) / fname
        if local.is_file():
            return torch.load(local, map_location="cpu")
    # Fallback: bajar con torchvision (requiere internet).
    weights = {
        "vgg16_bn": tvm.VGG16_BN_Weights.IMAGENET1K_V1,
        "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
    }[model_name]
    return tvm.get_model(model_name, weights=weights).state_dict()


def _apply_pretrained(model: nn.Module, model_name: str, pretrained: bool) -> None:
    if pretrained:
        model.load_state_dict(_load_pretrained_state(model_name))


def build_model(name: str,
                num_classes: int = config.NUM_CLASSES,
                freeze_backbone: bool = config.FREEZE_BACKBONE,
                pretrained: bool = True) -> nn.Module:
    """Construye el modelo con cabeza de `num_classes` salidas.

    Args:
        name: 'vgg16_bn' | 'resnet50'.
        num_classes: nº de clases (PAPER: 268).
        freeze_backbone: si True, congela el backbone convolucional y entrena solo
            las FC (PAPER). Si False, fine-tune completo.
        pretrained: cargar pesos de ImageNet (True) o init aleatoria (False, p/tests).
    """
    name = name.lower()
    if name == "vgg16_bn":
        model = tvm.vgg16_bn(weights=None)
        _apply_pretrained(model, name, pretrained)
        # La última capa del classifier (Linear 4096->1000) -> 4096->num_classes.
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            # PAPER: congelar features (conv), entrenar solo classifier (FC).
            for p in model.features.parameters():
                p.requires_grad = False

    elif name == "resnet50":
        model = tvm.resnet50(weights=None)
        _apply_pretrained(model, name, pretrained)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            # Congelar todo menos la fc.
            for pname, p in model.named_parameters():
                p.requires_grad = pname.startswith("fc.")

    else:
        raise ValueError(f"Modelo no soportado: {name}. Usar uno de {config.MODELS}.")

    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parámetros con requires_grad=True (los que ve el optimizador)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(entrenables, totales)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
