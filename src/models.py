"""models.py — Builders de VGG16_BN y ResNet-50.

PAPER:
- Transfer learning desde ImageNet.
- Se fine-tunean SOLO las capas fully-connected (backbone convolucional congelado).
- VGG16_BN es el modelo que replica el 98.7%. ResNet-50 es nuestro backbone propio
  (para reutilizar en domain adaptation), y además se prueba con full fine-tune.

Carga de pesos preentrenados sin depender de internet (Kaggle): si `config.PRETRAINED_DIR`
tiene el .pth correspondiente, se carga de ahí (`weights=None` + load_state_dict);
si no, se baja con la API de torchvision.

Etapa 2: `build_model(..., init_from=<ruta_checkpoint>)` permite inicializar el BACKBONE
desde un checkpoint propio (p.ej. el ResNet-50 ganador de la replicación,
outputs/checkpoints/resnet50_backbone.pt), descartando su cabeza de clasificación. Útil
como warm-start del encoder source de CMPD300. La carga de ImageNet (o init aleatoria) se
hace primero y el backbone se sobrescribe después; la cabeza queda siempre nueva.

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
from src.utils import get_logger


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


def load_backbone_from_checkpoint(model: nn.Module, ckpt_path: str | Path,
                                  skip_prefixes: tuple[str, ...] = ("fc.", "classifier.")
                                  ) -> tuple[int, int]:
    """Carga en `model` los pesos del BACKBONE desde un checkpoint propio.

    Descarta la cabeza de clasificación (`skip_prefixes`) y cualquier tensor cuyo shape
    no coincida (no se pisa nada incompatible). Acepta tanto un checkpoint guardado por
    `train.py` ({"model_state": ...}) como un state_dict crudo.

    Devuelve (n_cargados, n_descartados). No cambia `requires_grad` (el freeze se aplica
    aparte, antes o después).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"init_from: no existe el checkpoint {ckpt_path}. "
            f"¿Corriste la replicación (03_train_resnet.py) y se copió a "
            f"resnet50_backbone.pt?"
        )
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj

    model_sd = model.state_dict()
    to_load, skipped = {}, 0
    for k, v in state.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped += 1
    model.load_state_dict(to_load, strict=False)

    log = get_logger("models.init_from")
    log.info(f"init_from {ckpt_path.name}: backbone cargado ({len(to_load)} tensores), "
             f"{skipped} descartados (cabeza/no-match).")
    return len(to_load), skipped


def build_model(name: str,
                num_classes: int = config.NUM_CLASSES,
                freeze_backbone: bool = config.FREEZE_BACKBONE,
                pretrained: bool = True,
                init_from: str | Path | None = None) -> nn.Module:
    """Construye el modelo con cabeza de `num_classes` salidas.

    Args:
        name: 'vgg16_bn' | 'resnet50'.
        num_classes: nº de clases (PAPER: 268; CMPD300: del label_map).
        freeze_backbone: si True, congela el backbone convolucional y entrena solo
            las FC (PAPER). Si False, fine-tune completo.
        pretrained: cargar pesos de ImageNet (True) o init aleatoria (False, p/tests).
        init_from: ruta a un checkpoint propio para inicializar el BACKBONE (Etapa 2,
            warm-start). Se aplica DESPUÉS de ImageNet/aleatoria y de reemplazar la
            cabeza; la cabeza queda siempre nueva. Si se pasa, podés setear
            pretrained=False (el backbone viene del checkpoint).
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

    # Etapa 2: sobrescribir el backbone con un checkpoint propio (warm-start).
    if init_from:
        load_backbone_from_checkpoint(model, init_from)

    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parámetros con requires_grad=True (los que ve el optimizador)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(entrenables, totales)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
