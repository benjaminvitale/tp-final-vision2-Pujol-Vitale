"""embeddings.py — Extractor de embeddings (Fase 6).

Expone el vector del global average pool (**2048-d en ResNet-50**), **L2-normalizado**
(coseno = producto punto). Dos constructores:

- `from_checkpoint(ckpt)`: tu encoder entrenado (p.ej. cmpd300_source.pt), con el
  preprocesamiento con el que se entrenó (guardado en el run_config).
- `from_imagenet()`: ResNet-50 de ImageNet PURO (sin nada de hocico), como línea de base
  tonta para chequear cuánto del rendimiento es "gratis" (fotos parecidas) y cuánto aporta
  el encoder de hocico.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.transforms import build_transforms
from src.utils import get_device, get_logger


def _resnet50_backbone(model: nn.Module) -> nn.Module:
    """ResNet-50 sin la fc final → salida [B, 2048, 1, 1]."""
    return nn.Sequential(*list(model.children())[:-1])


class EmbeddingExtractor:
    """Envuelve un backbone y produce embeddings L2-norm de un conjunto de entries."""

    def __init__(self, backbone: nn.Module, image_size: int, use_imagenet_norm: bool,
                 device: str, name: str = "encoder"):
        self.device = device
        self.name = name
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm
        self.backbone = backbone.eval().to(device)
        self.tf = build_transforms(train=False, image_size=image_size,
                                   use_imagenet_norm=use_imagenet_norm)
        self.log = get_logger("reid.embeddings")
        self.log.info(f"encoder='{name}' | image_size={image_size} | "
                      f"imagenet_norm={use_imagenet_norm} | device={device}")

    # ---- constructores ----
    @classmethod
    def from_checkpoint(cls, ckpt_path: Path = config.CHECKPOINTS_DIR / "cmpd300_source.pt",
                        device: str | None = None) -> "EmbeddingExtractor":
        device = device or get_device()
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint {ckpt_path}. ¿Corriste la Fase 5?")
        obj = torch.load(ckpt_path, map_location="cpu")
        state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
        model_name = obj.get("model_name", "resnet50") if isinstance(obj, dict) else "resnet50"
        num_classes = obj.get("num_classes", config.NUM_CLASSES) if isinstance(obj, dict) else config.NUM_CLASSES
        if model_name != "resnet50":
            raise ValueError(f"Extractor no soportado para {model_name} (usar resnet50).")
        model = build_model("resnet50", num_classes=num_classes,
                            freeze_backbone=False, pretrained=False)
        model.load_state_dict(state)
        rc = obj.get("run_config", {}) if isinstance(obj, dict) else {}
        return cls(_resnet50_backbone(model),
                   rc.get("image_size") or config.IMAGE_SIZE_S2,
                   rc.get("use_imagenet_norm", config.USE_IMAGENET_NORM_S2),
                   device, name=ckpt_path.name)

    @classmethod
    def from_imagenet(cls, device: str | None = None) -> "EmbeddingExtractor":
        device = device or get_device()
        # ResNet-50 de ImageNet puro (backbone preentrenado, cabeza descartada).
        model = build_model("resnet50", num_classes=2, freeze_backbone=False, pretrained=True)
        return cls(_resnet50_backbone(model), 224, True, device, name="imagenet_resnet50")

    # ---- embeddings ----
    @torch.no_grad()
    def embed(self, entries: list[dict], data_dir: Path,
              batch_size: int = 64, num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """entries [{path,label}] → (embeddings [N,2048] L2-norm, labels [N])."""
        ds = MuzzleDataset(entries, transform=self.tf, data_dir=Path(data_dir))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available())
        embs, labs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            f = self.backbone(imgs).flatten(1)
            f = F.normalize(f, dim=1)
            embs.append(f.cpu().numpy())
            labs.append(np.asarray(labels))
        return np.concatenate(embs), np.concatenate(labs)
