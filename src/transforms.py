"""transforms.py — Preprocesamiento y data augmentation (receta del paper).

PAPER:
- Resolución 300x300.
- Normalización: intensidades a [0,1] (ToTensor ya lo hace). NO mean/std de ImageNet.
- Data augmentation (solo en train): flip horizontal, brillo (factor 0.2–0.5),
  rotación aleatoria [-15, +15], blur gaussiano con kernel ∈ {1,3,5}.

La normalización ImageNet queda como flag configurable (default OFF) para experimentar.
"""
from __future__ import annotations

from torchvision import transforms

import config


def build_transforms(train: bool,
                     image_size: int = config.IMAGE_SIZE,
                     use_imagenet_norm: bool = config.USE_IMAGENET_NORM) -> transforms.Compose:
    """Devuelve el pipeline de transforms.

    Args:
        train: si True incluye la data augmentation del paper; si False es el
            pipeline determinista de val/test (solo resize + ToTensor).
        image_size: lado del cuadrado de entrada (PAPER: 300).
        use_imagenet_norm: si True agrega Normalize(mean,std) de ImageNet
            (NOSOTROS, default OFF; el paper usa [0,1] crudo).
    """
    ops: list = [transforms.Resize((image_size, image_size))]

    if train:
        # PAPER: augmentation solo en train.
        blur_choices = [transforms.GaussianBlur(kernel_size=k)
                        for k in config.AUG_BLUR_KERNELS]
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS),
            transforms.RandomRotation(config.AUG_ROTATION_DEG),
            transforms.RandomChoice(blur_choices),
        ]

    # ToTensor: PIL [0,255] -> tensor [0,1]. Esta es la normalización del paper.
    ops.append(transforms.ToTensor())

    if use_imagenet_norm:
        ops.append(transforms.Normalize(mean=config.IMAGENET_MEAN,
                                        std=config.IMAGENET_STD))

    return transforms.Compose(ops)
