"""transforms.py — Preprocessing and data augmentation (paper recipe).

PAPER:
- Resolution 300×300.
- Normalization: intensities to [0,1] (ToTensor already does this). No ImageNet mean/std.
- Data augmentation (train only): horizontal flip, brightness (factor 0.2–0.5),
  random rotation [-15, +15], Gaussian blur with kernel ∈ {1,3,5}.

ImageNet normalization is left as a configurable flag (default OFF) for experiments.
"""
from __future__ import annotations

from torchvision import transforms

import config


def build_transforms(train: bool,
                     image_size: int = config.IMAGE_SIZE,
                     use_imagenet_norm: bool = config.USE_IMAGENET_NORM) -> transforms.Compose:
    """Return the transform pipeline.

    Args:
        train: if True, includes the paper's data augmentation; if False, the
            deterministic val/test pipeline (resize + ToTensor only).
        image_size: side of the square input (PAPER: 300).
        use_imagenet_norm: if True, appends Normalize(mean, std) from ImageNet
            (OUR CHOICE, default OFF; the paper uses raw [0,1]).
    """
    ops: list = [transforms.Resize((image_size, image_size))]

    if train:
        # PAPER: augmentation on train only.
        ops += _aug_ops()

    # ToTensor: PIL [0,255] → tensor [0,1]. This is the paper's normalization.
    ops.append(transforms.ToTensor())

    if use_imagenet_norm:
        ops.append(transforms.Normalize(mean=config.IMAGENET_MEAN,
                                        std=config.IMAGENET_STD))

    return transforms.Compose(ops)


def _aug_ops() -> list:
    """Paper augmentation ops (flip / brightness / rotation / blur), PIL level.

    Shared between `build_transforms(train=True)` and `build_pil_aug_transform` (precompute)
    so that augmentation is defined in exactly one place.
    """
    blur_choices = [transforms.GaussianBlur(kernel_size=k)
                    for k in config.AUG_BLUR_KERNELS]
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS),
        transforms.RandomRotation(config.AUG_ROTATION_DEG),
        transforms.RandomChoice(blur_choices),
    ]


def build_strong_train_transform(image_size: int = config.IMAGE_SIZE_S2,
                                 use_imagenet_norm: bool = config.USE_IMAGENET_NORM_S2
                                 ) -> transforms.Compose:
    """Strong augmentation FIXED across the 4-loss comparison (Stage 3 brief).

    Targets the fragmentation causes the robustness probe found (rotation sensitivity,
    context/background dependence): RandomResizedCrop varies how much surround is visible,
    RandomRotation(±30) attacks pose sensitivity, ColorJitter for lighting, RandomErasing
    forces not relying on any fixed region (incl. background), plus horizontal flip.
    Identical in all four conditions — the ONLY variable is the loss.
    """
    ops: list = [
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
    ]
    if use_imagenet_norm:
        ops.append(transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
    ops.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)))
    return transforms.Compose(ops)


def build_pil_aug_transform(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Resize + augmentation, WITHOUT ToTensor → returns a PIL image.

    Used to precompute synthetic images to disk (scripts/04_precompute_aug.py):
    applied once and the result is saved as a file.
    """
    return transforms.Compose([transforms.Resize((image_size, image_size)), *_aug_ops()])
