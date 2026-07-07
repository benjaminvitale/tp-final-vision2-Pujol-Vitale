"""encoders.py — Frozen encoder zoo for unsupervised cross-dataset re-ID (Stage 3).

Every encoder is FROZEN and exposes the same interface: `embed(entries, data_dir) ->
(embeddings [N, D] L2-normalized, labels [N])`, in the SAME order as `entries` so the
caller can reuse the embeddings for both clustering and retrieval without re-embedding.

Only the encoder changes; the pipeline downstream (DBSCAN/HDBSCAN + ARI/NMI + kNN) is
identical for all of them. That is the whole point of the comparison: does specializing
on muzzle help discover identities in a new field, or does a strong generic encoder
group them just as well?

Encoders available now (Phase 0):
- `dinov2`            : DINOv2 ViT (HuggingFace, self-supervised generic). 768-d for -base.
- `imagenet_resnet50`: plain ImageNet ResNet-50 (weak generic floor). 2048-d.
- `resnet50_checkpoint`: our muzzle-trained ResNet-50 (cmpd300_source.pt / ArcFace). 2048-d.

Each encoder uses its OWN native preprocessing (a foundation model is not helped by
forcing someone else's recipe). Only the gallery/probe SPLIT is shared across encoders —
see DEVIATIONS.md. MegaDescriptor is added in Phase 1 alongside these.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.utils import get_device, get_logger

log = get_logger("reid.encoders")


def _resnet50_backbone(model: nn.Module) -> nn.Module:
    """ResNet-50 without the final fc layer → output [B, 2048, 1, 1]."""
    return nn.Sequential(*list(model.children())[:-1])


class Encoder:
    """A frozen backbone + its native preprocessing + a feature-forward function.

    `forward_fn(model, batch) -> [B, D]` isolates the only per-encoder difference in the
    embedding loop (ResNet global-avg-pool vs. ViT CLS token). Everything else — dataset,
    dataloader, L2 normalization, ordering — is shared.
    """

    def __init__(self, model: nn.Module, transform: transforms.Compose,
                 forward_fn, name: str, device: str):
        self.model = model.eval().to(device)
        self.transform = transform
        self.forward_fn = forward_fn
        self.name = name
        self.device = device

    @torch.no_grad()
    def embed(self, entries: list[dict], data_dir: Path,
              batch_size: int = 64, num_workers: int = 2,
              corrupt=None) -> tuple[np.ndarray, np.ndarray]:
        """entries [{path,label}] → (embeddings [N,D] L2-norm, labels [N]) in entry order.

        `corrupt`: optional callable(tensor)->tensor applied to each preprocessed batch
        just before the forward pass (for the input-ablation robustness tests). None = the
        clean baseline.
        """
        ds = MuzzleDataset(entries, transform=self.transform, data_dir=Path(data_dir))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available())
        embs, labs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            if corrupt is not None:
                imgs = corrupt(imgs)
            feats = self.forward_fn(self.model, imgs)
            feats = F.normalize(feats.flatten(1), dim=1)
            embs.append(feats.cpu().numpy())
            labs.append(np.asarray(labels))
        emb = np.concatenate(embs)
        log.info(f"encoder='{self.name}' embedded {emb.shape[0]} imgs → dim {emb.shape[1]}")
        return emb, np.concatenate(labs)


# --------------------------------------------------------------------------- #
# Preprocessing pipelines
# --------------------------------------------------------------------------- #
def _resnet_square_tf(image_size: int, use_imagenet_norm: bool) -> transforms.Compose:
    """Square resize (no crop) — matches the Stage 2 ResNet/ImageNet re-ID baseline."""
    ops: list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if use_imagenet_norm:
        ops.append(transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
    return transforms.Compose(ops)


def _vit_tf() -> transforms.Compose:
    """DINOv2 / ViT native recipe: resize shortest side → center crop → ImageNet norm."""
    return transforms.Compose([
        transforms.Resize(config.VIT_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.VIT_CROP),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def imagenet_resnet50(device: str | None = None) -> Encoder:
    """Plain ImageNet ResNet-50 (no muzzle training). Weak generic floor. 2048-d."""
    device = device or get_device()
    model = build_model("resnet50", num_classes=2, freeze_backbone=False, pretrained=True)
    backbone = _resnet50_backbone(model)
    tf = _resnet_square_tf(224, use_imagenet_norm=True)
    return Encoder(backbone, tf, lambda m, x: m(x), "imagenet_resnet50", device)


def resnet50_checkpoint(ckpt_path: Path, device: str | None = None) -> Encoder:
    """Our muzzle-trained ResNet-50 (e.g. cmpd300_source.pt, arcface). 2048-d.

    Uses the preprocessing the checkpoint was trained with (stored in run_config).
    """
    device = device or get_device()
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model_name = obj.get("model_name", "resnet50") if isinstance(obj, dict) else "resnet50"
    num_classes = obj.get("num_classes", config.NUM_CLASSES) if isinstance(obj, dict) else config.NUM_CLASSES
    if model_name != "resnet50":
        raise ValueError(f"Extractor not supported for {model_name} (use resnet50).")
    model = build_model("resnet50", num_classes=num_classes, freeze_backbone=False, pretrained=False)
    model.load_state_dict(state)
    rc = obj.get("run_config", {}) if isinstance(obj, dict) else {}
    tf = _resnet_square_tf(rc.get("image_size") or config.IMAGE_SIZE_S2,
                           rc.get("use_imagenet_norm", config.USE_IMAGENET_NORM_S2))
    return Encoder(_resnet50_backbone(model), tf, lambda m, x: m(x), ckpt_path.name, device)


def dinov2(model_id: str = config.DINOV2_MODEL, device: str | None = None) -> Encoder:
    """DINOv2 ViT from HuggingFace, frozen. Self-supervised generic baseline.

    Embedding = the pooled CLS token (`pooler_output`), 768-d for dinov2-base. This is
    the critical baseline: a strong generic encoder that never saw a cow.
    """
    device = device or get_device()
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DINOv2 needs `transformers`: pip install transformers") from exc
    model = AutoModel.from_pretrained(model_id)

    def forward_fn(m, x):
        out = m(pixel_values=x)
        # DINOv2 exposes pooler_output (CLS after layernorm); fall back to CLS token.
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]

    return Encoder(model, _vit_tf(), forward_fn, model_id.split("/")[-1], device)


def dinov2_checkpoint(ckpt_path: Path, device: str | None = None) -> Encoder:
    """A DINOv2 ViT fine-tuned on muzzles (scripts/13_train_loss.py --backbone dinov2).

    Loads the fine-tuned AutoModel weights and embeds with the SAME pooled-CLS forward as
    the frozen `dinov2` baseline, so the comparison is apples-to-apples (768-d for -base).
    """
    device = device or get_device()
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model_id = (obj.get("run_config", {}) or {}).get("dinov2_model", config.DINOV2_MODEL)
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DINOv2 needs `transformers`: pip install transformers") from exc
    model = AutoModel.from_pretrained(model_id)
    model.load_state_dict(state)

    def forward_fn(m, x):
        out = m(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]

    return Encoder(model, _vit_tf(), forward_fn, ckpt_path.name, device)


def build_encoder(spec: str, ckpt: Path | None = None, device: str | None = None) -> Encoder:
    """Factory: 'dinov2' | 'imagenet' | 'resnet-ckpt' (needs `ckpt`)."""
    spec = spec.lower()
    if spec == "dinov2":
        return dinov2(device=device)
    if spec == "imagenet":
        return imagenet_resnet50(device=device)
    if spec in ("resnet-ckpt", "checkpoint"):
        if ckpt is None:
            raise ValueError("resnet-ckpt requires --ckpt")
        return resnet50_checkpoint(Path(ckpt), device=device)
    raise ValueError(f"Unknown encoder spec '{spec}' (dinov2|imagenet|resnet-ckpt).")
