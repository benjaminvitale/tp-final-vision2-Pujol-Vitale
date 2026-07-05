"""perturb.py — Controlled input ablations for the encoder robustness tests (Stage 3).

Three deliberate corruptions applied to the model's input tensor (after the encoder's
own preprocessing, so geometry is well-defined). The SAME transform is applied to every
image — this is a controlled ablation to probe the encoder, NOT training augmentation.

- `border_noise` : keep the central muzzle, replace the outer ring with noise.
                   Probes WHERE the signal is (muzzle vs surroundings).
- `rotate`       : rotate the muzzle, with REFLECT padding + center-crop so no black
                   corners are introduced (a black corner is itself a border artifact
                   that would contaminate the border test). Probes pose INVARIANCE.
- `both`         : rotate then border-noise.

Two things get measured downstream (see scripts/12_perturbation_tests.py):
1. Embedding drift: cosine(emb(x), emb(T(x))) — how far the corruption moves the vector
   (label-free invariance measure).
2. Task impact: Rank-1 / ARI when everything is re-embedded through T.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def border_noise(x: torch.Tensor, keep: float = 0.55, sigma: float = 1.0) -> torch.Tensor:
    """Keep the central `keep` fraction (the muzzle); replace the outer ring with N(0,sigma).

    Operates in normalized space, where 0 is the channel mean, so the noise is a plausible
    scale. `keep=0.55` keeps the central ~55% of each side (~30% of the area is muzzle).
    """
    b, c, h, w = x.shape
    h0, h1 = int(h * (1 - keep) / 2), int(h * (1 + keep) / 2)
    w0, w1 = int(w * (1 - keep) / 2), int(w * (1 + keep) / 2)
    out = torch.randn_like(x) * sigma
    out[:, :, h0:h1, w0:w1] = x[:, :, h0:h1, w0:w1]
    return out


def rotate(x: torch.Tensor, angle: float = 20.0) -> torch.Tensor:
    """Rotate by `angle` degrees with reflect padding + center-crop (no black corners)."""
    b, c, h, w = x.shape
    pad = int(max(h, w) * 0.30)
    xp = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    xr = TF.rotate(xp, angle, interpolation=TF.InterpolationMode.BILINEAR)
    return TF.center_crop(xr, [h, w])


def both(x: torch.Tensor, keep: float = 0.55, angle: float = 20.0,
         sigma: float = 1.0) -> torch.Tensor:
    """Rotation followed by border noise."""
    return border_noise(rotate(x, angle=angle), keep=keep, sigma=sigma)


def make_conditions(keep: float = 0.55, angle: float = 20.0, sigma: float = 1.0) -> dict:
    """Return {name: fn(tensor)->tensor} for the three ablations (baseline is no-op)."""
    return {
        "border": lambda x: border_noise(x, keep=keep, sigma=sigma),
        "rotation": lambda x: rotate(x, angle=angle),
        "both": lambda x: both(x, keep=keep, angle=angle, sigma=sigma),
    }


def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Undo Normalize(mean,std) → [0,1] tensor for saving example images."""
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)
