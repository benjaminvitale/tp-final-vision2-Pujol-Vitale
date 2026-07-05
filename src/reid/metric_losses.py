"""metric_losses.py — SupCon and batch-hard Triplet for the loss-comparison study (Stage 3).

Both operate on backbone embeddings and use the batch's labels. They need multiple
positives per identity in a batch → require the PK sampler (src/reid/sampler.py).

- `SupConLoss`       : Khosla et al. 2020. Pulls all same-class samples together,
                       pushes the rest away, over an L2-normalized projection.
- `TripletBatchHard` : Hermans et al. 2017. Per anchor, the hardest positive (farthest
                       same-class) vs the hardest negative (closest other-class); soft
                       (softplus) or hard-margin variant. Euclidean distance.
- `ProjectionHead`   : MLP 2048→2048→128 used ONLY for SupCon training; evaluation
                       always uses the 2048-d backbone (see the brief's guardrail #3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP head (2048→hidden→out), L2-normalized output. Training-only (SupCon)."""

    def __init__(self, in_dim: int = 2048, hidden: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive loss (Khosla et al. 2020). Input: L2-normalized [B, D]."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = feats.device
        b = feats.shape[0]
        sim = feats @ feats.T / self.t                      # [B, B] cosine / τ
        # numerical stability: subtract row max (detached)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)  # same identity
        self_mask = torch.eye(b, device=device)
        pos_mask = pos_mask - self_mask                     # exclude the anchor itself
        logits_mask = 1.0 - self_mask                       # exclude self from denominator

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        pos_count = pos_mask.sum(dim=1)
        # mean log-prob over positives; anchors with no positive contribute 0
        valid = pos_count > 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return -mean_log_prob_pos.mean() if valid.any() else feats.sum() * 0.0


class TripletBatchHard(nn.Module):
    """Batch-hard Triplet (Hermans et al. 2017). Input: backbone embeddings [B, D].

    `margin=None` → soft-margin (softplus); a float → hard margin. Euclidean distance.
    """

    def __init__(self, margin: float | None = None):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # pairwise euclidean distance [B, B]
        dist = torch.cdist(feats, feats, p=2)
        labels = labels.view(-1, 1)
        same = labels == labels.T
        eye = torch.eye(len(labels), dtype=torch.bool, device=feats.device)
        pos_mask = same & ~eye                              # positives (not self)
        neg_mask = ~same
        # hardest positive: max distance among positives; hardest negative: min among negatives
        d_pos = (dist - (~pos_mask).float() * 1e9).max(dim=1).values   # farthest positive
        d_neg = (dist + (~neg_mask).float() * 1e9).min(dim=1).values   # closest negative
        # only anchors that have at least one positive AND one negative count
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        d_pos, d_neg = d_pos[valid], d_neg[valid]
        if d_pos.numel() == 0:
            return feats.sum() * 0.0
        if self.margin is None:
            return F.softplus(d_pos - d_neg).mean()
        return F.relu(d_pos - d_neg + self.margin).mean()
