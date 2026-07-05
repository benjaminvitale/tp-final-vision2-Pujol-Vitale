"""sampler.py — PK sampler for metric learning (Stage 3 loss comparison).

Each batch = `P` identities × `K` images per identity (batch size P*K). SupCon and
Triplet REQUIRE several positives per identity in a batch; CE and ArcFace tolerate it.
Used in all four conditions so the batch construction is identical (only the loss varies).

Identities with fewer than K images are sampled WITH replacement (declared in the brief).
"""
from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Yield flat indices grouped as P identities × K images per batch.

    `num_batches` defaults to (num_images // (P*K)) so an "epoch" ≈ one pass over the data.
    """

    def __init__(self, labels: list[int], P: int = 16, K: int = 4,
                 num_batches: int | None = None, seed: int = 0):
        self.P, self.K = P, K
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.by_label[int(lab)].append(idx)
        self.labels = list(self.by_label.keys())
        if len(self.labels) < P:
            raise ValueError(f"PKSampler needs >= P={P} identities, got {len(self.labels)}.")
        self.num_batches = num_batches or max(1, len(labels) // (P * K))
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.labels, self.P)
            for lab in chosen:
                pool = self.by_label[lab]
                if len(pool) >= self.K:
                    picks = self.rng.sample(pool, self.K)
                else:
                    picks = [self.rng.choice(pool) for _ in range(self.K)]  # with replacement
                yield from picks

    def __len__(self) -> int:
        return self.num_batches * self.P * self.K
