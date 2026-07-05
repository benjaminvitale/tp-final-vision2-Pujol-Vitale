"""phash.py — perceptual-hash near-duplicate removal for the eval set (Stage 3).

The Zenodo session gate showed ~1 capture session per animal, so a cross-session split is
not possible. Instead we remove near-identical frames (burst twins) with a perceptual hash
BEFORE evaluating, so clustering/retrieval cannot cheat by matching pixel-copies.

Uses a dependency-free difference hash (dHash): resize to 9×8 grayscale, compare adjacent
columns → 64-bit signature. Two frames are near-duplicates if their Hamming distance ≤
`threshold`. Dedup is done WITHIN each identity (different cows are never near-dups).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def dhash(path: Path, hash_size: int = 8) -> int:
    """64-bit difference hash of an image (grayscale, resize (hash_size+1)×hash_size)."""
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    px = list(img.getdata())
    w = hash_size + 1
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * w + col]
            right = px[row * w + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_entries(entries: list[dict], data_dir: Path, threshold: int = 6) -> tuple[list[dict], dict]:
    """Drop near-duplicate frames WITHIN each identity (dHash Hamming ≤ threshold).

    Greedy: keep an image only if it is far enough from every already-kept image of the
    same identity. Returns (kept_entries, info). Deterministic (input order preserved).
    """
    by_label: dict[int, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)

    kept: list[dict] = []
    n_dropped = 0
    for _lab, items in sorted(by_label.items()):
        kept_hashes: list[int] = []
        for e in items:
            try:
                h = dhash(Path(data_dir) / e["path"])
            except Exception:  # noqa: BLE001 — unreadable image: keep it, don't crash
                kept.append(e)
                continue
            if any(hamming(h, kh) <= threshold for kh in kept_hashes):
                n_dropped += 1
                continue
            kept_hashes.append(h)
            kept.append(e)
    info = {"n_input": len(entries), "n_kept": len(kept), "n_dropped": n_dropped,
            "threshold": threshold}
    return kept, info
