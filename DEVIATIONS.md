# DEVIATIONS.md — Deviations from the paper

This file records **every** difference between our implementation and the recipe of the
reference paper:

> Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification
> Using Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.

The replication is evaluated by **fidelity**, not just by the number. We keep a clear
separation between "what the paper says" and "what we decided."

---

## Status: dataset matches the paper

Verified in Phase 0 (`scripts/00_inspect_data.py`, file-based count over all 4923 images):

| | Paper | Measured |
|---|---|---|
| Number of classes | 268 | 268 ✓ |
| Total images | 4923 | 4923 ✓ |
| Min images/class | 4 | 4 ✓ |
| Max images/class | 70 | 70 ✓ |
| Mean | — | 18.4 |

**No deviation in the data.** The paper's recipe (`N_max=70`, distribution 4–70) applies as-is.

---

## D1 — Clarification: 8 classes have 4 images (not 4)

**Paper:** mentions "the 4 cows with only 4 images (IDs 2100, 4549, 5355, 5925)."

**Actual:** there are **8** classes with exactly 4 images:
`2100, 3420, 4549, 5208, 5355, 5630, 5925, 8050`. The 4 named in the paper are a
subset of these.

**Impact:** none on the recipe. Only matters when reporting per-class accuracy: check the
**8** classes, not 4, as the candidates that drag down the average. This is not a deviation —
it is a factual correction of the paper's footnote.

---

## D2 — Values not specified by the paper (decided by us)

These are not deviations (the paper does not fix them), but are documented for reproducibility:

- **Batch size:** 32 (reduce if OOM with 300×300 + VGG16_BN).
- **Replication seeds:** original plan `(0, 1, 2, 3, 4)` = 5 runs. **We use 3 seeds
  `(0, 1, 2)`** due to T4 GPU budget (see D3): VGG16_BN at 300×300 takes ~43 min/run
  → 3 variants × 5 seeds ≈ 10.7 h, which doesn't fit together with ResNet within the
  12-hour "Save & Run All" limit. With 3 variants × 3 seeds ≈ 7.5 h it does. There is still
  a mean ± std (3 samples); the number of replicates was our decision, not the paper's.
  **`config.REPLICATE_SEEDS = (0, 1, 2)`** is now the default, so `02_train_vgg.py` without
  `--seeds` fits in a single commit.
- **Split seed:** 42, fixed; splits are saved to disk and reused.
- **DataLoader num_workers:** 4.
- **Weighted CE `N_max`:** weights are computed from the **train** split (no leakage):
  `N_i` = count in train, `N_max` = maximum of those counts (**≈46**, not 70, because
  train ≈65% of the class with 70 images). `N_max` is just a global scaling factor for
  the loss; the relative weight between classes (`N_max/N_i`) is preserved, which is what
  matters. The paper uses the literal 70 (maximum over the full dataset). Override available
  in `config.py` (`WCE_NMAX_OVERRIDE = 70`) to reproduce the paper's exact value and compare.

---

## D3 — Hardware: T4 instead of P100 (does not affect the recipe)

**Paper / plan:** GPU **P100** (used by the paper; `plan.md` §5 preferred it).

**Actual:** the current Kaggle image ships a PyTorch build **without Pascal (sm_60) support**,
so the P100 raises `CUDA error: no kernel image is available for execution` on the first
forward pass. We train on **GPU T4 ×2** (Turing, sm_75), which is within the supported
capabilities of that PyTorch build (sm_70–sm_120). A single T4 (`cuda:0`) is used.

**Impact on results:** none. This only affects the compute device; the recipe
(resolution, normalization, optimizer, losses, epochs, seeds) is unchanged. Low-level
numerical rounding may differ very slightly, but the methodology does not.

---

## D4 — ADDITIVE data augmentation (correcting toward the paper's method)

**Brightness: NO deviation.** The paper states textually: *"the brightness factor was set from
0.2 to 0.5 (minimum = 0, maximum = 1)"*. That is, it intentionally darkens images to 20–50%.
Our `ColorJitter(brightness=(0.2,0.5))` does the same. The value is faithful.

**Bug we had (and fixed):** the paper describes augmentation as *"a technique to
create synthesized images and increase limited datasets"* → **it creates synthetic images and
ENLARGES the dataset, keeping the originals** (especially for classes with few images). Our
first version used the standard PyTorch online augmentation, which **REPLACES** each image
with an augmented version every epoch. With brightness 0.2–0.5, the entire training set was
always dark (measured: ~3.5× darker than val/test) and the model never saw images at real
brightness → systematic train/test mismatch → the `ce_aug` variant **collapsed**
(test ~0.20 vs ~0.96 for `ce`; train 0.83 / val 0.17).

**Fix (faithful):** **additive** augmentation. We keep the originals under a clean transform
(resize + ToTensor) and **add** augmented synthetic copies. Per class, we expand up to
`min(AUG_TARGET_CAP, AUG_FACTOR·N_i)` (= `min(20, 2·N_i)`): classes with few images are
doubled or brought up to 20; classes with ≥20 images are left unchanged. Result: train
**3187 → 4675 images (1.47×)**, 234/268 classes expanded. Implemented in
`dataset.make_train_loader` / `build_augmented_entries`.

Synthetic copies are **precomputed to disk once** (`scripts/04_precompute_aug.py`
→ `outputs/aug_cache/`), consistent with the paper describing it as a *"preprocessing step"*:
the synthetic set is FIXED (does not vary per epoch), reproducible, and inspectable.
If the cache does not exist (e.g., smoke-test), the code falls back to equivalent online
augmentation.

**What is ours vs. the paper's:** the *method* (additive, expand small classes) is the paper's;
the *exact multiplier* `min(20, 2·N_i)` is our decision (the paper does not give the number
of synthetic images per class) — chosen to keep compute cost manageable on the T4.
Override in `config.py` (`AUG_TARGET_CAP`, `AUG_FACTOR`).

---

## D5 — Stage 3: per-encoder native preprocessing (clustering re-ID)

In the unsupervised clustering stage each frozen encoder is fed its **own native
preprocessing**, not a single shared recipe:
- ResNet-50 (ImageNet baseline / our muzzle checkpoints): square resize 224 + ImageNet
  normalization (the same as the Stage 2 re-ID baseline, so numbers stay comparable).
- DINOv2 / ViT foundation encoders: resize shortest side 256 → center-crop 224 → ImageNet
  normalization (their native recipe).

**Why:** a foundation model is evaluated on the input distribution it was trained for;
forcing a foreign preprocessing would handicap it and confound the comparison. What is
held **identical across encoders** is the gallery/probe split and the clustering protocol
— that is what must be controlled for a fair "generic vs specialized" comparison, not the
pixel preprocessing. Values in `config.py` (`VIT_RESIZE`, `VIT_CROP`).

---

## D6 — (reserved)

Record any future deviation here (resolution, normalization, optimizer,
unfreezing backbone, etc.) with its justification **before** applying it.

---

> Historical note: an initial inspection using `unzip -l | grep cattle_[0-9]+` reported
> 8–140 images (mean 36.7). Cause: each path contains the ID twice
> (`.../cattle_0100/cattle_0100_x.jpg`), doubling the count. The file-based count is
> the correct one and matches the paper.
