# Individual Cattle Identification by Muzzle Image

Final project for **Advanced Computer Vision** (Universidad de San Andrés).

The project has two stages. The first replicates the paper by Li, Erickson & Xiong (2022)
on closed-set cattle identification from muzzle images. The second builds on that foundation
a re-identification system and analyzes whether knowledge learned from muzzles transfers to
another domain (bovine faces) and to another muzzle dataset.

---

## Stages and status

### Stage 1 — Replication (complete ✅)

Replication of **Li, Erickson & Xiong (2022)**, *Individual Beef Cattle Identification Using
Muzzle Images and Deep Learning Techniques*, Animals 12(11):1453.

- **Task:** closed-set classification. 268 cattle = 268 classes. Given a muzzle image, predict the individual.
- **Dataset:** Zenodo Muzzle DB — 268 cattle, 4923 muzzle images.
- **Main model (replicates the paper):** VGG16_BN — best test accuracy **96.99%** (`ce_aug`, 3 seeds; 95.98–96.99% across variants; target ~96–98%+, paper reports 98.7%).
- **Own backbone:** ResNet-50, trained in parallel as the starting point for Stage 2.
- Deviations from the paper are documented in [`DEVIATIONS.md`](DEVIATIONS.md).

### Stage 2 — Re-identification and domain adaptation analysis (complete ✅)

- **Task:** open-set re-identification. ResNet-50 encoder trained on CMPD300 (muzzles);
  gallery/probe with disjoint identities, Rank-1 and mAP metrics.
- **Source dataset:** CMPD300 — bovine muzzle dataset with pre-existing splits.
- **Target domain:** bovine face dataset (Ahmed) and Zenodo muzzle dataset (cross-dataset).
- **Main finding:** a plain ImageNet ResNet-50 (no cattle training) matches or beats **every**
  muzzle-specialized encoder tried — classification (CE), metric learning (ArcFace), and
  domain-adversarial adaptation (DANN, including a warm-started variant). Specializing on the
  source dataset transfers *worse*, not better. Conclusion: in these datasets muzzle biometrics
  do **not** transfer cross-domain; the limit is the data/domain, not the model. Full evidence in
  [`RESULTS_STAGE2.md`](RESULTS_STAGE2.md); Grad-CAM over cosine similarity helps explain why.

---

## Repository structure

```
tp-final-vision2-Pujol-Vitale/
├── config.py                     # SINGLE source of truth for paths, hyperparameters, seeds
├── requirements.txt              # pinned dependencies
├── DEVIATIONS.md                 # all deviations from the paper's recipe, documented
├── RESULTS_STAGE2.md             # Stage 2 evidence: re-ID cross-domain + domain adaptation
│
├── src/                          # reusable logic (not scripts, not notebooks)
│   ├── dataset.py                # MuzzleDataset + make_dataloader
│   ├── transforms.py             # preprocessing and augmentation pipelines
│   ├── models.py                 # build_model() — VGG16_BN and ResNet-50
│   ├── losses.py                 # CE and Weighted Cross-Entropy
│   ├── train.py                  # training loop (RunConfig + run_epoch)
│   ├── evaluate.py               # test evaluation: global + per-class accuracy
│   ├── utils.py                  # seeds, logging, I/O, get_device
│   └── reid/
│       ├── embeddings.py         # EmbeddingExtractor: backbone → 2048-d L2-norm vector
│       ├── eval_reid.py          # gallery/probe metrics: Rank-1, Rank-5, mAP
│       └── reid_dataset.py       # entries + split_gallery_probe (random or by session)
│
├── scripts/                      # per-phase execution, numbered in order
│   ├── 00_inspect_data.py        # Stage 1: validate Zenodo dataset (sanity checks)
│   ├── 00_inspect_cmpd300.py     # Stage 2: validate CMPD300 + generate split JSONs
│   ├── 01_make_splits.py         # Stage 1: stratified 65/15/20 split (Zenodo)
│   ├── 02_train_vgg.py           # Stage 1: VGG16_BN replication (4 variants × N seeds)
│   ├── 03_train_resnet.py        # Stage 1: ResNet-50 backbone (freeze + finetune)
│   ├── 04_precompute_aug.py      # Stage 1: additive augmentation precomputed to disk
│   ├── 05_train_source.py        # Stage 2: ResNet-50 encoder on CMPD300
│   ├── 06_eval_reid.py           # Stage 2: re-ID harness + ImageNet baseline
│   ├── 07_train_arcface.py       # Stage 2: ArcFace metric-learning encoder (CMPD300)
│   ├── 08_train_dann.py          # Stage 2: DANN domain-adversarial (source→target)
│   ├── crop_muzzles.py           # Stage 2: zero-shot muzzle crop from faces (GroundingDINO)
│   ├── download_zenodo.py        # utility: download Zenodo dataset
│   └── kaggle_upload.py          # utility: upload datasets to Kaggle via kagglehub
│
├── notebooks/                    # orchestrators for Colab/Kaggle with GPU
│   ├── kaggle_runner.ipynb       # Stage 1 full run on Kaggle (phases 0→4)
│   ├── colab_runner.ipynb        # Stage 1 full run on Colab Pro (phases 0→4)
│   ├── colab_fase5_source.ipynb  # Stage 2: train encoder on CMPD300
│   ├── colab_fase6_reid.ipynb    # Stage 2: re-ID harness + muzzle→face gap (Ahmed)
│   ├── colab_gap_muzzle.ipynb    # Stage 2: cross-dataset muzzle→muzzle gap (Zenodo)
│   ├── colab_gradcam_reid.ipynb  # Grad-CAM over cosine similarity of embeddings
│   └── gradcam_runner.ipynb      # visual test: VGG16_BN Grad-CAM on Ahmed faces
│
└── outputs/                      # all generated files (large files gitignored, splits versioned)
    ├── splits/                   # Zenodo splits: train/val/test.json + label_map.json
    ├── splits_cmpd300/           # CMPD300 splits: same format
    ├── checkpoints/              # saved weights (.pt): resnet50_backbone.pt, cmpd300_source.pt
    ├── results/                  # per-run metric CSVs and JSONs
    ├── logs/                     # training logs per run
    └── aug_cache/                # precomputed synthetic images (additive augmentation)
```

---

## Requirements and installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies (pinned in `requirements.txt`):

| Library | Version |
|---|---|
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| scikit-learn | 1.4.2 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| Pillow | 10.3.0 |
| tqdm | 4.66.4 |

> **Kaggle/Colab:** base images already include torch and torchvision. Do not reinstall torch;
> only install missing pure-Python libraries.

---

## Datasets and path configuration

Datasets are **not included in the repository**. They are hosted on Kaggle (Stage 1) or Google
Drive (Stage 2).

### Stage 1 dataset — Zenodo Muzzle DB

268 cattle, 4923 muzzle images. Zenodo record 6324361.

`config.py` resolves the path automatically in this order:

1. Environment variable `CATTLE_DATA_DIR` (manual override).
2. Kaggle: `/kaggle/input/<slug>/BeefCattle_Muzzle_Individualized`.
3. Local: `data_local/BeefCattle_Muzzle_Individualized` (for smoke-tests).

```bash
# local extraction for smoke-test
unzip -q BeefCattle_Muzzle_database.zip -d data_local
python config.py       # prints resolved DATA_DIR and full configuration
```

To upload to Kaggle (automated via kagglehub):

```bash
pip install kagglehub
export KAGGLE_USERNAME=your_username KAGGLE_KEY=xxxxxxxx   # kaggle.com → Settings → API
python scripts/kaggle_upload.py --user your_username        # uploads images + ImageNet weights
```

**Pretrained ImageNet weights (offline on Kaggle):** download once and upload as a Kaggle Dataset:

```bash
curl -L -O https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
curl -L -O https://download.pytorch.org/models/resnet50-0676ba61.pth
```

`config.py` detects them by filename. Override with `CATTLE_PRETRAINED_DIR`.

### Stage 2 dataset — CMPD300 (source)

Bovine muzzle dataset with pre-existing splits (`train/`, `val/`, `test/`). Not re-split;
`scripts/00_inspect_cmpd300.py` generates the split JSONs in the format read by `src/dataset.py`.

Path resolved from `CMPD300_DATA_DIR` (env) → `/kaggle/input/.../Baseline` → `datasets/Baseline/`.

### Stage 2 dataset — Ahmed (target, bovine faces)

Bovine face dataset (~13.9 GB). A subset is extracted directly in the Colab notebooks.
Not configured in `config.py`; notebooks handle it from Google Drive.
Reference: Cows Frontal Face Dataset — Zenodo record 10535934. Muzzles cropped from these faces
with `scripts/crop_muzzles.py` (GroundingDINO zero-shot).

---

## Running each phase

> Always run in order. Do not proceed if the Phase 0 sanity checks fail.

### Stage 1 — Paper replication

**Phase 0 — Dataset inspection**

```bash
python scripts/00_inspect_data.py                  # report + sanity checks (268 classes, 4923 imgs)
python scripts/00_inspect_data.py --check-corrupt  # also verifies unreadable images (slow)
```

Verifies: 268 classes, 4923 images, 4–70 per class (mean 18.4). Saves report to
`outputs/results/00_inspect_report.json`.

**Phase 1 — Splits**

```bash
python scripts/01_make_splits.py
```

Generates a class-stratified 65/15/20 split with a fixed seed (`config.SPLIT_SEED = 42`).
Actual result: 64.7% / 15.2% / 20.0%. Saves splits to `outputs/splits/` and reuses them
across all runs (no re-splitting).

**Phase 2b — Augmentation precomputation (optional, recommended)**

```bash
python scripts/04_precompute_aug.py           # generates if not present
python scripts/04_precompute_aug.py --force   # regenerates
```

Generates synthetic augmentation images **once** and saves them to `outputs/aug_cache/`.
Training picks them up automatically if the cache exists; otherwise falls back to equivalent
online augmentation.

**Phase 3 — VGG16_BN replication**

```bash
python scripts/02_train_vgg.py                     # 4 variants × 3 seeds (config), 50 epochs
python scripts/02_train_vgg.py --seeds 0           # single seed (fast)
python scripts/02_train_vgg.py --epochs 50 --seeds 0 1 2
python scripts/02_train_vgg.py --smoke             # fast pipeline: 1 seed, 2 epochs, subset
```

Variants trained:

| Variant | Loss | Augmentation |
|---|---|---|
| `ce` | Cross-Entropy | No |
| `ce_aug` | Cross-Entropy | Yes (additive) |
| `wce` | Weighted CE | No |
| `wce_aug` | Weighted CE | Yes (additive) |

> **Note:** `plan.md` originally defined 3 variants (ce / ce_aug / wce); the implementation
> in `02_train_vgg.py` has 4 (wce_aug was added).

Generates a summary table at `outputs/results/02_vgg_summary.json` and a per-class CSV for
the best run.

**Phase 4 — ResNet-50 backbone**

```bash
python scripts/03_train_resnet.py                  # freeze + finetune, seed 0, 50 epochs
python scripts/03_train_resnet.py --modes freeze   # paper mode only (frozen backbone)
python scripts/03_train_resnet.py --aug --loss wce # stronger backbone
python scripts/03_train_resnet.py --smoke          # fast pipeline on CPU
```

Runs two modes: `freeze` (FC only, as in the paper) and `finetune` (full fine-tuning). The
best run by val accuracy is copied to `outputs/checkpoints/resnet50_backbone.pt`.

### Stage 2 — Re-identification

**Phase 5 — CMPD300 inspection + encoder training**

```bash
# 1. Inspection and split generation
python scripts/00_inspect_cmpd300.py                   # report + writes JSONs to outputs/splits_cmpd300/
python scripts/00_inspect_cmpd300.py --check-corrupt   # also verifies unreadable images

# 2. Encoder training
python scripts/05_train_source.py                      # freeze (FC only), 50 epochs, seed 0
python scripts/05_train_source.py --no-freeze          # full fine-tuning
python scripts/05_train_source.py --aug                # + online augmentation in train
python scripts/05_train_source.py --smoke              # fast pipeline on CPU
```

Trains ResNet-50 on CMPD300 (224 px, ImageNet normalization). Best checkpoint saved to
`outputs/checkpoints/cmpd300_source.pt`.

**Phase 6 — Re-identification harness**

```bash
# Intra-CMPD300 sanity check (validates the harness; result NOT reportable due to leakage)
python scripts/06_eval_reid.py --source-dir /path/to/CMPD300/train

# Muzzle→face gap (reportable): CMPD300 encoder on Ahmed faces
python scripts/06_eval_reid.py \
    --target-dir /path/to/Ahmed \
    --single-shot \
    --compare-imagenet

# Muzzle→muzzle cross-dataset gap: CMPD300 encoder on Zenodo
python scripts/06_eval_reid.py \
    --target-dir /path/to/Zenodo/BeefCattle_Muzzle_Individualized \
    --by-session \
    --single-shot \
    --compare-imagenet
```

Key flags:

| Flag | Description |
|---|---|
| `--compare-imagenet` | Also runs a plain ImageNet ResNet-50 as a control baseline |
| `--single-shot` | 1 image/session per individual in gallery (harder; reduces burst-photo leakage) |
| `--by-session` | Split by capture session (avoids splitting a burst between gallery and probe) |
| `--ckpt` | Path to encoder checkpoint (default: `outputs/checkpoints/cmpd300_source.pt`) |

**Phase 7 — Alternative encoders + domain adaptation (this work)**

```bash
# Muzzle crop from faces (zero-shot, GroundingDINO) — needs: pip install "transformers==4.44.2"
python scripts/crop_muzzles.py --faces-dir /path/to/Ahmed --out-dir ~/data/cows_face_muzzle

# ArcFace metric-learning encoder on CMPD300
python scripts/07_train_arcface.py --train-dir /path/to/CMPD300/train --epochs 40

# DANN domain-adversarial (source=CMPD300, target=unlabeled crops), + baseline ablation
python scripts/08_train_dann.py --target-dir ~/data/cows_face_muzzle --mode dann \
    --warmup-epochs 5 --lam-max 0.5 --feat-dim 512 --dropout 0.1 --gamma 5 --eval-every 5
python scripts/08_train_dann.py --target-dir ~/data/cows_face_muzzle --mode baseline

# Re-evaluate any encoder with the harness above (--ckpt outputs/checkpoints/<encoder>.pt)
```

### Colab / Kaggle notebooks

The notebooks in `notebooks/` orchestrate the scripts above in GPU environments:

| Notebook | Platform | What it does |
|---|---|---|
| `kaggle_runner.ipynb` | Kaggle | Stage 1 full run (phases 0→4) |
| `colab_runner.ipynb` | Colab Pro | Stage 1 full run (phases 0→4) |
| `colab_fase5_source.ipynb` | Colab | Train CMPD300 encoder (Phase 5) |
| `colab_fase6_reid.ipynb` | Colab | Re-ID harness + muzzle→face gap (Ahmed) |
| `colab_gap_muzzle.ipynb` | Colab | Cross-dataset muzzle→muzzle gap (Zenodo) |
| `colab_gradcam_reid.ipynb` | Colab | Grad-CAM over cosine similarity of embeddings |
| `gradcam_runner.ipynb` | Colab | Visual test: VGG16_BN Grad-CAM on Ahmed faces |

All logic lives in `src/` and `scripts/`; notebooks only orchestrate.

---

## Results and findings

### Stage 1

| Model | Variant | Seeds | Test global (mean ± std) | Test balanced (mean ± std) |
|---|---|---|---|---|
| VGG16_BN | ce | 3 | 0.9598 ± 0.0027 | 0.9317 ± 0.0019 |
| VGG16_BN | ce_aug | 3 | 0.9699 ± 0.0017 | 0.9493 ± 0.0011 |
| VGG16_BN | wce | 3 | 0.9648 ± 0.0033 | 0.9615 ± 0.0025 |
| VGG16_BN | wce_aug | 3 | 0.9686 ± 0.0014 | 0.9692 ± 0.0028 |
| ResNet-50 | freeze | 1 | _(pending)_ | _(pending)_ |
| ResNet-50 | finetune | 1 | _(pending)_ | _(pending)_ |

> **Paper's thesis reproduced.** Global accuracy is similar across variants (~0.96–0.97),
> but **balanced accuracy** (equal weight to rare classes) climbs `ce` 0.9317 → `wce` 0.9615
> → `wce_aug` 0.9692: weighted CE and augmentation help the 8 classes with only 4 images.

The paper reports 98.7% for VGG16_BN. The replication aims to reproduce the trend (wce and
augmentation help the 8 classes with only 4 images), not to match the exact decimal.

Key deviations from the paper (see [`DEVIATIONS.md`](DEVIATIONS.md) for details):
- **3 seeds** instead of 5 (T4 GPU budget, ~43 min/run).
- **T4 GPU** instead of P100 (Kaggle's current PyTorch build does not support sm_60).
- **Additive augmentation** (precomputed to disk): the paper describes augmentation as a step
  that *creates synthetic images and enlarges the dataset*, not as per-epoch online augmentation.
- Weighted CE `N_max` computed from the training split (≈46), not the paper's literal 70.

### Stage 2

Protocol: session split + single-shot, Rank-1. CMPD300-trained encoder vs. plain ImageNet.

| Experiment | CMPD300 encoder | Plain ImageNet |
|---|---|---|
| Muzzle→face (Ahmed crops, 349 ids) | 0.724 | 0.724 |
| Muzzle→muzzle (Zenodo, 268 ids) | 0.864 | 0.894 |

Beyond the classification encoder, **ArcFace** (metric learning) and **DANN** (domain-adversarial,
including a variant warm-started from the source encoder) were also tested — **none beats a plain
ImageNet ResNet-50**, and specializing harder transfers *worse*. Full breakdown (all encoders,
per-epoch DANN transfer curve, muzzle-crop resolution) in [`RESULTS_STAGE2.md`](RESULTS_STAGE2.md).

**Main finding:** across all experiments, a plain ImageNet ResNet-50 (with no muzzle
training whatsoever) matches or outperforms the CMPD300-specialized encoder. This indicates
that the observed performance does not measure muzzle biometric recognition, but generic visual
similarity that the ImageNet backbone already captures. There is no domain adaptation gap
attributable to the muzzle modality that can be closed with adaptation techniques, at least
with these datasets and this protocol.

Grad-CAM visualizations (`colab_gradcam_reid.ipynb`) show which image regions each encoder
attends to when computing cosine similarity, helping explain why ImageNet matches the specialist.

---

## References

**Replicated paper:**
> Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification Using
> Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.
> DOI: [10.3390/ani12111453](https://doi.org/10.3390/ani12111453)

**Stage 1 dataset:**
> Zenodo Muzzle DB — record 6324361. 268 bovines, 4923 muzzle images.
> [https://zenodo.org/record/6324361](https://zenodo.org/record/6324361)

**Stage 2 source dataset:**
> CMPD300 — bovine muzzle dataset with pre-existing splits (train/val/test = MuzzleSplit).
> Full reference: _(pending — add the original CMPD300 source citation)_

**Stage 2 target dataset:**
> Bovine face dataset (Ahmed) — Cows Frontal Face Dataset, Zenodo record 10535934.
> [https://zenodo.org/records/10535934](https://zenodo.org/records/10535934)

---

## Stage 3 — Unsupervised cross-dataset re-ID by clustering (muzzle → muzzle)

Frozen encoder → embed an **unlabelled** second muzzle dataset → **cluster** to discover
the identities (estimating how many cattle there are) → evaluate against the real labels
(labels used **only** to evaluate). Question: does specializing on muzzle help discover
identities in a new field, or does a strong generic encoder group them just as well?

**Phase 0 (this code):** the full circuit with DINOv2 as the generic baseline.

```bash
# DINOv2 + plain ImageNet, on the Zenodo Muzzle DB (target = unlabelled new field)
python scripts/10_cluster_reid.py --target-dir /path/to/BeefCattle_Muzzle_Individualized \
    --encoders dinov2 imagenet

# add our muzzle-trained encoders (disjoint identities vs their CMPD300 training)
python scripts/10_cluster_reid.py --target-dir /path/to/zenodo \
    --encoders dinov2 imagenet resnet-ckpt --ckpt outputs/checkpoints/cmpd300_source.pt
```

On Colab use [`notebooks/colab_fase0_cluster.ipynb`](notebooks/colab_fase0_cluster.ipynb)
(mounts Drive, extracts the zip, runs the script).

Each encoder passes through the **same** pipeline; the only thing that changes is the
encoder. Reported per encoder (`outputs/results/10_cluster_summary.json`):
- **Clustering:** DBSCAN over an eps grid + HDBSCAN (auto) + k-means with the real *k*
  (a "ceiling that cheats"). Metrics: **ARI, NMI**, clusters found vs real, noise fraction.
  Reported on the full set **and** on a one-photo-per-session view (anti-burst).
- **Retrieval (kNN):** Rank-1 / Rank-5 / mAP on the single-shot-by-session split.
- **Session diagnostics:** whether the filename-based session split is even meaningful for
  the dataset — if `session_split_is_meaningful=False`, the anti-burst control is weak and
  the numbers may reflect photo similarity, not biometrics. **Read this first.**

Modules: [`src/reid/encoders.py`](src/reid/encoders.py) (encoder zoo),
[`src/reid/cluster.py`](src/reid/cluster.py) (clustering + metrics),
session helpers in [`src/reid/reid_dataset.py`](src/reid/reid_dataset.py).

---

## Reproducibility and integrity

- Splits saved to disk in `outputs/splits/` and reused; never re-split between runs.
- Fixed seed for splitting (`SPLIT_SEED = 42`) and explicit seeds for each replicate.
- All deviations from the paper's recipe are documented in [`DEVIATIONS.md`](DEVIATIONS.md) with justification.
- Full configuration is logged at the start of every run (see `outputs/logs/`).
- No fabricated metrics: real numbers go to `outputs/results/`, including per-class accuracy
  for the 8 cattle with only 4 images.