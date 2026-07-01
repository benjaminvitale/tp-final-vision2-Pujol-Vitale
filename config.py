"""config.py — Única fuente de verdad de rutas, hiperparámetros y semillas.

Toda la receta del paper (Li, Erickson & Xiong, 2022) vive acá. Los scripts y
módulos de `src/` importan de este archivo; no se hardcodean valores sueltos.

Lo que dice el PAPER vs lo que decidimos NOSOTROS está marcado en los comentarios.
Las desviaciones respecto del paper se documentan además en DEVIATIONS.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
# Carpeta raíz del dataset: contiene una subcarpeta por animal (cattle_XXXX/).
# Resolución por orden de prioridad:
#   1. Variable de entorno CATTLE_DATA_DIR (override manual).
#   2. Rutas típicas de Kaggle (/kaggle/input/.../BeefCattle_Muzzle_Individualized).
#   3. Extracción local para smoke-test (data_local/...), buscando hacia arriba
#      desde este archivo (sirve también corriendo dentro de un git worktree).
_DATASET_DIRNAME = "BeefCattle_Muzzle_Individualized"

# Raíz del proyecto = carpeta que contiene este config.py.
PROJECT_ROOT = Path(__file__).resolve().parent


def _find_dataset_dir() -> Path:
    """Devuelve la primera ruta válida que contenga el dataset."""
    # 1. Override por entorno.
    env = os.environ.get("CATTLE_DATA_DIR")
    if env:
        return Path(env)

    candidates: list[Path] = []

    # 2. Kaggle: el dataset puede estar bajo cualquier slug en /kaggle/input.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # match directo .../<slug>/BeefCattle_Muzzle_Individualized
        candidates += list(kaggle_input.glob(f"*/{_DATASET_DIRNAME}"))
        # por si el slug ya ES el dataset
        candidates += list(kaggle_input.glob(f"*/*/{_DATASET_DIRNAME}"))
        # fallback recursivo: el mount puede estar más profundo, p.ej.
        # /kaggle/input/datasets/<user>/<slug>/BeefCattle_Muzzle_Individualized
        deep = next(kaggle_input.rglob(_DATASET_DIRNAME), None)
        if deep is not None:
            candidates.append(deep)

    # 3. Local: data_local/ en la raíz del proyecto o en repos padre (worktree).
    for base in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidates.append(base / "data_local" / _DATASET_DIRNAME)
        candidates.append(base / "data" / _DATASET_DIRNAME)

    for c in candidates:
        if c.is_dir():
            return c

    # Fallback: devolver la ruta local esperada aunque no exista todavía,
    # para que el mensaje de error apunte a algo concreto.
    return PROJECT_ROOT / "data_local" / _DATASET_DIRNAME


DATA_DIR = _find_dataset_dir()

# Outputs (todo lo generado por el pipeline).
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"
AUG_CACHE_DIR = OUTPUTS_DIR / "aug_cache"   # imágenes sintéticas precomputadas (ver scripts/04)

# --------------------------------------------------------------------------- #
# Dataset (valores verificados por conteo de archivos en Fase 0; coinciden con el paper)
# --------------------------------------------------------------------------- #
NUM_CLASSES = 268          # PAPER: 268 vacas = 268 clases. Confirmado en Fase 0.
EXPECTED_IMAGES = 4923     # PAPER: 4923 imágenes. Confirmado en Fase 0.
# PAPER: min=4 / max=70. Verificado por conteo de archivos en Fase 0: coincide.
# (Hay 8 clases con 4 imágenes; el paper menciona solo 4 IDs — ver DEVIATIONS.md.)
MIN_IMAGES_PER_CLASS = 4
MAX_IMAGES_PER_CLASS = 70

# --------------------------------------------------------------------------- #
# Splits (PAPER: 65/15/20, por imagen, estratificado, reshuffle)
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.65
VAL_FRAC = 0.15
TEST_FRAC = 0.20
SPLIT_SEED = 42            # NOSOTROS: semilla fija para el split (reproducible).

# --------------------------------------------------------------------------- #
# Preprocesamiento / transforms (PAPER)
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 300          # PAPER: 300x300 (NO 224).
# PAPER: normaliza a [0,1] crudo (ToTensor ya lo hace). NO usa mean/std ImageNet.
USE_IMAGENET_NORM = False  # NOSOTROS: flag para experimentar; default OFF (paper).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Data augmentation (PAPER, solo en train).
AUG_BRIGHTNESS = (0.2, 0.5)   # ColorJitter brightness factor.
AUG_ROTATION_DEG = 15         # RandomRotation [-15, +15].
AUG_BLUR_KERNELS = (1, 3, 5)  # GaussianBlur kernel ∈ {1,3,5}.

# Augmentation ADITIVA (ver DEVIATIONS D4). PAPER: la augmentation "crea imágenes
# sintéticas y agranda el dataset", MANTENIENDO los originales (sobre todo para las
# clases con pocas imágenes). NO reemplaza cada imagen como el augmentation online típico.
# Por clase se expande hasta min(AUG_TARGET_CAP, AUG_FACTOR * N_i) agregando copias
# aumentadas; las clases con >= AUG_TARGET_CAP imágenes no se tocan.
AUG_TARGET_CAP = 20   # NOSOTROS: tope de imágenes por clase tras expandir.
AUG_FACTOR = 2        # NOSOTROS: como mucho duplicar las imágenes de la clase.

# --------------------------------------------------------------------------- #
# Entrenamiento (PAPER)
# --------------------------------------------------------------------------- #
EPOCHS = 50
OPTIMIZER = "sgd"
MOMENTUM = 0.9
LR = 0.001
LR_STEP_SIZE = 7          # StepLR step_size.
LR_GAMMA = 0.1            # StepLR gamma.
FREEZE_BACKBONE = True    # PAPER: fine-tunea SOLO las FC (backbone congelado).

# NOSOTROS (no especificados por el paper, ver DEVIATIONS.md):
BATCH_SIZE = 32           # bajar si hay OOM con 300x300 + VGG16_BN.
NUM_WORKERS = 4
REPLICATE_SEEDS = (0, 1, 2)  # 3 réplicas → media ± std. Bajamos de 5 a 3 por el
                             # presupuesto de 12 h en T4 (ver DEVIATIONS.md D2). Así
                             # `02_train_vgg.py` sin --seeds ya entra en un solo commit.

# --------------------------------------------------------------------------- #
# Weighted Cross-Entropy
# --------------------------------------------------------------------------- #
# PAPER: w_i = N_max / N_i, con N_max=70 (máximo del dataset completo).
# Nosotros calculamos los pesos desde el split de TRAIN (sin fuga): N_i = conteo en
# train, N_max = máximo de esos conteos (≈46, porque train ≈65% de la clase de 70).
# N_max es solo un factor de escala GLOBAL de la loss; el peso relativo entre clases
# (N_max/N_i) es lo que importa y se preserva. Para forzar el valor literal del paper
# (70) y comparar, setear el override. Ver DEVIATIONS.md.
WCE_NMAX_OVERRIDE: int | None = None  # None = empírico del train (≈46).

# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
MODELS = ("vgg16_bn", "resnet50")  # vgg16_bn = replica el paper; resnet50 = backbone propio.

# Pesos preentrenados de ImageNet. Para no depender de internet en Kaggle, los
# bajamos una vez y los subimos como Kaggle Dataset; models.py los carga desde acá.
# Nombres canónicos de torchvision (NO renombrar los .pth):
PRETRAINED_FILES = {
    "vgg16_bn": "vgg16_bn-6c64b313.pth",
    "resnet50": "resnet50-0676ba61.pth",
}


def _find_pretrained_dir() -> Path | None:
    """Carpeta con los .pth preentrenados, o None si no se encuentra (→ se baja).

    Orden: env CATTLE_PRETRAINED_DIR → /kaggle/input/*/ (donde estén los .pth) →
    local pretrained_weights/ (en el proyecto o repos padre / worktree).
    """
    env = os.environ.get("CATTLE_PRETRAINED_DIR")
    if env:
        return Path(env)

    any_file = next(iter(PRETRAINED_FILES.values()))
    candidates: list[Path] = []

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # carpeta de Kaggle que contenga el .pth (en cualquier nivel razonable)
        candidates += [p.parent for p in kaggle_input.glob(f"*/{any_file}")]
        candidates += [p.parent for p in kaggle_input.glob(f"*/*/{any_file}")]
        # fallback recursivo: el mount puede estar más profundo
        # (/kaggle/input/datasets/<user>/<slug>/...).
        deep = next(kaggle_input.rglob(any_file), None)
        if deep is not None:
            candidates.append(deep.parent)

    for base in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidates.append(base / "pretrained_weights")
        # ...o cualquier subcarpeta inmediata que contenga los .pth (detección por
        # NOMBRE DE ARCHIVO, igual que en Kaggle): así también sirve imagenet-pretrained/.
        candidates += [p.parent for p in base.glob(f"*/{any_file}")]

    for c in candidates:
        if (c / any_file).is_file():
            return c
    return None


PRETRAINED_DIR = _find_pretrained_dir()

# --------------------------------------------------------------------------- #
# ETAPA 2 — Dataset CMPD300 (SOURCE, hocico) — baseline cross-modality
# --------------------------------------------------------------------------- #
# CMPD300 ya viene splitteado en carpetas: <CMPD300_DIR>/{train,val,test}/<ID>/*.JPG
# NO se re-splittea. `scripts/00_inspect_cmpd300.py` lee esas carpetas y genera los
# JSON de split + label_map (mismo formato {"path","label"} que usa src/dataset.py).
# Ojo: los IDs NO son contiguos y val no tiene todas las clases (lo reporta el script).
_CMPD300_DIRNAME = "Baseline"   # carpeta que contiene train/ val/ test/


def _find_cmpd300_dir() -> Path:
    """Primera ruta válida con CMPD300 (env → Kaggle → local datasets/)."""
    env = os.environ.get("CMPD300_DATA_DIR")
    if env:
        return Path(env)

    candidates: list[Path] = []

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates += list(kaggle_input.glob(f"*/{_CMPD300_DIRNAME}"))
        candidates += list(kaggle_input.glob(f"*/*/{_CMPD300_DIRNAME}"))
        deep = next(kaggle_input.rglob(f"{_CMPD300_DIRNAME}/train"), None)
        if deep is not None:
            candidates.append(deep.parent)

    for base in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidates.append(base / "datasets" / _CMPD300_DIRNAME)

    for c in candidates:
        if (c / "train").is_dir():
            return c

    return PROJECT_ROOT / "datasets" / _CMPD300_DIRNAME


CMPD300_DIR = _find_cmpd300_dir()
CMPD300_SPLITS_DIR = OUTPUTS_DIR / "splits_cmpd300"   # JSON de split + label_map de CMPD300

# Preprocesamiento ETAPA 2 (cross-modality). Decisión NUESTRA, no es la receta del
# paper. 224 + norm ImageNet: matchea lo que el backbone preentrenado espera y es más
# liviano que 300. EL MISMO preprocesamiento se usa después en el target (caras Ahmed)
# para no inflar el gap por mismatch. Para comparar con la receta del paper, cambiar a
# IMAGE_SIZE_S2=300 / USE_IMAGENET_NORM_S2=False.
IMAGE_SIZE_S2 = 224
USE_IMAGENET_NORM_S2 = True


def ensure_output_dirs() -> None:
    """Crea las carpetas de outputs si no existen."""
    for d in (SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, CMPD300_SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict:
    """Snapshot serializable de la config (para loguear con cada run)."""
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }


if __name__ == "__main__":
    import json

    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}  (existe: {DATA_DIR.is_dir()})")
    print(json.dumps(as_dict(), indent=2, ensure_ascii=False))
