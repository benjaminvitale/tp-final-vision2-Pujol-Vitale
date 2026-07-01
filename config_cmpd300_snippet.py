# =========================================================================== #
#  PEGAR ESTE BLOQUE EN config.py
#  Ubicación sugerida: justo DESPUÉS del bloque de "Modelos" / PRETRAINED_DIR,
#  y ANTES de la función ensure_output_dirs().
#  Nota: as_dict() ya levanta automáticamente cualquier variable UPPERCASE nueva,
#  así que estas constantes se loguean solas con cada run.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# ETAPA 2 — Dataset CMPD300 (SOURCE, hocico) — baseline cross-modality
# --------------------------------------------------------------------------- #
# CMPD300 ya viene splitteado en carpetas: <CMPD300_DIR>/{train,val,test}/<ID>/*.JPG
# NO se re-splittea. `scripts/00_inspect_cmpd300.py` lee esas carpetas y genera los
# JSON de split + label_map (mismo formato {"path","label"} que usa src/dataset.py).
# Ojo: los IDs NO son contiguos y val no tiene todas las clases (lo reporta el script).
_CMPD300_DIRNAME = "Baseline"   # carpeta que contiene train/ val/ test/


def _find_cmpd300_dir() -> "Path":
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

# --------------------------------------------------------------------------- #
# Preprocesamiento ETAPA 2 (cross-modality). Decisión NUESTRA, no es la receta del
# paper. 224 + norm ImageNet: matchea lo que el backbone preentrenado espera y es más
# liviano que 300. EL MISMO preprocesamiento se usa después en el target (caras Ahmed)
# para no inflar el gap por mismatch. Para comparar con la receta del paper, cambiar a
# IMAGE_SIZE_S2=300 / USE_IMAGENET_NORM_S2=False.
# --------------------------------------------------------------------------- #
IMAGE_SIZE_S2 = 224
USE_IMAGENET_NORM_S2 = True
