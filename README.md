# Cattle Re-ID — Identificación individual de ganado por hocico

Replicación de **Li, Erickson & Xiong (2022)**, *Individual Beef Cattle Identification
Using Muzzle Images and Deep Learning Techniques* (Animals 12(11):1453), sobre el
dataset Zenodo Muzzle DB (268 vacas, 4923 imágenes de hocico).

TP final de **Visión Artificial Avanzada** (Universidad de San Andrés).

- `plan.md` — spec detallada (la receta del paper, fase por fase). **Fuente de verdad.**
- `CLAUDE.md` — contexto del proyecto y principios de trabajo.
- `DEVIATIONS.md` — toda diferencia respecto del paper, documentada.
- `config.py` — única fuente de hiperparámetros, rutas y semillas.

## Tarea

Clasificación de conjunto cerrado: 268 vacas = 268 clases. Dada una imagen de hocico,
predecir el individuo. Meta: accuracy en test **~96–98%+** (el paper reporta 98.7% con
VGG16_BN) y reproducir que Weighted CE + data augmentation ayudan a las clases con
pocas imágenes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

**Dataset.** No se commitea (ver `.gitignore`). Resolución de `DATA_DIR` (en `config.py`):

1. Variable de entorno `CATTLE_DATA_DIR` (override).
2. Kaggle: `/kaggle/input/<slug>/BeefCattle_Muzzle_Individualized`.
3. Local: `data_local/BeefCattle_Muzzle_Individualized` (extraer el zip ahí para
   smoke-test). Busca también en repos padre, así funciona dentro de un git worktree.

```bash
# extracción local para smoke-test (desde la carpeta que tiene el zip)
unzip -q BeefCattle_Muzzle_database.zip -d data_local
python config.py        # imprime DATA_DIR resuelto y la config completa
```

## Pipeline (correr en orden)

| Fase | Comando | Qué hace |
|---|---|---|
| 0 | `python scripts/00_inspect_data.py` | Inspecciona y valida el dataset (✅ implementado) |
| 1 | `python scripts/01_make_splits.py` | Split 65/15/20 por imagen, estratificado (✅ implementado) |
| 3 | `python scripts/02_train_vgg.py` | Replicación VGG16_BN: 3 variantes × 5 semillas (✅ implementado) |
| 4 | `python scripts/03_train_resnet.py` | Backbone propio ResNet-50: freeze + full fine-tune (✅ implementado) |

> **Siempre correr Fase 0 primero.** No avanzar si los sanity checks no pasan.

### Fase 0 — inspección (implementada)

```bash
python scripts/00_inspect_data.py                # reporte + sanity checks
python scripts/00_inspect_data.py --check-corrupt # además verifica imágenes ilegibles
```

Verificado: **268 clases, 4923 imágenes, 4–70 por clase (media 18.4)** → coincide con
el paper. Reporte en `outputs/results/00_inspect_report.json`.

### Fase 1 — splits + dataset + transforms (implementada)

```bash
python scripts/01_make_splits.py    # genera outputs/splits/{train,val,test}.json + label_map.json
```

- **Split por imagen 65/15/20**, estratificado por clase con `SPLIT_SEED` fijo, con
  garantía de que las 268 clases estén en los 3 splits (necesario para las clases
  con 4 imágenes). Resultado real: 64.7 / 15.2 / 20.0%.
- Rutas guardadas **relativas a `DATA_DIR`** → el mismo split sirve en Kaggle y local.
- `src/transforms.py`: pipeline base (resize 300×300 + ToTensor = [0,1], **sin**
  ImageNet) y de train con la augmentation del paper (flip, brillo 0.2–0.5,
  rotación ±15°, blur gaussiano kernel ∈ {1,3,5}). Flag `USE_IMAGENET_NORM` (default OFF).
- `src/dataset.py`: `MuzzleDataset` + `make_dataloader` leyendo desde los JSON.

### Fase 2 — modelos + losses (implementada)

- `src/models.py`: `build_model('vgg16_bn'|'resnet50', num_classes=268, freeze_backbone=True)`.
  Reemplaza la cabeza por `Linear(..., 268)` y, con `freeze_backbone=True` (PAPER),
  congela el backbone conv y entrena solo las FC.
- `src/losses.py`: `build_loss('ce'|'wce')`. Weighted CE con `w_i = N_max/N_i`
  calculado desde el split de train (ver `DEVIATIONS.md` sobre `N_max`).

**Pesos preentrenados (Kaggle, sin internet).** Para no depender de internet en el
notebook, bajá una vez los `.pth` de ImageNet y subilos como Kaggle Dataset:

| Modelo | Archivo (no renombrar) |
|---|---|
| VGG16_BN | `vgg16_bn-6c64b313.pth` |
| ResNet-50 | `resnet50-0676ba61.pth` |

```bash
curl -L -O https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
curl -L -O https://download.pytorch.org/models/resnet50-0676ba61.pth
```

`config.py` autodetecta esos `.pth` **por nombre de archivo**: en Kaggle bajo
`/kaggle/input/*/`, y en local en cualquier subcarpeta que los contenga (p.ej.
`imagenet-pretrained/` o `pretrained_weights/`). Override con `CATTLE_PRETRAINED_DIR`.
Si no los encuentra, `models.py` los baja por internet.

**Subir los datasets a Kaggle (automatizado).** En vez de la UI, `scripts/kaggle_upload.py`
sube ambos datasets con `kagglehub` (rutas derivadas de `config.py`). Correr desde el
checkout local que tiene la data:

```bash
pip install kagglehub
export KAGGLE_USERNAME=tu_usuario KAGGLE_KEY=xxxxxxxx   # kaggle.com → Settings → API
python scripts/kaggle_upload.py --user tu_usuario        # sube imágenes + pesos
python scripts/kaggle_upload.py --user tu_usuario --only weights --version-notes "v2"
```

Sube `data_local/` (preserva la carpeta `BeefCattle_Muzzle_Individualized/`) y la
carpeta de pesos. **No commitear credenciales** (`KAGGLE_KEY` va por entorno).

### Fase 4 — backbone propio ResNet-50 (implementada)

```bash
python scripts/03_train_resnet.py            # freeze + full fine-tune, seed 0, 50 épocas
python scripts/03_train_resnet.py --smoke    # pipeline rápido en CPU (subset, 2 épocas, sin ImageNet)
```

- **Misma receta del paper** (300×300, [0,1] crudo, SGD mom=0.9, lr=0.001,
  StepLR(7, 0.1), 50 épocas), reusando `src/train.py` y `src/evaluate.py` —no se
  reimplementa el loop. ResNet-50 **NO** replica el 98.7% (ese es VGG16_BN); es el
  backbone que se reutilizará en domain adaptation.
- Corre **dos modos**: `freeze` (`freeze_backbone=True`, solo FC, como el paper) y
  `finetune` (`freeze_backbone=False`, fine-tune completo). Flags útiles:
  `--modes freeze`, `--seeds 0 1 2` (→ media ± std), `--loss wce`, `--aug`.
- El **mejor run por val accuracy** (selección sin fuga de test) se copia a
  `outputs/checkpoints/resnet50_backbone.pt` → ese es el checkpoint que toma la fase
  futura de domain adaptation.
- Reporta test global + balanced (media ± std por modo) en
  `outputs/results/03_resnet_summary.json`, CSV por clase por run, y la accuracy de las
  8 clases con 4 imágenes (ver `DEVIATIONS.md`) sin esconderla detrás del promedio.

## Estructura

```
config.py                 # hiperparámetros, rutas, semillas (single source of truth)
src/
  utils.py                # seeds, logging, I/O
  dataset.py              # (Fase 1) Dataset + DataLoader
  transforms.py           # (Fase 1) preprocesamiento + data augmentation
  models.py               # VGG16_BN, ResNet-50 (freeze backbone / full finetune)
  losses.py               # CE, Weighted CE
  train.py / evaluate.py  # (Fase 3) entrenamiento y evaluación
scripts/                  # 00..03, orquestación por fase
outputs/                  # splits/ checkpoints/ results/ logs/  (pesados gitignored)
notebooks/                # runner para Kaggle
```

## Reproducibilidad e integridad

- Splits guardados a disco y reusados; seeds fijas en `src/utils.py`.
- Nada de métricas fabricadas: se reportan los números reales, incluida la accuracy
  por clase (sobre todo las clases con 4 imágenes).
- Toda desviación de la receta del paper va a `DEVIATIONS.md`.
