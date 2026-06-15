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
| 3 | `python scripts/02_train_vgg.py` | Replicación VGG16_BN: 3 variantes × 5 semillas (pendiente) |
| 4 | `python scripts/03_train_resnet.py` | Backbone propio ResNet-50 (pendiente) |

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

## Estructura

```
config.py                 # hiperparámetros, rutas, semillas (single source of truth)
src/
  utils.py                # seeds, logging, I/O
  dataset.py              # (Fase 1) Dataset + DataLoader
  transforms.py           # (Fase 1) preprocesamiento + data augmentation
  models.py               # (Fase 2) VGG16_BN, ResNet-50
  losses.py               # (Fase 2) CE, Weighted CE
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
