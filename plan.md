# Plan de implementación — Replicación de identificación bovina por hocico

Spec para construir con Claude Code. Objetivo: **replicar los resultados de Li, Erickson & Xiong (2022)**, "Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques" (*Animals* 12(11):1453), sobre el dataset Zenodo Muzzle DB. Este es el requisito base del TP (replicar un paper publicado de ID bovina por hocico). Las fases posteriores (cross-dataset + domain adaptation) se construyen sobre esto.

---

## 0. Contexto y objetivo

- **Tarea:** clasificación de conjunto cerrado. 268 vacas = 268 clases. Dada una imagen de hocico, predecir a qué individuo pertenece.
- **Dataset:** 4923 imágenes de hocico de 268 vacas de feedlot (EE.UU.), ya descargado localmente.
- **Resultado a igualar:** mejor accuracy del paper = **98.7%** (modelo VGG16_BN con cross-entropy + data augmentation). El segundo mejor estable fue VGG19_BN con weighted cross-entropy.
- **Criterio de éxito de la replicación:** accuracy en test en el rango **~96–98%+**. Igualar 98.7% al decimal es improbable por la aleatoriedad del reshuffle; caer en ese rango y **reproducir la tendencia** (que weighted CE y data augmentation ayudan a las clases con pocas imágenes) cuenta como replicación válida.

> Nota importante: el paper reporta que **VGG16_BN es el mejor, no ResNet-50**. Para replicar fielmente hay que usar VGG16_BN. Igual entrenamos también ResNet-50 porque es el backbone que vamos a reutilizar en las fases de domain adaptation.

---

## 1. La receta exacta del paper (target, no improvisar)

| Componente | Valor del paper |
|---|---|
| Resolución de entrada | **300×300** px (NO 224×224) |
| Normalización | intensidades por canal a rango **[0,1]** (dividir por 255). NO usa mean/std de ImageNet |
| Split | **65% train / 15% val / 20% test**, aleatorio, reshuffle, **por imagen** (todas las 268 clases presentes en cada split) |
| Transfer learning | preentrenado en ImageNet; **se fine-tunean SOLO las capas fully-connected** (backbone convolucional congelado) |
| Épocas | 50 |
| Optimizador | SGD, momentum 0.9 |
| Learning rate | inicial 0.001, decaído por factor 0.1 cada 7 épocas (StepLR step_size=7, gamma=0.1) |
| Loss base | Cross-Entropy |
| Réplicas | 5 corridas con misma semilla → reportar **accuracy media en test** |
| Métrica | top-1 accuracy global + por clase. Opcional: velocidad de procesamiento (ms/imagen) |

**Desbalance de clases (importante):** las imágenes por animal van de **4 a 70**. Las 4 vacas con solo 4 imágenes (IDs 2100, 4549, 5355, 5925) son las que dan 0% sin optimización.

**Optimización de desbalance (lo que sube de 98.4% a 98.7%):**
- **Weighted Cross-Entropy (WCE):** peso por clase `w_i = N_max / N_i`, con `N_max = 70` (imágenes de la vaca con más muestras), `N_i` = imágenes de la clase i.
- **Data augmentation** (solo en train): flip horizontal; brillo con factor entre 0.2 y 0.5; rotación aleatoria entre −15° y +15°; blur gaussiano con kernel entre 1 y 5.

**Valores no especificados en el paper (decidir y documentar):**
- Batch size: usar 32 (ajustar si la GPU se queda sin memoria con 300×300).
- Semillas: fijar 5 semillas explícitas (p.ej. 0,1,2,3,4) para las réplicas.

---

## 2. Estructura del proyecto a generar

```
cattle-reid/
├── plan.md                  # este archivo
├── README.md                # cómo correr cada fase
├── requirements.txt
├── config.py                # rutas, hiperparámetros, semillas (single source of truth)
├── data/
│   └── (NO commitear el dataset; solo apuntar DATA_DIR desde config)
├── src/
│   ├── dataset.py           # Dataset + DataLoader, split estratificado
│   ├── transforms.py        # preprocesamiento + data augmentation del paper
│   ├── models.py            # builder de VGG16_BN y ResNet-50 (freeze backbone / full finetune)
│   ├── losses.py            # CE y Weighted CE
│   ├── train.py             # loop de entrenamiento + validación, 1 corrida
│   ├── evaluate.py          # accuracy global y por clase en test
│   └── utils.py             # seeds, logging, guardado de checkpoints/métricas
├── scripts/
│   ├── 00_inspect_data.py   # verifica estructura del dataset y reporta stats
│   ├── 01_make_splits.py    # genera y guarda los splits (json con paths+labels)
│   ├── 02_train_vgg.py      # replicación: VGG16_BN, 5 réplicas, 3 variantes
│   └── 03_train_resnet.py   # backbone propio: ResNet-50
├── outputs/
│   ├── splits/              # splits guardados (reproducibilidad)
│   ├── checkpoints/         # pesos guardados
│   └── results/             # csv/json de métricas + tabla resumen
└── notebooks/
    └── colab_runner.ipynb   # wrapper para correr en Colab/Kaggle con GPU
```

---

## 3. Tareas por fase

### Fase 0 — Inspección de datos (`scripts/00_inspect_data.py`)
**Antes de escribir nada de entrenamiento.** El dataset ya está descargado pero hay que confirmar su estructura real.
1. Recibir `DATA_DIR` desde `config.py`.
2. Listar subcarpetas; confirmar que hay ~268 carpetas (una por animal).
3. Contar imágenes por carpeta. Reportar: nº de clases, total de imágenes (esperado ~4923), min/max/media de imágenes por clase, histograma simple.
4. Verificar que el min sea 4 y el max 70 (sanity check contra el paper).
5. Detectar extensiones de imagen presentes (.jpg/.png), imágenes corruptas o ilegibles.
6. Imprimir un reporte. **No avanzar a Fase 1 hasta que el reporte cuadre.** Si la estructura difiere (p.ej. un solo nivel de carpetas, o un csv de labels), adaptar `dataset.py` en consecuencia.

### Fase 1 — Dataset, splits y transforms (`dataset.py`, `transforms.py`, `01_make_splits.py`)
1. **Split estratificado por imagen** 65/15/20 con semilla fija. Estratificar por clase para que cada una de las 268 aparezca en train, val y test. Guardar los splits como JSON (lista de `(path, label)`) en `outputs/splits/` para reproducibilidad — **no re-splitear en cada corrida**.
2. `LabelEncoder` carpeta→entero 0..267; guardar el mapeo.
3. **Transforms base (val/test):** resize 300×300 → ToTensor (esto ya escala a [0,1]). **No** aplicar normalización ImageNet (el paper usa [0,1] crudo). Dejar la opción de normalización ImageNet como flag configurable para experimentar después.
4. **Transforms de train (data augmentation, variante con aug):** resize 300×300 + RandomHorizontalFlip + ColorJitter(brightness=(0.2,0.5)) + RandomRotation(15) + GaussianBlur(kernel ∈ {1,3,5}) → ToTensor.
5. `Dataset` de PyTorch que lea desde los JSON de split. `DataLoader` con `num_workers` configurable.

### Fase 2 — Modelos y losses (`models.py`, `losses.py`)
1. `build_model(name, num_classes=268, freeze_backbone=True)`:
   - `vgg16_bn`: cargar `torchvision.models.vgg16_bn(weights=IMAGENET)`, reemplazar la última capa del clasificador por `Linear(..., 268)`. Si `freeze_backbone`, congelar `features` y entrenar solo `classifier`.
   - `resnet50`: cargar preentrenado, reemplazar `fc` por `Linear(2048, 268)`. Soportar `freeze_backbone=True` (solo fc) y `False` (fine-tune completo).
2. `losses.py`:
   - CE estándar.
   - **Weighted CE:** calcular pesos `w_i = 70 / N_i` desde los conteos del split de train; pasar a `nn.CrossEntropyLoss(weight=...)`.

### Fase 3 — Entrenamiento y evaluación (`train.py`, `evaluate.py`, `02_train_vgg.py`)
1. `train.py`: loop estándar — SGD(momentum=0.9, lr=0.001), StepLR(step=7, gamma=0.1), 50 épocas, trackear val accuracy por época, guardar el mejor checkpoint por val acc.
2. `evaluate.py`: cargar mejor checkpoint, calcular **top-1 accuracy global** y **accuracy por clase** en test. Guardar a CSV. (Opcional: ms/imagen.)
3. `02_train_vgg.py` corre la **replicación completa**:
   - Modelo VGG16_BN, `freeze_backbone=True`.
   - **3 variantes** × **5 semillas**:
     - (a) CE sola, sin augmentation
     - (b) CE + data augmentation
     - (c) Weighted CE (sin augmentation)
   - Reportar accuracy media ± std en test por variante.
   - **Validación de éxito:** la mejor variante debe caer ~96–98%+, y (b)/(c) deben mejorar la accuracy de las clases con pocas imágenes vs (a). Generar una tabla resumen en `outputs/results/`.

### Fase 4 — Backbone propio ResNet-50 (`03_train_resnet.py`)
1. Misma receta, modelo ResNet-50.
2. Correr **dos modos**: `freeze_backbone=True` (como el paper) y `freeze_backbone=False` (fine-tune completo).
3. Guardar los pesos de la mejor corrida en `outputs/checkpoints/` — **este es el modelo que se reutiliza en domain adaptation**.
4. Reportar su accuracy. No se espera que iguale exactamente a VGG16_BN.

---

## 4. Gotchas (verificar explícitamente)

- **Split por imagen, NO por animal.** En clasificación closed-set las 268 clases deben estar en train/val/test. Nunca dejar un animal solo en test (rompería la tarea). El split por-animal/disjunto recién aplica en la fase futura de re-identificación.
- **Normalización [0,1], no ImageNet.** El paper escala a [0,1] crudo. `ToTensor` ya hace eso. Si se agrega `Normalize(mean,std)` de ImageNet se obtienen números distintos a los del paper. Dejarlo como flag, default off.
- **Backbone congelado.** El paper entrena solo las FC. Si se descongela todo, el resultado no es comparable con el reportado. Para la replicación: `freeze_backbone=True`.
- **Las 4 clases con 4 imágenes.** Si la accuracy global da bien pero más baja que el paper, revisar la accuracy por clase de los IDs 2100/4549/5355/5925 — probablemente sean ellas las que tiran el promedio.
- **Memoria GPU con 300×300 + VGG16_BN.** VGG es pesado en VRAM. Si hay OOM, bajar batch size antes de tocar la resolución (la resolución es parte de la receta).
- **Reproducibilidad.** Fijar seeds de `random`, `numpy`, `torch` y `torch.cuda` en `utils.py`. Guardar los splits a disco y reusarlos.

---

## 5. Entorno de ejecución: Kaggle Notebooks

**Plataforma elegida: Kaggle.** GPU gratis (P100 o T4×2), 30 hs/semana, sin setup de drivers ni de cuotas, sin colas de disponibilidad. La P100 es la misma GPU que usó el paper. (Google Cloud queda reservado para la fase futura de domain adaptation.)

### 5.1. Dependencias
- Python 3.10+, PyTorch + torchvision, scikit-learn (split estratificado), Pillow, numpy, pandas, tqdm. La imagen de Kaggle ya trae casi todo; instalar solo lo que falte.
- Mantener un `requirements.txt` con versiones fijadas para reproducibilidad local, pero en Kaggle confiar en la imagen base + pip puntual.

### 5.2. Subir el dataset a Kaggle
El dataset (~643 MB) ya está descargado localmente. Subirlo como **Kaggle Dataset privado** (Create → New Dataset → subir el .zip o la carpeta). Una vez creado, se adjunta al notebook con "Add Input" y queda montado en modo **solo lectura** en:
```
/kaggle/input/<slug-del-dataset>/
```
Confirmar la estructura real ahí dentro en la Fase 0 (puede haber una carpeta extra de nivel superior según cómo se haya zippeado).

### 5.3. Subir el código
Dos opciones, en orden de preferencia:
1. **Repo en GitHub** → en el notebook, con internet activado, `git clone`. Es lo más limpio para iterar y versionar.
2. Subir `src/` como un **Kaggle Dataset de tipo "utility script"** y adjuntarlo.

### 5.4. Configuración del notebook
- **Accelerator:** Settings → Accelerator → **GPU P100** (o T4×2; para entrenar un modelo a la vez alcanza con una sola GPU, no complicar con multi-GPU).
- **Internet: ON.** Necesario para `git clone`, `pip install` y —clave— para que **torchvision descargue los pesos preentrenados de ImageNet** la primera vez. Requiere cuenta de Kaggle verificada por teléfono.

### 5.5. Rutas (mapear en `config.py`)
```
DATA_DIR   = /kaggle/input/<slug-del-dataset>/...   # solo lectura
OUTPUT_DIR = /kaggle/working/outputs                # escritura + persiste al guardar versión
```
Importante: **`/kaggle/working/` es el único directorio escribible que persiste** (hasta 20 GB, se guarda al hacer "Save Version"). Todos los checkpoints, splits y resultados van ahí. Cualquier cosa fuera de `/kaggle/working/` se pierde al cerrar la sesión.

### 5.6. Límites de sesión (planificar el sweep en función de esto)
- Sesión interactiva: se corta a las ~12 h; idle timeout ~20–40 min (la sesión muere si no hay actividad).
- Ejecución en background ("Save & Run All" / commit): corre hasta ~12 h sin que tengas que estar conectado. **Usar este modo para el sweep largo** (3 variantes × 5 semillas).
- Cuota: **30 h de GPU por semana**.
- El presupuesto alcanza de sobra: con `freeze_backbone=True` solo se entrena la cabeza FC, así que cada corrida es rápida. Aun así, **validar primero el pipeline con 1 sola semilla y pocas épocas**, y recién después lanzar el sweep completo en background. No quemar la cuota debuggeando.

### 5.7. Checklist de arranque en el notebook
1. `nvidia-smi` → confirmar la GPU.
2. `python -c "import torch; print(torch.cuda.is_available())"` → debe dar `True`.
3. `git clone` del repo (o adjuntar el dataset de código).
4. Setear `DATA_DIR` al path de `/kaggle/input/...` y correr `scripts/00_inspect_data.py`.
5. Verificar que el reporte cuadre (268 clases, ~4923 imágenes, min 4 / max 70) antes de entrenar.

---

## 6. Entregable de esta etapa

1. Pipeline reproducible que, apuntando a `DATA_DIR` (montado en `/kaggle/input/...`), corre Fase 0→4 de punta a punta en un notebook de Kaggle con GPU.
2. Tabla resumen en `outputs/results/` con accuracy media ± std por variante (VGG16_BN: CE / CE+aug / WCE) y de ResNet-50.
3. Checkpoint de ResNet-50 guardado para reutilizar.
4. README con instrucciones de ejecución.

---

## 7. Fases futuras (fuera de alcance ahora, dejar el código preparado)

No implementar todavía, pero diseñar `models.py` y `dataset.py` para que extiendan a:
- **Extractor de embeddings:** tomar el backbone ResNet-50 entrenado, quitar la cabeza de clasificación, exponer embeddings.
- **Protocolo gallery/probe** con identidades disjuntas, métricas Rank-1 y mAP, para evaluación cross-dataset (Pakistán/Ahmed, etc.).
- **Domain adaptation:** DANN con Gradient Reversal Layer, y self-training con pseudo-labels por clustering.

---

## Referencia

Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453. DOI: 10.3390/ani12111453. Dataset: Zenodo record 6324361.
