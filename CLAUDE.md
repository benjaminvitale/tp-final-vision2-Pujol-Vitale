# CLAUDE.md — Contexto del proyecto

> Este archivo es el contexto persistente para Claude Code. Leerlo al inicio de cada sesión, junto con `plan.md`. Si algo de acá entra en conflicto con una instrucción puntual del usuario, preguntar antes de avanzar.

---

## Qué es este proyecto

Trabajo práctico final de **Visión Artificial Avanzada** (Universidad de San Andrés). El objetivo de esta etapa es **replicar un paper publicado de identificación individual de ganado bovino a partir de imágenes de hocico (muzzle)** y, sobre esa base, construir progresivamente un benchmark cross-dataset y experimentos de domain adaptation.

**Paper a replicar:** Li, Erickson & Xiong (2022), *Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques*, Animals 12(11):1453. DOI 10.3390/ani12111453.

**Dataset:** Zenodo Muzzle DB (record 6324361). 4923 imágenes de hocico de 268 vacas, organizadas por individuo. Ya está descargado.

**Tarea (esta etapa):** clasificación de conjunto cerrado, 268 vacas = 268 clases. Dada una imagen de hocico, predecir el individuo.

**Qué significa "éxito":** accuracy en test ~96–98%+ (el paper reporta 98.7% con VGG16_BN) **y** reproducir la tendencia de que weighted cross-entropy y data augmentation ayudan a las clases con pocas imágenes. Igualar el número al decimal NO es el objetivo ni es esperable.

---

## Estado y roadmap

- ✅ **Planning** — completo. Spec detallada en `plan.md`.
- 🔨 **En construcción (alcance ACTUAL):** Fases 0–4 de `plan.md` — inspección de datos, splits, dataset/transforms, modelos (VGG16_BN para replicar + ResNet-50 como backbone propio), entrenamiento, evaluación.
- 🔜 **Futuro (NO implementar todavía):** extractor de embeddings, protocolo gallery/probe (Rank-1/mAP) cross-dataset, domain adaptation (DANN + self-training). Diseñar el código para que extienda a esto, pero no construirlo aún.

**`plan.md` es la fuente de verdad del "qué hacer".** No reimplementar la receta del paper de memoria: está toda ahí (resolución, split, optimizador, losses, augmentation, hiperparámetros). Si Claude Code necesita un valor, va a `plan.md`.

---

## Cómo trabajar en este repo

1. **Leer `plan.md` antes de escribir código.** Trabajar fase por fase, en orden. No saltearse la Fase 0 (inspección de datos): el dataset puede no tener la estructura que asumimos.
2. **Cambios chicos y revisables.** Un commit por unidad lógica de trabajo, con mensaje claro. No mega-commits.
3. **Preguntar antes de desviarse de la receta del paper** o de tomar decisiones de arquitectura no triviales. Las desviaciones se documentan (ver más abajo).
4. **Validar antes de escalar.** Probar el pipeline con 1 semilla y pocas épocas antes de lanzar el sweep completo (3 variantes × 5 semillas). No quemar cuota de GPU debuggeando.
5. Mantener `README.md` actualizado con cómo correr cada fase.

---

## Stack y estructura

- **Lenguaje/frameworks:** Python 3.10+, PyTorch + torchvision, scikit-learn, Pillow, numpy, pandas, tqdm.
- **Ejecución:** Kaggle Notebooks con GPU (P100 / T4). Ver sección 5 de `plan.md` para detalles de montaje de datos, rutas y límites de sesión.
- **Estructura de carpetas:** definida en `plan.md` sección 2 (`src/`, `scripts/`, `outputs/`, `config.py` como single source of truth de hiperparámetros y rutas).

---

## Comandos clave

> Completar/ajustar a medida que se construye. Mantener esta lista al día.

```bash
# inspección de datos (correr SIEMPRE primero)
python scripts/00_inspect_data.py

# generar y guardar los splits
python scripts/01_make_splits.py

# replicación VGG16_BN (3 variantes × N semillas)
python scripts/02_train_vgg.py

# backbone propio ResNet-50
python scripts/03_train_resnet.py

# verificación de GPU (en Kaggle)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Convenciones de código

- **`config.py` es la única fuente de hiperparámetros, rutas y semillas.** No hardcodear valores sueltos en scripts.
- Funciones con type hints; docstrings cortos donde aporten.
- Logging legible por corrida: imprimir/guardar la config completa al inicio de cada entrenamiento.
- Nada de notebooks como fuente de verdad de la lógica: la lógica vive en `src/`, los notebooks solo orquestan.
- Determinismo: fijar seeds de `random`, `numpy`, `torch` y `torch.cuda` en un único lugar (`utils.py`).

---

## Principios de responsabilidad (IMPORTANTE)

Esto es un trabajo académico de replicación. La integridad de los resultados es lo primero.

- **Nunca fabricar, ajustar a mano ni hardcodear métricas para que "den" como el paper.** Reportar los números reales, incluso si son peores. Si no logramos reproducir, se documenta el porqué — eso es un resultado válido, no un fracaso a ocultar.
- **Documentar toda desviación de la receta del paper.** Si por alguna razón cambiamos resolución, normalización, optimizador, etc., queda anotado explícitamente (en el README o en un `DEVIATIONS.md`) con la justificación. La replicación se evalúa por fidelidad, no solo por el número.
- **Reproducibilidad real:** splits guardados a disco y reusados (no re-splitear por corrida), seeds fijas, versiones de librerías registradas, config logueada con cada run.
- **Separar claramente "lo que dice el paper" de "lo que decidimos nosotros"** en comentarios y documentación.
- **Resultados honestos por clase, no solo el agregado.** Reportar accuracy por clase (especialmente las 4 vacas con 4 imágenes), no esconder el peor caso detrás del promedio.

---

## Higiene del repositorio

- **NO commitear el dataset** ni binarios pesados (imágenes, checkpoints grandes). Usar `.gitignore`. El dataset vive en Kaggle (montado en `/kaggle/input/...`), no en git.
- **NO commitear credenciales, tokens ni API keys.** Nada de Kaggle/GCP en el repo.
- `outputs/` (checkpoints, resultados) fuera de git salvo las tablas/CSV de métricas finales, que sí conviene versionar.
- `requirements.txt` con versiones fijadas.
- Commits atómicos y descriptivos.

---

## Hechos del dominio que NO hay que equivocar

- **Es HOCICO (muzzle), no cara.** El patrón del hocico es como una huella digital individual. No confundir con reconocimiento facial bovino (es otra modalidad).
- **268 clases**, no 256.
- **El mejor modelo del paper es VGG16_BN, no ResNet-50.** Para replicar el 98.7% se usa VGG16_BN. ResNet-50 se entrena además como backbone propio para reutilizar en domain adaptation, pero no es el modelo que replica el número del paper.
- **Esta etapa es clasificación closed-set: el split es POR IMAGEN, las 268 clases en train/val/test.** El split por animal / identidades disjuntas (gallery/probe, Rank-1/mAP) recién aplica en la fase futura de re-identificación cross-dataset. No mezclar los dos protocolos.
- **Backbone congelado** para la replicación fiel (el paper solo fine-tunea las FC).
- **Normalización [0,1] crudo**, no mean/std de ImageNet (lo que usa el paper).

---

## Qué NO hacer

- No empezar a entrenar sin haber corrido la Fase 0 y confirmado la estructura real del dataset.
- No implementar las fases futuras (embeddings, DANN, self-training) en esta etapa; solo dejar el diseño preparado para extenderlas.
- No introducir frameworks pesados (PyTorch Lightning, Hydra, etc.) sin acordarlo: mantener el stack simple y legible.
- No optimizar prematuramente (multi-GPU, mixed precision, etc.) hasta tener el baseline andando y validado.
- No "mejorar" la receta del paper durante la replicación. Primero replicar fiel; las mejoras vienen después y por separado.

---

## Referencia

Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453. DOI: 10.3390/ani12111453. Dataset: Zenodo record 6324361.
