# DEVIATIONS.md — Desviaciones respecto del paper

Este archivo registra **toda** diferencia entre lo que hacemos nosotros y la receta
del paper de referencia:

> Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification
> Using Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.

La replicación se evalúa por **fidelidad**, no solo por el número. Separar claramente
"lo que dice el paper" de "lo que decidimos nosotros".

---

## Estado: el dataset coincide con el paper

Verificado en Fase 0 (`scripts/00_inspect_data.py`, conteo por archivos sobre las
4923 imágenes reales):

| | Paper | Real (medido) |
|---|---|---|
| Nº de clases | 268 | 268 ✓ |
| Total imágenes | 4923 | 4923 ✓ |
| Min imágenes/clase | 4 | 4 ✓ |
| Max imágenes/clase | 70 | 70 ✓ |
| Media | — | 18.4 |

**No hay desviación en los datos.** La receta del paper (`N_max=70`, distribución
4–70) aplica tal cual.

---

## D1 — Aclaración: hay 8 clases con 4 imágenes (no 4)

**Paper:** menciona "las 4 vacas con solo 4 imágenes (IDs 2100, 4549, 5355, 5925)".

**Real:** hay **8** clases con exactamente 4 imágenes:
`2100, 3420, 4549, 5208, 5355, 5630, 5925, 8050`. Las 4 que nombra el paper son un
subconjunto de estas.

**Impacto:** ninguno sobre la receta. Solo importa al reportar accuracy por clase:
revisar las **8**, no 4, como las candidatas a tirar el promedio. No es una
desviación, es una corrección factual de la nota del paper.

---

## D2 — Valores no especificados por el paper (decididos por nosotros)

No son desviaciones (el paper no los fija), pero se documentan para reproducibilidad:

- **Batch size:** 32 (bajar si hay OOM con 300×300 + VGG16_BN).
- **Semillas de réplica:** `(0, 1, 2, 3, 4)` para las 5 corridas.
- **Semilla de split:** 42, fija; los splits se guardan a disco y se reusan.
- **num_workers** del DataLoader: 4.
- **Weighted CE `N_max`:** los pesos se calculan desde el split de **train** (sin
  fuga): `N_i` = conteo en train, `N_max` = máximo de esos conteos (**≈46**, no 70,
  porque train ≈65% de la clase de 70 imágenes). `N_max` es solo un factor de escala
  global de la loss; el peso relativo entre clases (`N_max/N_i`) se preserva, que es
  lo que importa. El paper usa el literal 70 (máximo del dataset completo). Override
  disponible en `config.py` (`WCE_NMAX_OVERRIDE = 70`) para reproducir el valor exacto
  del paper y comparar.

---

## D3 — Hardware: T4 en vez de P100 (no afecta la receta)

**Paper / plan:** GPU **P100** (la que usó el paper; `plan.md` §5 la prefería).

**Real:** la imagen actual de Kaggle trae un PyTorch compilado **sin soporte Pascal
(sm_60)**, así que la P100 tira `CUDA error: no kernel image is available for execution`
en el primer forward. Entrenamos en **GPU T4 ×2** (Turing, sm_75), que sí está en las
capabilities soportadas por ese PyTorch (sm_70–sm_120). Se usa una sola T4 (`cuda:0`).

**Impacto en resultados:** ninguno. Es solo el dispositivo de cómputo; la receta
(resolución, normalización, optimizador, losses, épocas, seeds) no cambia. Puede variar
levemente el tiempo por época y el redondeo numérico de bajo nivel, no la metodología.

---

## D4 — (reservado)

Anotar acá cualquier desviación futura (resolución, normalización, optimizador,
descongelar backbone, etc.) con su justificación **antes** de aplicarla.

---

> Nota histórica: una primera inspección por `unzip -l | grep cattle_[0-9]+` reportó
> erróneamente 8–140 (media 36.7). Causa: cada ruta contiene el ID dos veces
> (`.../cattle_0100/cattle_0100_x.jpg`), duplicando el conteo. El conteo por archivos
> es el válido y coincide con el paper.
