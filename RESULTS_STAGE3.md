# Etapa 3 — Re-ID bovino no supervisado por clustering (resultados)

> Documento de resultados del TP. Números reales, single seed, todos reproducidos por el mismo
> pipeline de evaluación. La discusión de honestidad metodológica está integrada, no escondida.

---

## 1. El problema que elegimos

**Pregunta.** ¿Especializar un encoder en el hocico (muzzle) ayuda a **descubrir identidades en
un campo nuevo sin etiquetas**, o un encoder genérico fuerte las agrupa igual de bien?

Es un salto respecto de las etapas anteriores (clasificación cerrada supervisada, 268 clases).
Acá el problema es **más duro y más realista**: llega un campo nuevo, con vacas que el modelo
**nunca vio**, **sin saber cuántas hay**, y hay que agruparlas por individuo.

**Setup (transferencia zero-shot).**
1. Entrenar un encoder en el **source etiquetado** — CMPD300 (300 identidades, 1500 imágenes).
2. **Congelarlo** (el encoder nunca ve el target).
3. Extraer embeddings del **target no etiquetado** — Zenodo Muzzle DB (268 identidades, 4923
   imágenes → **1554** tras dedup por pHash).
4. Clusterizar con HDBSCAN y medir contra las etiquetas reales (usadas **solo** para evaluar).

**Métrica primaria: HDBSCAN ARI.** Por dos razones: (1) **deployability** — en el campo real no
conocés cuántas vacas hay, así que la tarea desplegable es clusterizar sin saber *k*, no retrieval;
(2) **Rank-1 satura** en este target: SupCon 0.810 ≈ ImageNet 0.803, o sea que no discrimina entre
encoders — hasta features genéricas la pasan. La ventaja del especialista aparece recién en **ARI**,
que es la tarea más dura. (Los hocicos de Zenodo son crops apretados **sin fondo**, así que el
Rank-1 alto no se explica por contexto/entorno.) Rank-1 se reporta como secundaria, con ese caveat.

---

## 2. Protocolo de evaluación (fijo para todas las corridas)

- **Anti-fuga:** dedup por perceptual hash (dHash) sobre Zenodo → saca frames casi idénticos
  (burst twins). Se cae de 4923 a 1554 imágenes. Sin esto, clustering y retrieval harían trampa
  matcheando copias.
- **Clustering:** HDBSCAN sobre distancia coseno de embeddings L2-normalizados, `min_cluster_size`
  fijo. **Primaria = ARI**; secundarias = NMI, #clusters.
- **Diagnósticos (no objetivo):** k-means con k=268 (oráculo, "hace trampa" porque conoce el
  conteo), y kNN → Rank-1.
- **Reproducibilidad:** splits/dedup deterministas; el mismo checkpoint da el mismo ARI al
  reevaluar. **Limitación declarada:** una sola seed de entrenamiento por condición.

---

## 3. Comparación de losses (backbone ResNet-50, augmentation fijo, 60 epochs)

Primer experimento: fijar backbone + augmentation + sampler y variar **solo la loss**. Sampler
**PK** (P=16 identidades × K=4 imágenes por batch), necesario para SupCon y Triplet.

| Loss | HDBSCAN ARI | k-means (oráculo) | Rank-1 |
|---|---|---|---|
| **SupCon** (τ=0.07) | **0.542** | 0.743 | 0.810 |
| Cross-Entropy | 0.492 | 0.694 | 0.800 |
| ArcFace (s=30, m=0.5) | 0.267 | 0.628 | 0.721 |
| Triplet batch-hard (soft-margin) | 0.210 | 0.588 | 0.689 |

**Hallazgos (validados contra los logs de entrenamiento — las 4 convergieron limpio):**
- **SupCon gana.** Optimiza la **geometría** del espacio directamente (misma identidad cerca,
  distinta lejos), sin prototipos por clase fijos → transfiere a identidades nuevas.
- **ArcFace sobre-especializa.** Aprendió el source (87% train acc) pero su margen angular
  comprime las clases del source alrededor de prototipos que no existen para vacas nuevas →
  peor transferencia. Es la hipótesis nula "especializar más no ayuda", confirmada.
- **Triplet quedó sub-entrenado** en este régimen de pocos datos (soft-margin + ~5 fotos/vaca +
  positivos con reemplazo). No es "triplet es inferior", es un setup sub-potente.
- **CE es un control razonable** pero su embedding es un subproducto, no un objetivo → peor que
  SupCon para clustering.

→ **SupCon queda elegida como la loss del proyecto.**

---

## 4. Augmentation: `strong` vs `heavy`

Con el diagnóstico de que el problema era la **varianza intra-identidad** (ver §6), probamos una
augmentation mucho más agresiva, dirigida a forzar el foco en el hocico y no en el fondo.

| | `strong` (receta base) | `heavy` (agresiva, box-free) |
|---|---|---|
| RandomResizedCrop | scale 0.5–1.0 | **scale 0.3–0.9** (más zoom) |
| Rotación | ±30° | **±45°** + RandomPerspective |
| ColorJitter | 0.4 | 0.5 + GaussianBlur |
| RandomErasing | 1× (p=0.5) | **2×** (borra más regiones) |

**Resultado — la heavy EMPEORÓ los dos backbones:**

| Backbone | strong | heavy | Δ |
|---|---|---|---|
| ResNet-50 | 0.542 | 0.483 | **−0.059** |
| DINOv2-base | 0.687 | 0.663 | **−0.024** |

**Hallazgo honesto:** más augmentation **no** fue la palanca. El crop agresivo + doble erasing
**destruyeron señal del hocico**. `strong` es la receta. (Es un resultado negativo válido, se
reporta tal cual.)

---

## 5. Modelos / backbones y el resultado final

Baselines congelados (sin entrenar) y encoders fine-tuneados con SupCon+strong sobre CMPD300.

Dos números de ARI, ambos honestos: **eps fijo** (conservador) y **eps\* label-free**
(seleccionado por silhouette sobre el target, sin etiquetas — ver §6).

| Encoder | init | fine-tune | ARI (eps=0) | **ARI (eps\* LF)** | k-means | Rank-1 |
|---|---|---|---|---|---|---|
| **DINOv2-large + SupCon** | DINOv2-large | sí | 0.716 | **0.831** | 0.796 | 0.872 |
| DINOv2-base + SupCon | DINOv2-base | sí | 0.687 | 0.759 | 0.788 | 0.876 |
| ResNet-50 + SupCon | ImageNet | sí | 0.542 | 0.641 | 0.743 | 0.809 |
| ImageNet ResNet-50 | ImageNet | no (frozen) | 0.461 | 0.566 | 0.737 | 0.803 |
| DINOv2-base | DINOv2 | no (frozen) | 0.150 | — | 0.574 | 0.667 |

**La escalera del encoder ganador sube monótona** (ARI eps* label-free):
`ImageNet 0.566 → ResNet+SupCon 0.641 → DINOv2-base+SupCon 0.759 → DINOv2-large+SupCon 0.831`.
Con eps fijo la escalera es la misma: `0.461 → 0.542 → 0.687 → 0.716`.

**Atribución limpia (cuadrado backbone × aug):**
- **Backbone (misma aug strong):** ResNet→DINOv2 = **+0.145**. La palanca real.
- **Tamaño:** base→large = **+0.029** (modesto; single-seed cae cerca de la banda de ruido, pero
  consistente — k-means también sube 0.788→0.796).
- **Augmentation:** `heavy` restó en ambos.

**El hallazgo más interesante:** DINOv2 **congelado es el PEOR** (0.150) — sus features crudos no
son discriminativos para el hocico — pero como **init para fine-tunear es el MEJOR**. La conclusión
no es "genérico vs especializado", es: **el mejor encoder = init genérico fuerte + especialización
en hocico con una loss clusterizable (SupCon).**

---

## 6. Diagnóstico de clustering (por qué el techo es el embedding)

Con el encoder SupCon original (0.542) exploramos si el clustering post-hoc podía mejorar:

- **NMI 0.93 ≫ ARI 0.54** → los clusters son **puros pero sobre-partidos**: el modelo casi nunca
  mezcla dos vacas, pero parte las ~5 fotos de una misma vaca en ~1.4 clusters.
- **Reducción de dimensión (PCA-50, UMAP-32) no ayudó** (UMAP incluso empeoró).
- **Subir `min_cluster_size` empeora** (con ~5.8 fotos/vaca, exigir clusters grandes fusiona
  identidades o manda fotos a ruido).
- **`cluster_selection_epsilon`** tiene una ventana finísima: eps≈0.02 sube el ARI (fusiona las
  mitades partidas), pero eps>0.03 colapsa todo → el **margen entre vacas distintas es < 0.05** en
  coseno.

**Conclusión:** la variación intra-vaca ≈ el margen inter-vaca. El post-hoc de clustering está
**agotado**; el techo es el **embedding**. Por eso el trabajo se movió a mejorar el encoder
(§4–5), y por eso DINOv2-large ayuda: aprieta esa varianza.

### Nota metodológica — selección de `eps` (label-free)

El eps óptimo **depende del dataset** (la densidad del espacio cambia), así que adaptarlo al
target es legítimo — **siempre que no se use la etiqueta del target**. Distinguimos:
- **eps por ARI** (mirando etiquetas) = **oráculo, NO reportable**. Da ~0.835 para DINOv2-large,
  pero en despliegue no tenés esas etiquetas.
- **eps por validez interna** (silhouette coseno sobre los embeddings del target, sin etiquetas)
  = **desplegable**. Se barre un grid, se elige el eps que maximiza silhouette (con guardas
  anti-degeneración), y recién después se mide el ARI de ese eps.

**Resultado:** el criterio label-free eligió eps\*=0.048 para DINOv2-large → **ARI 0.831**, casi
idéntico al techo-oráculo (0.835@0.05). O sea, **la ventaja del oráculo era recuperable sin
etiquetas**: la sobre-partición se corrige a nivel de clustering una vez que el embedding es bueno.
El número reportable pasa de 0.716 (eps fijo) a **0.831 (eps\* label-free)**.

*Caveat menor:* una de las guardas anti-degeneración usa `n_true` (el conteo real) como piso de
#clusters — un mini-leak. No afectó el resultado (los eps elegidos dan 306–352 clusters, muy por
encima del piso) y se reemplaza por un piso absoluto para ser 100% label-free.

---

## 7. Detalles de entrenamiento (comunes)

- Sampler **PK** P=16, K=4 (batch 64). Optimizador **Adam** + cosine con warmup corto.
- LR: 3e-4 (ResNet) / **3e-5** (DINOv2, el ViT necesita LR baja). ~60 epochs.
- DINOv2-large corre en batch 64 en una GPU L4 gracias a **gradient checkpointing**.
- Para evaluar se usan SIEMPRE las **features del backbone** (2048-d ResNet / 768-d o 1024-d ViT),
  no la cabeza de proyección de SupCon (que es solo para entrenar). Comparación justa.

---

## 8. Conclusiones

1. **SupCon** es la mejor loss de las cuatro para transferencia por clustering: moldea un espacio
   que se agrupa solo en un campo nuevo. ArcFace sobre-especializa; Triplet quedó sub-entrenado.
2. **El backbone es la palanca dominante.** Un init genérico fuerte (DINOv2), fine-tuneado con
   SupCon, supera por lejos a ResNet/ImageNet. DINOv2-large es el mejor: **ARI 0.716** con eps
   fijo, **0.831** con selección de eps label-free.
3. **Más augmentation agresiva perjudicó** — resultado negativo válido.
4. **La sobre-partición del clustering se corrige sin etiquetas** una vez que el embedding es
   bueno: el eps label-free recupera casi todo el techo-oráculo (0.831 vs 0.835).
5. Enmarcar bien: 0.831 de ARI en descubrimiento **no supervisado, cross-dataset, sin conocer el
   conteo** es un régimen mucho más duro que el 98.7% de clasificación cerrada del paper original.
   No son comparables.

## 9. Limitaciones

- **Una sola seed** por condición. Los efectos grandes (backbone +0.145) son robustos; el
  base→large (+0.029) cae cerca de la banda de ruido.
- La ventaja de tamaño (base→large) es modesta.
- La guarda de selección de eps usa `n_true` (mini-leak, no afectó el resultado; ver §6).

## 10. Trabajo futuro

- Piso de #clusters **absoluto** en la selección de eps (quitar el `n_true` → 100% label-free).
- **SpCL / Design B:** self-training sobre el target no etiquetado, arrancando de DINOv2-large.
  Es **otro protocolo** (adaptación, no transferencia zero-shot) — mantener los claims separados.
- Confirmar base→large con 2–3 seeds si se quiere afirmar con rigor.
