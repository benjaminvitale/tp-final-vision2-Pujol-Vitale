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

**Dos protocolos (ambos con datos reales, nada fabricado):**
- **A — crudo + `mcs=4` (optimista):** sin dedup (4923 imgs). En crudo el mínimo real es 4
  fotos/vaca, así que `mcs=4` no disuelve ninguna. Los near-duplicates de ráfaga facilitan el
  clustering → **límite superior, con "trampa" de duplicados**.
- **B — dedup + `mcs=2` (limpio, HEADLINE):** pHash dedup (1554 imgs). Anti-fuga, desplegable.
- El delta A−B mide cuánto inflaban los duplicados (ver §5). No se augmenta el eval: fabricar
  copias inflaría el ARI y reintroduce los near-dups que el dedup sacó.

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

**Matiz (con eps\* label-free, protocolo B):** el efecto se separa por backbone. En **ResNet la
heavy sigue restando fuerte** (0.641 → 0.523), pero en **DINOv2-base se lava** (strong 0.785 ≈
heavy 0.788). O sea: "la heavy perjudica" es sólido para ResNet; para DINOv2 el efecto es chico y
depende del eps. En ambos casos, `strong` es igual o mejor → es la receta elegida.

---

## 5. Modelos / backbones y el resultado final

Baselines congelados (sin entrenar) y encoders fine-tuneados con SupCon+strong sobre CMPD300.
Cada ARI con eps fijo (0) y **eps\* label-free** (silhouette, sin etiquetas — ver §6).

**Escalera de encoders — protocolo B (dedup, mcs=2, HEADLINE):**

| Encoder | ARI eps=0 | **ARI eps\*** | BCubed-F1 | homog / compl | Rank-1 |
|---|---|---|---|---|---|
| **DINOv2-large + SupCon** | 0.716 | **0.835** | 0.905 | 0.984 / 0.960 | 0.872 |
| DINOv2-base + SupCon | 0.687 | 0.785 | 0.885 | 0.978 / 0.949 | 0.876 |
| ResNet-50 + SupCon | 0.542 | 0.641 | 0.841 | 0.965 / 0.926 | 0.809 |
| ImageNet ResNet-50 (frozen) | 0.461 | 0.550 | 0.829 | 0.955 / 0.920 | 0.802 |
| DINOv2-base (frozen) | 0.153 | 0.155 | 0.728 | 0.878 / 0.876 | 0.669 |

**Batería completa del ganador (DINOv2-large + SupCon, protocolo B, eps\*):** ARI 0.835 · AMI 0.916 ·
NMI 0.972 · homog 0.984 / compl 0.960 · BCubed P/R/F1 0.962/0.855/0.905 · #clusters 304 (vs 268,
ratio 1.13) · noise 0.01 · Rank-1 0.872 · k-means 0.790.

**Robustez — A (optimista) vs B (limpio) en el ganador:**

| protocolo | ARI eps=0 | ARI eps\* | Rank-1 |
|---|---|---|---|
| A (crudo, con duplicados) | 0.833 | 0.846 | 0.920 |
| B (dedup, limpio) | 0.716 | 0.835 | 0.872 |
| **Δ (A−B)** | **0.117** | **0.011** | 0.048 |

A eps=0 los duplicados inflan fuerte (Δ=0.117). Pero **con eps\* label-free la brecha casi
desaparece (Δ=0.011)**: el número limpio y desplegable (**0.835**) llega prácticamente al límite
superior optimista (0.846). Quitar la muleta de los duplicados casi no cuesta → **0.835 es real, no
un artefacto del protocolo.** (Rank-1 sí cae más, 0.920→0.872: retrieval se beneficia más de los
duplicados; 0.872 es el número honesto.)

**Atribución limpia (protocolo B, eps\*):**
- **Backbone (misma aug strong):** ResNet→DINOv2 = **+0.144**. La palanca real.
- **Tamaño:** base→large = **+0.050** (consistente; BCubed-F1 y Rank-1 acompañan).
- **Augmentation:** `heavy` resta en ResNet, se lava en DINOv2 (ver §4).

**El hallazgo más interesante:** DINOv2 **congelado es el PEOR** (0.155) — sus features crudos no
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

**Resultado:** el criterio label-free eligió eps\*=0.050 para DINOv2-large → **ARI 0.835**, idéntico
al techo-oráculo (0.835@0.05). O sea, **la ventaja del oráculo era recuperable sin etiquetas**: la
sobre-partición se corrige a nivel de clustering una vez que el embedding es bueno (el ganador baja
de ~386 a 304 clusters, ratio 1.13, con noise 0.01). El número reportable pasa de 0.716 (eps fijo)
a **0.835 (eps\* label-free)**.

La selección usa un **piso absoluto** de #clusters (implementado en `scripts/reid_eval.py`, sin
`n_true`), así que es 100% label-free — no mira el conteo real del target.

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
   fijo, **0.835** con selección de eps label-free (BCubed-F1 0.905, Rank-1 0.872).
3. **Más augmentation agresiva perjudicó** al ResNet; en DINOv2 el efecto se lava — resultado
   negativo válido, `strong` es la receta.
4. **La sobre-partición del clustering se corrige sin etiquetas** una vez que el embedding es
   bueno: el eps label-free recupera todo el techo-oráculo (0.835).
5. **El resultado limpio es robusto:** en el protocolo optimista (crudo, con duplicados) el ganador
   da 0.846 y en el limpio (dedup) 0.835 — con eps label-free la brecha es solo 0.011, o sea que el
   número desplegable casi iguala al límite superior. No es un artefacto del dedup.
6. Enmarcar bien: 0.835 de ARI en descubrimiento **no supervisado, cross-dataset, sin conocer el
   conteo** es un régimen mucho más duro que el 98.7% de clasificación cerrada del paper original.
   No son comparables.

## 9. Limitaciones

- **Una sola seed** por condición. Los efectos grandes (backbone +0.144) son robustos; el
  base→large (+0.050) es consistente pero conviene confirmarlo con seeds.
- **8 vacas (3%) tienen 1 sola foto** post-dedup → son inherentemente no-clusterizables (un cluster
  necesita ≥2 puntos). Es un límite del dataset, no del método.

## 10. Trabajo futuro

- **SpCL / Design B:** self-training sobre el target no etiquetado, arrancando de DINOv2-large.
  Es **otro protocolo** (adaptación, no transferencia zero-shot) — mantener los claims separados.
- Confirmar base→large con 2–3 seeds si se quiere afirmar con rigor.
