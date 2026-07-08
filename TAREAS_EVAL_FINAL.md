# Etapa 3 — Evaluación final para entrega (orden de trabajo)

> Spec ejecutable para Claude Code. Objetivo: dejar la Etapa 3 **lista para entregar** —
> métricas completas, tablas y gráficos— reusando `reid_eval.py` (ya testeado, no reimplementar).
> Todos los pasos son **re-evaluación de checkpoints ya entrenados** (deterministas), salvo lo
> marcado como *opcional (requiere reentrenar)*.

---

## 0. Regla de oro

- Usar **`reid_eval.py`** para métricas, selección de eps y gráficos. No reinventar.
- Toda evaluación usa **features del backbone** (2048-d ResNet / 768-d DINOv2-base / 1024-d
  DINOv2-large), L2-normalizadas. **Nunca** la cabeza de proyección de SupCon.
- Métrica primaria del proyecto = **ARI** (con selección de eps label-free). Rank-1 es secundaria
  y va con caveat (ver §7, nota 2).
- Cada número reportado debe reproducir el "valor esperado" del apéndice ±0.005. Si no matchea,
  el pipeline está mal — frenar y avisar.

---

## 1. Prerrequisitos

### 1.1 Módulo
- `reid_eval.py` en el path. Funciones: `full_metrics`, `metrics_table`, `bcubed`, `eps_sweep`,
  `plot_encoder_staircase`, `plot_eps_sweep`, `plot_learning_curves`.

### 1.2 Target de evaluación (fijo para TODO)
- **Zenodo Muzzle DB**, dedup por perceptual hash → **1554 imágenes, 268 identidades**.
- Reusar EXACTAMENTE el mismo conjunto dedup de 1554 en todas las corridas (mismo split, misma
  lista de archivos) para que los números sean comparables entre modelos.
- Guardar la lista de 1554 archivos + sus labels en `data/zenodo_1554_index.csv`
  (columnas: `path,label`) para reproducibilidad.

### 1.3 Checkpoints (mapear a los nombres reales del repo)

| Rol | Checkpoint (nombre en logs / placeholder) | Backbone | Notas |
|---|---|---|---|
| Baseline genérico | `imagenet_resnet50` (pesos ImageNet, sin finetune) | ResNet-50 | frozen |
| Baseline genérico | `dinov2_base_frozen` (DINOv2 sin finetune) | DINOv2-base | frozen |
| Loss: SupCon | `cmpd300_supcon.pt` | ResNet-50 | aug strong |
| Loss: CE | `cmpd300_ce.pt` *(confirmar nombre)* | ResNet-50 | aug strong |
| Loss: ArcFace | `cmpd300_arcface.pt` *(confirmar)* | ResNet-50 | s=30, m=0.5 |
| Loss: Triplet | `cmpd300_triplet.pt` *(confirmar)* | ResNet-50 | batch-hard soft-margin |
| Aug: heavy (ResNet) | `cmpd300_supcon_heavy.pt` *(confirmar)* | ResNet-50 | SupCon + heavy |
| Encoder | `cmpd300_supcon_dinov2_strong.pt` | DINOv2-base | SupCon + strong |
| Aug: heavy (DINOv2) | `cmpd300_supcon_dinov2_heavy.pt` *(confirmar)* | DINOv2-base | SupCon + heavy |
| Encoder ganador | `cmpd300_supcon_dinov2L_strong.pt` | DINOv2-large | SupCon + strong |

> Si falta alguno de los checkpoints marcados *(confirmar)*, listar cuáles y seguir con los que
> haya (no bloquear todo por uno).

---

## 2. Protocolo de evaluación (idéntico para cada modelo)

1. Cargar checkpoint, poner en `eval()`, extraer features del backbone del set de 1554.
2. L2-normalizar los embeddings.
3. Clusterizar con **HDBSCAN (coseno)**, `min_cluster_size = 4`, en dos configuraciones:
   - **eps = 0** (fijo, conservador).
   - **eps\*** = elegido por `eps_sweep(...)` con silhouette **label-free** (piso absoluto de
     #clusters; **NO** pasar `n_true`).
4. Calcular `full_metrics(y_true, y_pred)` para ambas configuraciones.
5. Guardar por modelo en `outputs/eval/<modelo>/`:
   - `embeddings.npy` (N×D), `assignments_eps0.npy`, `assignments_epsstar.npy`
   - `metrics.json` (dict de `full_metrics` para eps0 y eps\*, + `eps_star`)
   - `eps_sweep.csv` (salida de `eps_sweep`, para el gráfico Fig-4)

> **Sanity gate:** antes de seguir, verificar que `imagenet` da ARI@eps0 ≈ 0.461 y
> `dinov2L` da ARI@eps0 ≈ 0.716 / ARI@eps\* ≈ 0.831. Si no, el pipeline no está replicando.

---

## 3. Métricas a computar (la batería completa)

Para **cada** modelo, en eps0 y eps\* (las da `full_metrics`):

| Métrica | Qué mide | Por qué la incluimos |
|---|---|---|
| **ARI** | acuerdo con labels, corregido por azar | primaria del proyecto |
| **AMI** | info mutua ajustada | reemplaza/complementa NMI (NMI se infla con la sobre-partición) |
| **NMI** | info mutua normalizada | ya la reportábamos; se mantiene al lado de AMI |
| **homogeneity** | ¿cada cluster es una sola vaca? | va a salir **alta** → "clusters puros" |
| **completeness** | ¿cada vaca cae en un solo cluster? | va a salir **más baja** → "sobre-particionado" |
| **V-measure** | media armónica de las dos | resume homog/compl |
| **BCubed P/R/F1** | precision/recall per-ítem | el estándar de re-ID; robusto a clusters desbalanceados |
| **#clusters vs #true (ratio)** | 306/268, etc. | cuánto sobre-particiona |
| **noise_frac** | % mandado a ruido por HDBSCAN | ya la teníamos |

> El par **homogeneity/completeness** cuantifica el "puro pero sobre-particionado" del §6 que hoy
> está solo en prosa. Es la métrica que más suma para la defensa.

---

## 4. Tablas a producir

Generar en Markdown con `metrics_table(...)` y guardar en `outputs/tables/`.

### T1 — Comparación de losses (ResNet-50, aug strong fijo, 60 ep)
Modelos: SupCon, CE, ArcFace, Triplet. Columnas: la batería del §3 (eps0 **y** eps\*).
Números **ya conocidos** a reproducir (columna ARI@eps0) + Rank-1 y k-means como referencia:

| Loss | ARI@eps0 (esperado) | k-means oráculo | Rank-1 |
|---|---|---|---|
| SupCon | 0.542 | 0.743 | 0.810 |
| CE | 0.492 | 0.694 | 0.800 |
| ArcFace | 0.267 | 0.628 | 0.721 |
| Triplet | 0.210 | 0.588 | 0.689 |

→ Completar AMI / homog / compl / BCubed-F1 / #clust / noise (columnas nuevas, aún sin valor).

### T2 — Augmentation strong vs heavy
Modelos: ResNet-50 (strong/heavy) y DINOv2-base (strong/heavy). Columna clave: **ARI y Δ**.
Números ya conocidos:

| Backbone | strong | heavy | Δ (esperado) |
|---|---|---|---|
| ResNet-50 | 0.542 | 0.483 | −0.059 |
| DINOv2-base | 0.687 | 0.663 | −0.024 |

→ Es un **resultado negativo válido** (heavy empeora). Reportar tal cual.

### T3 — Comparación final de encoders/backbones
Modelos: `imagenet`, `resnet+supcon`, `dinov2-base+supcon`, `dinov2L+supcon`, y `dinov2-base frozen`.
Batería completa (eps0 y eps\*). Números conocidos:

| Encoder | ARI@eps0 | eps\* | ARI@eps\* | NMI | #clust | noise |
|---|---|---|---|---|---|---|
| ImageNet ResNet-50 (frozen) | 0.461 | 0.048 | 0.566 | 0.939 | 351 | 0.05 |
| ResNet-50 + SupCon | 0.542 | 0.020 | 0.641 | 0.945 | 352 | 0.04 |
| DINOv2-base + SupCon | 0.687 | 0.030 | 0.759 | 0.960 | 329 | 0.02 |
| **DINOv2-large + SupCon** | **0.716** | 0.048 | **0.831** | 0.971 | 306 | 0.01 |
| DINOv2-base (frozen) | 0.150 | — | — | — | — | — |

> **Chequeo importante para DINOv2 frozen (0.150):** confirmar cómo se extrae el feature del ViT
> (CLS token vs mean-pool de patch tokens). Documentar cuál se usa y usar el **mismo** criterio en
> frozen y finetune, para que el 0.150 no sea un artefacto de pooling.

---

## 5. Gráficos a producir

Guardar en `outputs/figures/` a 150 dpi.

| Fig | Qué es | Cómo |
|---|---|---|
| **Fig-1** | Escalera de encoders (ARI@eps\*), ganador resaltado | `plot_encoder_staircase(res_T3, metric="ARI")` |
| **Fig-2** | Comparación de losses (barras) | `plot_encoder_staircase(res_T1, metric="ARI")` o barras agrupadas eps0/eps\* |
| **Fig-3** | Augmentation strong vs heavy (barras agrupadas por backbone) | matplotlib; resaltar que heavy resta |
| **Fig-4** | Barrido de eps del **ganador** (DINOv2-large): silhouette + ARI-ref, marcar eps\* | `plot_eps_sweep(rows, eps_star)` con el `eps_sweep.csv` de §2 |
| **Fig-5** | **Homogeneity vs Completeness** por encoder (barras apareadas o scatter) | ilustra "puro pero sobre-particionado"; usar T3 |
| **Fig-6** | Curvas de aprendizaje de las 4 losses (train_loss + val por época) | `plot_learning_curves(logs)` — **requiere logs por época** (ver §6) |

> Fig-4 es importante para la defensa: muestra la **ventana finísima de eps** (margen inter-vaca
> < 0.05) que sostiene el salto de 0.716 → 0.831. Si la curva de silhouette tiene un pico claro en
> eps\*, es evidencia visual de que la selección label-free es legítima.

---

## 6. Curvas de aprendizaje (Fig-6) — qué exportar

`plot_learning_curves` espera:
```python
logs = {
  "SupCon":  {"epoch": [...], "train_loss": [...], "val_metric": [...]},
  "CE":      {...}, "ArcFace": {...}, "Triplet": {...},
}
```
- Si los logs de entrenamiento existen (TensorBoard / CSV / prints), parsearlos a ese formato.
- `val_metric` puede ser val-loss, val-acc o (mejor) ARI de validación por época si se guardó.
- Si **no** hay logs por época guardados, Fig-6 es opcional; anotarlo como limitación y seguir.
  (No vale reentrenar solo para esto salvo que sobre tiempo.)

---

## 7. Checklist "listo para entregar"

- [ ] `data/zenodo_1554_index.csv` generado y reutilizado por todos.
- [ ] Sanity gate pasado (imagenet 0.461, dinov2L 0.716/0.831).
- [ ] `metrics.json` por modelo (eps0 + eps\*), con la batería completa.
- [ ] T1, T2, T3 en Markdown, columnas nuevas completadas.
- [ ] Fig-1 a Fig-5 generadas (Fig-6 si hay logs).
- [ ] **eps 100% label-free:** confirmar que la selección usa piso **absoluto**, no `n_true`
      (quita el mini-leak que era limitación declarada).
- [ ] Criterio de pooling de DINOv2 documentado (para el 0.150 frozen).
- [ ] `min_cluster_size` usado documentado en cada tabla.

### Notas para el informe (no son código, pero cerrarlas evita que las piquen)
1. **Encuadre de la métrica:** ARI es primaria por **deployability** (no conocés el número de
   identidades → Rank-1/retrieval no es lo desplegable; clustering sin saber k sí). **Sacar** del
   doc la justificación de "Rank-1 se infla con el fondo": en Zenodo el hocico va con mucho zoom y
   **no hay fondo**, así que ese argumento no aplica a este target (venía de las caras de la Etapa 2).
2. **Rank-1 como secundaria:** en Rank-1 SupCon (0.810) ≈ ImageNet (0.803) — la métrica satura y
   hasta features genéricas la pasan; la ventaja del especialista aparece en ARI (tarea más dura).
   Ese es el argumento correcto, más fuerte que "background".
3. **Mecanismo del resultado positivo (párrafo a escribir):** augmentation fuerza invariancia a la
   foto → SupCon premia esa geometría → transfiere a vacas nuevas. ArcFace comprime hacia
   prototipos del source que no existen para vacas nuevas → transfiere peor. Aug + SupCon son el
   mismo mecanismo.
4. **Loss test:** encuadrar como "bajo presupuesto fijo de 60 ep (parte del diseño: no
   sobre-especializar)"; Triplet quedó sub-entrenado *en ese presupuesto* (no es que sea inferior).

### Opcional (requiere reentrenar, no solo re-evaluar)
- [ ] 2ª seed para **SupCon vs CE** (+0.050) y **base vs large** (+0.072) → barra de ruido real.
      La re-evaluación es determinista, así que esto sólo aporta si se reentrena. Si hay GPU, son
      ~4 corridas y blindan los dos claims más finos.

---

## Apéndice — Valores esperados (sanity check de reproducción)

```
modelo                    ARI@eps0   eps*    ARI@eps*    NMI   #clust  noise
imagenet                    0.461    0.048     0.566    0.939    351    0.05
resnet50_supcon             0.542    0.020     0.641    0.945    352    0.04
dinov2_base_supcon_strong   0.687    0.030     0.759    0.960    329    0.02
dinov2_large_supcon_strong  0.716    0.048     0.831    0.971    306    0.01

losses (ResNet-50, ARI@eps0):  SupCon 0.542 | CE 0.492 | ArcFace 0.267 | Triplet 0.210
augmentation (ARI):  ResNet strong 0.542 / heavy 0.483 | DINOv2-base strong 0.687 / heavy 0.663
```
