"""
reid_eval.py — Evaluación de clustering para re-ID bovino no supervisado (Etapa 3).

Batería de métricas + selección de eps label-free + gráficos, lista para el TP.

Uso mínimo:
    from reid_eval import full_metrics, metrics_table
    m = full_metrics(y_true, y_pred)          # y_pred = asignaciones de HDBSCAN (con -1 = ruido)
    print(metrics_table({"dinov2L": m, ...})) # tabla formateada para el informe

Requisitos: numpy, scikit-learn (>=1.3 para HDBSCAN nativo), matplotlib, seaborn.
No requiere el paquete `hdbscan` externo: usa sklearn.cluster.HDBSCAN.
"""

from __future__ import annotations
import numpy as np
from sklearn import metrics as skm
from sklearn.cluster import HDBSCAN


# --------------------------------------------------------------------------- #
# 1) BCubed (estándar en re-ID / clustering de identidades)                    #
# --------------------------------------------------------------------------- #
def bcubed(y_true, y_pred, noise_label=-1, noise_handling="singleton"):
    """
    BCubed precision / recall / F1 (Amigó et al. 2009).

    - precision_i = |misma etiqueta Y mismo cluster que i| / |mismo cluster que i|
    - recall_i    = |misma etiqueta Y mismo cluster que i| / |misma etiqueta que i|
    - P, R = promedio sobre ítems; F1 = media armónica.

    noise_handling: cómo tratar los puntos de ruido (label == noise_label):
      - "singleton": cada punto de ruido es su propio cluster único (recomendado
        para re-ID: penaliza sobre-partición sin fusionar identidades distintas).
      - "exclude":  se descartan del cálculo.
      - "as_is":    se tratan como un único cluster grande (NO recomendado).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).copy()

    if noise_handling == "exclude":
        keep = y_pred != noise_label
        y_true, y_pred = y_true[keep], y_pred[keep]
    elif noise_handling == "singleton":
        mask = y_pred == noise_label
        if mask.any():
            start = int(y_pred.max()) + 1 if y_pred.size else 0
            y_pred[mask] = np.arange(start, start + mask.sum())
    # "as_is": no se toca

    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    # tamaños de cada cluster y de cada etiqueta verdadera
    _, pred_inv = np.unique(y_pred, return_inverse=True)
    _, true_inv = np.unique(y_true, return_inverse=True)
    cluster_size = np.bincount(pred_inv)
    label_size = np.bincount(true_inv)

    # nº de ítems que comparten cluster y etiqueta con cada ítem i
    # (contamos por celda (cluster, etiqueta) y mapeamos de vuelta a cada ítem)
    pair = pred_inv.astype(np.int64) * (true_inv.max() + 1) + true_inv
    _, pair_inv, pair_counts = np.unique(pair, return_inverse=True, return_counts=True)
    correct_i = pair_counts[pair_inv]  # para cada ítem: |mismo cluster Y misma etiqueta|

    precision = np.mean(correct_i / cluster_size[pred_inv])
    recall = np.mean(correct_i / label_size[true_inv])
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


# --------------------------------------------------------------------------- #
# 2) Batería completa de métricas                                              #
# --------------------------------------------------------------------------- #
def full_metrics(y_true, y_pred, noise_label=-1, bcubed_noise="singleton"):
    """
    Devuelve un dict con todas las métricas.

    Las métricas de sklearn (ARI, AMI, NMI, homogeneity, completeness, V) se
    calculan sobre las etiquetas TAL CUAL (con -1 como una categoría más), que es
    el comportamiento por defecto y probablemente el que ya usa tu pipeline —
    así los números matchean tu tabla actual. BCubed usa bcubed_noise
    ("singleton" por defecto), la convención honesta para re-ID.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_pred)
    n_clusters = int((labels != noise_label).sum())
    n_noise = int((y_pred == noise_label).sum())
    n_true = int(len(np.unique(y_true)))

    bp, br, bf = bcubed(y_true, y_pred, noise_label, noise_handling=bcubed_noise)

    return {
        "ARI": skm.adjusted_rand_score(y_true, y_pred),
        "AMI": skm.adjusted_mutual_info_score(y_true, y_pred),
        "NMI": skm.normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": skm.homogeneity_score(y_true, y_pred),
        "completeness": skm.completeness_score(y_true, y_pred),
        "v_measure": skm.v_measure_score(y_true, y_pred),
        "bcubed_P": bp,
        "bcubed_R": br,
        "bcubed_F1": bf,
        "n_clusters": n_clusters,
        "n_true": n_true,
        "cluster_ratio": n_clusters / n_true if n_true else float("nan"),
        "n_noise": n_noise,
        "noise_frac": n_noise / len(y_pred) if len(y_pred) else float("nan"),
    }


_TABLE_COLS = [
    ("ARI", "ARI", "{:.3f}"),
    ("AMI", "AMI", "{:.3f}"),
    ("homogeneity", "homog", "{:.3f}"),
    ("completeness", "compl", "{:.3f}"),
    ("bcubed_F1", "BCub-F1", "{:.3f}"),
    ("n_clusters", "#clust", "{:d}"),
    ("cluster_ratio", "clust/true", "{:.2f}"),
    ("noise_frac", "noise", "{:.2f}"),
]


def metrics_table(results: dict) -> str:
    """results = {nombre_modelo: dict de full_metrics}. Devuelve tabla en texto."""
    name_w = max(len("modelo"), *(len(k) for k in results))
    header = "modelo".ljust(name_w) + "  " + "  ".join(h.rjust(10) for _, h, _ in _TABLE_COLS)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        row = name.ljust(name_w)
        for key, _, fmt in _TABLE_COLS:
            row += "  " + fmt.format(m[key]).rjust(10)
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3) Selección de eps label-free + barrido                                     #
# --------------------------------------------------------------------------- #
def eps_sweep(
    embeddings,
    y_true=None,
    eps_grid=None,
    min_cluster_size=4,
    metric="cosine",
    min_cluster_floor=None,
):
    """
    Corre HDBSCAN para cada eps del grid y arma la curva de selección.

    Devuelve (rows, eps_star) donde rows es una lista de dicts con:
      eps, silhouette (interno, label-free), ARI (si y_true), n_clusters, noise_frac.
    eps_star = el eps que MAXIMIZA silhouette (label-free) con guardas
    anti-degeneración: descarta soluciones con < min_cluster_floor clusters.

    IMPORTANTE (anti-leak): min_cluster_floor debe ser un piso ABSOLUTO, no n_true.
    Por defecto = 2. No pases el conteo real acá o dejás de ser label-free.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    # L2-normalizar para trabajar en coseno
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if eps_grid is None:
        eps_grid = np.round(np.arange(0.0, 0.121, 0.005), 3)
    if min_cluster_floor is None:
        min_cluster_floor = 2

    rows = []
    for eps in eps_grid:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_epsilon=float(eps),
            copy=True,
        )
        y_pred = clusterer.fit_predict(X)
        mask = y_pred != -1
        n_clusters = len(np.unique(y_pred[mask])) if mask.any() else 0
        noise_frac = float((~mask).mean())

        # silhouette interno (solo sobre puntos no-ruido, y sólo si hay >=2 clusters)
        sil = np.nan
        if n_clusters >= 2 and mask.sum() > n_clusters:
            try:
                sil = skm.silhouette_score(X[mask], y_pred[mask], metric="cosine")
            except Exception:
                sil = np.nan

        row = {
            "eps": float(eps),
            "silhouette": float(sil) if sil == sil else np.nan,
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
        }
        if y_true is not None:
            row["ARI"] = float(skm.adjusted_rand_score(y_true, y_pred))  # oráculo, solo para graficar
        rows.append(row)

    # eps* label-free: max silhouette con guarda de piso absoluto de #clusters
    valid = [r for r in rows if r["n_clusters"] >= min_cluster_floor and r["silhouette"] == r["silhouette"]]
    eps_star = max(valid, key=lambda r: r["silhouette"])["eps"] if valid else float(eps_grid[0])
    return rows, eps_star


# --------------------------------------------------------------------------- #
# 4) Gráficos (matplotlib + seaborn)                                           #
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook")


def plot_learning_curves(logs, out="fig_learning_curves.png", title=None):
    """
    logs: dict {nombre_modelo: {"epoch":[...], "train_loss":[...], "val_metric":[...]}}
    Si un modelo no tiene val_metric, se omite ese panel para él.
    """
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, d in logs.items():
        ep = d["epoch"]
        if "train_loss" in d:
            axes[0].plot(ep, d["train_loss"], label=name, linewidth=2)
        if "val_metric" in d:
            axes[1].plot(ep, d["val_metric"], label=name, linewidth=2)
    axes[0].set_title("Pérdida de entrenamiento", fontweight="bold")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Métrica de validación", fontweight="bold")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Val")
    for ax in axes:
        ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_encoder_staircase(results, metric="ARI", out="fig_staircase.png", order=None):
    """
    results: {nombre_modelo: dict de full_metrics}. Barra horizontal ordenada
    por `metric`, resaltando el mejor. Ideal para 'la escalera del encoder'.
    """
    import matplotlib.pyplot as plt
    _style()
    items = list(results.items())
    if order:
        items = [(k, results[k]) for k in order]
    else:
        items = sorted(items, key=lambda kv: kv[1][metric])
    names = [k for k, _ in items]
    vals = [m[metric] for _, m in items]
    colors = ["#c9ccd1"] * len(vals)
    colors[int(np.argmax(vals))] = "#2b6cb0"

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} por encoder (más alto = mejor)", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_eps_sweep(rows, eps_star=None, out="fig_eps_sweep.png", title="Selección de eps (label-free)"):
    """
    rows: salida de eps_sweep. Grafica silhouette (label-free, eje izq) y, si está,
    ARI (oráculo, eje der) contra eps. Marca eps* (elegido por silhouette).
    """
    import matplotlib.pyplot as plt
    _style()
    eps = [r["eps"] for r in rows]
    sil = [r["silhouette"] for r in rows]
    has_ari = "ARI" in rows[0]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    l1, = ax1.plot(eps, sil, color="#2b6cb0", marker="o", linewidth=2, label="silhouette (label-free)")
    ax1.set_xlabel("cluster_selection_epsilon"); ax1.set_ylabel("silhouette (coseno)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")

    lines = [l1]
    if has_ari:
        ax2 = ax1.twinx()
        l2, = ax2.plot(eps, [r["ARI"] for r in rows], color="#dd6b20", marker="s",
                       linewidth=2, alpha=0.7, label="ARI (oráculo — solo referencia)")
        ax2.set_ylabel("ARI (oráculo)", color="#dd6b20"); ax2.tick_params(axis="y", labelcolor="#dd6b20")
        ax2.spines["top"].set_visible(False)
        lines.append(l2)

    if eps_star is not None:
        ax1.axvline(eps_star, color="#718096", linestyle="--", linewidth=1.5)
        ax1.text(eps_star, ax1.get_ylim()[1], f" eps*={eps_star:.3f}", va="top", fontsize=10, color="#718096")

    ax1.set_title(title, fontweight="bold")
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="lower center")
    ax1.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
