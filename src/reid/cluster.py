"""cluster.py — Unsupervised identity discovery + evaluation (Stage 3, Phase 0).

Clustering answers the core question: *does the encoder discover the identities of a new
field WITHOUT being told how many cattle there are?* We estimate the partition and score
it against the real labels (labels used ONLY to evaluate, never to cluster).

Three clusterers, three roles:
- `dbscan`   : primary estimator over cosine distance. eps is a hyperparameter fixed
               WITHOUT looking at labels; we sweep a grid to expose its sensitivity.
- `hdbscan`  : parameter-light estimator (no eps). Primary "auto" number.
- `kmeans`   : reference with k = the REAL number of cattle. A ceiling "that cheats"
               (it knows the count); useful only as an upper bound.

Metrics (all label-permutation invariant, robust to a wrong cluster count):
- ARI  (adjusted Rand index), NMI (normalized mutual information),
- homogeneity / completeness / V-measure,
- n_clusters_found vs n_clusters_true, and the noise fraction (DBSCAN's -1 points).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score,
                             homogeneity_completeness_v_measure,
                             normalized_mutual_info_score)


def clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Score a predicted partition against the true labels.

    DBSCAN marks outliers as -1; those are counted as noise (each its own singleton for
    the purpose of the metrics, which is how sklearn treats distinct labels). ARI/NMI
    already handle a mismatched number of clusters, so noise is not hidden.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    n_noise = int((pred_labels == -1).sum())
    # Clusters found = distinct non-noise labels.
    found = np.unique(pred_labels[pred_labels != -1])
    homo, comp, vmeas = homogeneity_completeness_v_measure(true_labels, pred_labels)
    return {
        "ARI": round(float(adjusted_rand_score(true_labels, pred_labels)), 4),
        "NMI": round(float(normalized_mutual_info_score(true_labels, pred_labels)), 4),
        "homogeneity": round(float(homo), 4),
        "completeness": round(float(comp), 4),
        "v_measure": round(float(vmeas), 4),
        "n_clusters_found": int(len(found)),
        "n_clusters_true": int(len(np.unique(true_labels))),
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / max(len(pred_labels), 1), 4),
        "n_points": int(len(pred_labels)),
    }


def dbscan(emb: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN over cosine distance. Embeddings are assumed L2-normalized."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb)


def hdbscan(emb: np.ndarray, min_cluster_size: int, min_samples: int | None = None):
    """HDBSCAN (auto eps). Returns labels, or None if no HDBSCAN backend is available.

    Prefers sklearn's built-in (>=1.3); falls back to the `hdbscan` package. Uses cosine
    distance where the backend supports it (sklearn HDBSCAN accepts metric='cosine').
    """
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                       metric="cosine").fit_predict(emb)
    except ImportError:
        pass
    try:
        import hdbscan as _hdb
    except ImportError:
        return None
    # The standalone package does not take cosine directly; on L2-normalized vectors
    # euclidean is a monotone function of cosine, so the neighbourhood structure matches.
    return _hdb.HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples).fit_predict(emb)


def kmeans_reference(emb: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means with k = the real number of cattle. Ceiling that 'cheats' (knows k)."""
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)


def dbscan_grid(emb: np.ndarray, true_labels: np.ndarray, eps_grid, min_samples: int) -> list[dict]:
    """Run DBSCAN across an eps grid; return one metrics row per eps.

    Reporting the whole grid (not the best-ARI eps) keeps eps selection honest: choosing
    eps by looking at ARI would be oracle tuning. The grid shows the sensitivity instead.
    """
    rows = []
    for eps in eps_grid:
        labels = dbscan(emb, eps=float(eps), min_samples=min_samples)
        rows.append({"eps": float(eps), **clustering_metrics(true_labels, labels)})
    return rows


def evaluate_clustering(emb: np.ndarray, true_labels: np.ndarray, *,
                        eps_grid, min_samples: int, hdbscan_min_cluster_size: int,
                        kmeans_seed: int = 0) -> dict:
    """Full Phase-0 clustering panel: DBSCAN grid + HDBSCAN(auto) + k-means(real k)."""
    n_true = int(len(np.unique(true_labels)))
    grid = dbscan_grid(emb, true_labels, eps_grid, min_samples)

    hdb_labels = hdbscan(emb, min_cluster_size=hdbscan_min_cluster_size, min_samples=min_samples)
    hdb = clustering_metrics(true_labels, hdb_labels) if hdb_labels is not None else None

    km = clustering_metrics(true_labels, kmeans_reference(emb, k=n_true, seed=kmeans_seed))
    return {"dbscan_grid": grid, "hdbscan": hdb, "kmeans_real_k": km,
            "n_clusters_true": n_true}
