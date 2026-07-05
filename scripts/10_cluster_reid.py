"""10_cluster_reid.py — Stage 3, Phase 0: unsupervised re-ID by clustering.

End-to-end pipeline (the reproducible circuit the whole stage is built on):

    frozen encoder → embed the unlabelled target → cluster (discover identities) → evaluate

For each encoder, on the SAME data, we report:
  1. Session diagnostics (is the anti-burst session split even meaningful here?).
  2. Clustering: DBSCAN grid + HDBSCAN(auto) + k-means(real k), on the full set AND on a
     one-photo-per-session view (anti-burst). Metrics: ARI, NMI, n_clusters found vs real.
  3. Retrieval (kNN): Rank-1 / Rank-5 / mAP on the honest single-shot-by-session split.

Labels are used ONLY to evaluate, never to cluster.

Usage:
    # Phase 0 baseline — DINOv2 on the Zenodo muzzle DB
    python scripts/10_cluster_reid.py --encoders dinov2 imagenet

    # add our muzzle-trained encoders (disjoint identities vs their CMPD300 training)
    python scripts/10_cluster_reid.py --encoders dinov2 imagenet resnet-ckpt \
        --ckpt outputs/checkpoints/cmpd300_source.pt --target-dir /path/to/zenodo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.reid.cluster import evaluate_clustering
from src.reid.encoders import build_encoder
from src.reid.eval_reid import rank_metrics
from src.reid.reid_dataset import (entries_from_folders, one_per_session_indices,
                                   session_single_shot_indices, session_stats)
from src.utils import get_logger, save_json

log = get_logger("reid.cluster")


def _best_dbscan(grid: list[dict]) -> dict:
    """Row of the eps grid with the highest ARI (for the summary line only — NOT for
    picking eps in production; the full grid is always saved)."""
    return max(grid, key=lambda r: r["ARI"]) if grid else {}


def run_encoder(spec: str, entries: list[dict], target_dir: Path, args,
                dedup_idx: list[int], gal_idx: list[int], prb_idx: list[int]) -> dict:
    enc = build_encoder(spec, ckpt=args.ckpt)
    emb, lab = enc.embed(entries, data_dir=target_dir,
                         batch_size=args.batch_size, num_workers=args.num_workers)

    # --- clustering: full set and anti-burst (one photo per session) ---
    cl_full = evaluate_clustering(emb, lab, eps_grid=config.DBSCAN_EPS_GRID,
                                  min_samples=config.DBSCAN_MIN_SAMPLES,
                                  hdbscan_min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                                  kmeans_seed=args.seed)
    cl_dedup = evaluate_clustering(emb[dedup_idx], lab[dedup_idx],
                                   eps_grid=config.DBSCAN_EPS_GRID,
                                   min_samples=config.DBSCAN_MIN_SAMPLES,
                                   hdbscan_min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                                   kmeans_seed=args.seed)

    # --- retrieval: single-shot by session ---
    retrieval = None
    if gal_idx and prb_idx:
        retrieval = rank_metrics(emb[prb_idx], lab[prb_idx], emb[gal_idx], lab[gal_idx])

    result = {"encoder": enc.name, "spec": spec, "embed_dim": int(emb.shape[1]),
              "clustering_full": cl_full, "clustering_session_dedup": cl_dedup,
              "retrieval_session_single_shot": retrieval}

    # readable log
    bf = _best_dbscan(cl_full["dbscan_grid"])
    bd = _best_dbscan(cl_dedup["dbscan_grid"])
    log.info(f"[{enc.name}] clustering (anti-burst, one/session):")
    log.info(f"    DBSCAN best-ARI={bd.get('ARI')} (eps={bd.get('eps')}, "
             f"{bd.get('n_clusters_found')} clusters vs {bd.get('n_clusters_true')} real)")
    if cl_dedup["hdbscan"]:
        h = cl_dedup["hdbscan"]
        log.info(f"    HDBSCAN(auto) ARI={h['ARI']} NMI={h['NMI']} "
                 f"({h['n_clusters_found']} clusters vs {h['n_clusters_true']} real)")
    log.info(f"    k-means(real k) ARI={cl_dedup['kmeans_real_k']['ARI']} (ceiling, cheats)")
    if retrieval:
        log.info(f"    retrieval Rank-1={retrieval['rank1']:.3f} mAP={retrieval['mAP']:.3f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 Phase 0 — unsupervised re-ID by clustering.")
    ap.add_argument("--target-dir", default=str(config.TARGET_DIR),
                    help="Unlabelled muzzle dataset root (folders = real ids, eval only).")
    ap.add_argument("--encoders", nargs="+", default=["dinov2", "imagenet"],
                    help="Any of: dinov2 imagenet resnet-ckpt.")
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "cmpd300_source.pt"),
                    help="Checkpoint for the resnet-ckpt encoder.")
    ap.add_argument("--seed", type=int, default=config.REID_SEED)
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--max-per-id", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_output_dirs()
    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        log.error(f"target-dir does not exist: {target_dir}"); sys.exit(1)

    entries, id_map = entries_from_folders(target_dir, max_per_id=args.max_per_id)
    log.info(f"target={target_dir} | {len(id_map)} ids | {len(entries)} images")

    # ---- session diagnostics (resolves the blocking 'is the split real?' question) ----
    diag = session_stats(entries)
    log.info("== SESSION DIAGNOSTICS ==")
    for k, v in diag.items():
        log.info(f"    {k}: {v}")
    if not diag["session_split_is_meaningful"]:
        log.warning("Session split looks DEGENERATE (≈1 photo per session): filenames may "
                    "not encode capture sessions. The anti-burst control is then weak — "
                    "clustering/retrieval on this target may reflect photo similarity. "
                    "Check the filename convention before trusting session-based numbers.")

    # Shared, encoder-independent index sets (computed once, reused for every encoder).
    dedup_idx = one_per_session_indices(entries, seed=args.seed)
    gal_idx, prb_idx = session_single_shot_indices(entries, seed=args.seed,
                                                   min_sessions=args.min_sessions)
    log.info(f"anti-burst: {len(dedup_idx)}/{len(entries)} imgs kept (one per session) | "
             f"retrieval split: gallery={len(gal_idx)} probe={len(prb_idx)}")

    results = {"target_dir": str(target_dir), "n_ids": len(id_map), "n_images": len(entries),
               "session_diagnostics": diag, "seed": args.seed, "encoders": []}
    for spec in args.encoders:
        try:
            results["encoders"].append(
                run_encoder(spec, entries, target_dir, args, dedup_idx, gal_idx, prb_idx))
        except Exception as exc:  # noqa: BLE001 — one bad encoder should not sink the run
            log.error(f"encoder '{spec}' failed: {exc}")
            results["encoders"].append({"encoder": spec, "error": str(exc)})

    out = Path(args.out) if args.out else config.RESULTS_DIR / "10_cluster_summary.json"
    save_json(results, out)
    log.info(f"summary saved to {out}")
    _print_table(results)


def _print_table(results: dict) -> None:
    print("\n" + "=" * 92)
    print("STAGE 3 — PHASE 0: UNSUPERVISED RE-ID BY CLUSTERING")
    print(f"target: {results['target_dir']}  |  {results['n_ids']} cattle, "
          f"{results['n_images']} images")
    d = results["session_diagnostics"]
    print(f"sessions: {d['avg_sessions_per_id']} avg/id, "
          f"{d['frac_ids_multi_session']*100:.0f}% ids multi-session, "
          f"split_meaningful={d['session_split_is_meaningful']}")
    print("=" * 92)
    print("(clustering numbers below = ANTI-BURST view: one photo per session)")
    hdr = f"{'encoder':22} {'DBSCAN ARI':>11} {'HDBSCAN ARI':>12} {'kmeans ARI':>11} " \
          f"{'#clust/real':>12} {'Rank-1':>8} {'mAP':>7}"
    print(hdr); print("-" * 92)
    for e in results["encoders"]:
        if "error" in e:
            print(f"{e['encoder']:22} ERROR: {e['error'][:60]}"); continue
        cd = e["clustering_session_dedup"]
        best = max(cd["dbscan_grid"], key=lambda r: r["ARI"]) if cd["dbscan_grid"] else {}
        hdb = cd["hdbscan"] or {}
        km = cd["kmeans_real_k"]
        r = e["retrieval_session_single_shot"] or {}
        clr = f"{best.get('n_clusters_found','-')}/{cd['n_clusters_true']}"
        print(f"{e['encoder']:22} {best.get('ARI','-'):>11} {hdb.get('ARI','-'):>12} "
              f"{km['ARI']:>11} {clr:>12} "
              f"{r.get('rank1','-'):>8} {r.get('mAP','-'):>7}")
    print("=" * 92)
    print("DBSCAN ARI = best over the eps grid (full grid in the JSON; eps is NOT tuned on "
          "labels in production).\nk-means ARI knows the real k → ceiling that cheats.")


if __name__ == "__main__":
    main()
