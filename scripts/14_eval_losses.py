"""14_eval_losses.py — Evaluate the 4-loss encoders (+ baselines) on Zenodo (Stage 3).

Same pipeline for every run: pHash-dedup the target → embed (backbone 2048-d, L2-norm) →
HDBSCAN (ARI = PRIMARY) + NMI + #clusters, plus secondary k-means(real-k) and kNN. One run
per loss (no seed sweep); reports the delta over CE-with-augmentation so the loss effect is
isolated from the augmentation effect. Baselines (ImageNet, DINOv2) are the frozen
references to beat.

Checkpoints are looked up as `<ckpt-dir>/cmpd300_<loss>.pt`.

Usage:
    python scripts/14_eval_losses.py --target-dir /path/to/zenodo \
        --ckpt-dir outputs/checkpoints --losses ce arcface supcon triplet \
        --baselines imagenet dinov2
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.reid.cluster import clustering_metrics, hdbscan, kmeans_reference
from src.reid.encoders import build_encoder, resnet50_checkpoint
from src.reid.eval_reid import rank_metrics
from src.reid.phash import dedup_entries
from src.reid.reid_dataset import entries_from_folders
from src.utils import get_logger, save_json

log = get_logger("reid.eval_losses")


def single_shot_idx(labels: np.ndarray, seed: int = 0):
    rng = random.Random(seed)
    by: dict[int, list[int]] = {}
    for i, l in enumerate(labels.tolist()):
        by.setdefault(l, []).append(i)
    gal, prb = [], []
    for _l, idxs in sorted(by.items()):
        if len(idxs) < 2:
            continue
        idxs = idxs[:]; rng.shuffle(idxs)
        gal.append(idxs[0]); prb.extend(idxs[1:])
    return gal, prb


def eval_embeddings(emb, lab, gal, prb, n_true, seed) -> dict:
    hdb = hdbscan(emb, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                  min_samples=config.DBSCAN_MIN_SAMPLES)
    hm = clustering_metrics(lab, hdb) if hdb is not None else {}
    km = clustering_metrics(lab, kmeans_reference(emb, k=n_true, seed=seed))
    r = rank_metrics(emb[prb], lab[prb], emb[gal], lab[gal])
    return {"hdbscan_ari": hm.get("ARI"), "hdbscan_nmi": hm.get("NMI"),
            "n_clusters": hm.get("n_clusters_found"), "kmeans_ari": km["ARI"],
            "rank1": round(r["rank1"], 4), "mAP": round(r["mAP"], 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-dir", default=str(config.TARGET_DIR))
    ap.add_argument("--ckpt-dir", default=str(config.CHECKPOINTS_DIR))
    ap.add_argument("--losses", nargs="*", default=["ce", "arcface", "supcon", "triplet"])
    ap.add_argument("--baselines", nargs="*", default=["imagenet", "dinov2"])
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--no-phash", action="store_true", help="skip near-dup dedup (not recommended)")
    ap.add_argument("--seed", type=int, default=config.REID_SEED)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_output_dirs()
    target = Path(args.target_dir)
    if not target.is_dir():
        log.error(f"target-dir does not exist: {target}"); sys.exit(1)

    entries, id_map = entries_from_folders(target)
    n_true = len(id_map)
    log.info(f"target={target} | {n_true} ids | {len(entries)} imgs")
    if not args.no_phash:
        entries, dinfo = dedup_entries(entries, target, threshold=args.phash_threshold)
        log.info(f"pHash dedup: {dinfo}")
    lab_all = np.array([e["label"] for e in entries])
    gal, prb = single_shot_idx(lab_all, seed=args.seed)
    log.info(f"eval set: {len(entries)} imgs | gallery={len(gal)} probe={len(prb)}")

    def embed_and_eval(enc):
        emb, lab = enc.embed(entries, data_dir=target, batch_size=args.batch_size,
                             num_workers=args.num_workers)
        return eval_embeddings(emb, lab, gal, prb, n_true, args.seed)

    results = {"target": str(target), "n_ids": n_true, "n_eval_images": len(entries),
               "phash": not args.no_phash, "per_loss": {}, "baselines": {}}

    for b in args.baselines:
        try:
            results["baselines"][b] = embed_and_eval(build_encoder(b))
            log.info(f"baseline {b}: {results['baselines'][b]}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"baseline {b} failed: {exc}"); results["baselines"][b] = {"error": str(exc)}

    ckdir = Path(args.ckpt_dir)
    for loss in args.losses:
        ck = ckdir / f"cmpd300_{loss}.pt"
        if not ck.is_file():
            log.warning(f"missing checkpoint {ck} — skipping"); continue
        try:
            m = embed_and_eval(resnet50_checkpoint(ck))
            results["per_loss"][loss] = m
            log.info(f"{loss}: HDBSCAN ARI={m['hdbscan_ari']} nclust={m['n_clusters']} "
                     f"kmeans={m['kmeans_ari']} Rank-1={m['rank1']}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"{loss} failed: {exc}")

    out = Path(args.out) if args.out else config.RESULTS_DIR / "14_loss_comparison.json"
    save_json(results, out)
    log.info(f"summary saved to {out}")
    _print_table(results)


def _print_table(res: dict) -> None:
    print("\n" + "=" * 82)
    print("STAGE 3 — LOSS COMPARISON (PRIMARY = HDBSCAN ARI)")
    print(f"target: {res['target']}  |  {res['n_ids']} ids, {res['n_eval_images']} imgs "
          f"(pHash dedup={res['phash']})")
    print("=" * 82)
    print(f"  {'encoder':16} {'HDBSCAN ARI':>12} {'kmeans ARI':>11} {'Rank-1':>9} {'#clust':>8}")
    print("  " + "-" * 60)
    for b, m in res["baselines"].items():
        if "error" in m:
            print(f"  {b:16} ERROR"); continue
        print(f"  {b:16} {str(m['hdbscan_ari']):>12} {str(m['kmeans_ari']):>11} "
              f"{str(m['rank1']):>9} {str(m['n_clusters']):>8}  (baseline)")
    ce = res["per_loss"].get("ce", {}).get("hdbscan_ari")
    for loss, m in res["per_loss"].items():
        delta = f"  Δce={m['hdbscan_ari']-ce:+.3f}" if ce is not None and m["hdbscan_ari"] is not None else ""
        print(f"  {loss:16} {str(m['hdbscan_ari']):>12} {str(m['kmeans_ari']):>11} "
              f"{str(m['rank1']):>9} {str(m['n_clusters']):>8}{delta}")
    print("=" * 82)
    print("Bar = beat ImageNet/DINOv2 in HDBSCAN ARI. Δce isolates the loss effect from the "
          "shared augmentation.")


if __name__ == "__main__":
    main()
