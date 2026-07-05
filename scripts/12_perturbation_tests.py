"""12_perturbation_tests.py — Stage 3: encoder robustness ablations.

Three controlled input corruptions (+ clean baseline), each applied to ALL images, to
probe what the frozen encoders actually rely on:

  - border   : keep the central muzzle, replace the outer ring with noise → does the
               encoder use the MUZZLE or the surroundings?
  - rotation : rotate the muzzle (reflect padding, no black corners) → is the embedding
               INVARIANT to pose? (A rotation-sensitive encoder would scatter the same
               cow's photos → a candidate cause of the ~2x over-segmentation.)
  - both     : rotation + border noise.

Per encoder × condition it reports:
  1. drift        : mean cosine(emb(clean), emb(corrupted)) — label-free invariance.
                    1.0 = the corruption does not move the embedding at all.
  2. Rank-1 / mAP : re-embed everything through the corruption, single-shot retrieval.
  3. HDBSCAN ARI, kmeans(real-k) ARI : clustering quality under the corruption.
The interesting quantity is the DELTA vs the clean baseline, not the absolute.

Usage:
    python scripts/12_perturbation_tests.py --encoders dinov2 imagenet --save-examples
    python scripts/12_perturbation_tests.py --encoders dinov2 imagenet resnet-ckpt \
        --ckpt outputs/checkpoints/cmpd300_source.pt --target-dir /path/to/zenodo
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
from src.reid.encoders import build_encoder
from src.reid.eval_reid import rank_metrics
from src.reid.perturb import denormalize, make_conditions
from src.reid.reid_dataset import entries_from_folders
from src.utils import get_logger, save_json

log = get_logger("reid.perturb")


def single_shot_idx(labels: np.ndarray, seed: int = 0) -> tuple[list[int], list[int]]:
    """One random gallery image per id, the rest to probe (per-image single-shot).

    Same split reused across all conditions/encoders so the delta is comparable.
    """
    rng = random.Random(seed)
    by_lab: dict[int, list[int]] = {}
    for i, l in enumerate(labels.tolist()):
        by_lab.setdefault(l, []).append(i)
    gal, prb = [], []
    for l, idxs in sorted(by_lab.items()):
        if len(idxs) < 2:
            continue
        idxs = idxs[:]; rng.shuffle(idxs)
        gal.append(idxs[0]); prb.extend(idxs[1:])
    return gal, prb


def task_metrics(emb: np.ndarray, lab: np.ndarray, gal: list[int], prb: list[int],
                 n_true: int, seed: int) -> dict:
    r = rank_metrics(emb[prb], lab[prb], emb[gal], lab[gal])
    hdb = hdbscan(emb, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                  min_samples=config.DBSCAN_MIN_SAMPLES)
    hdb_ari = clustering_metrics(lab, hdb)["ARI"] if hdb is not None else None
    km_ari = clustering_metrics(lab, kmeans_reference(emb, k=n_true, seed=seed))["ARI"]
    return {"rank1": round(r["rank1"], 4), "mAP": round(r["mAP"], 4),
            "hdbscan_ari": hdb_ari, "kmeans_ari": km_ari}


def save_examples(entries, target_dir, enc, conditions, out_dir: Path, n: int = 4):
    """Save clean vs corrupted example tiles so the corruption can be eyeballed."""
    try:
        import torch
        from torchvision.utils import save_image

        from src.dataset import MuzzleDataset
        ds = MuzzleDataset(entries[:n], transform=enc.transform, data_dir=Path(target_dir))
        batch = torch.stack([ds[i][0] for i in range(min(n, len(ds)))])
        tiles = [denormalize(batch, config.IMAGENET_MEAN, config.IMAGENET_STD)]
        for _name, fn in conditions.items():
            tiles.append(denormalize(fn(batch), config.IMAGENET_MEAN, config.IMAGENET_STD))
        grid = torch.cat(tiles, dim=0)  # rows: clean, border, rotation, both
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "12_perturb_examples.png"
        save_image(grid, p, nrow=n)
        log.info(f"example tiles saved to {p} (rows: clean, {', '.join(conditions)})")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"could not save example tiles: {exc}")


def run_encoder(spec, entries, target_dir, conditions, gal, prb, n_true, args) -> dict:
    enc = build_encoder(spec, ckpt=args.ckpt)
    emb_clean, lab = enc.embed(entries, data_dir=target_dir,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    base = task_metrics(emb_clean, lab, gal, prb, n_true, args.seed)
    log.info(f"[{enc.name}] baseline: Rank-1={base['rank1']} "
             f"HDBSCAN_ARI={base['hdbscan_ari']} kmeans_ARI={base['kmeans_ari']}")

    rows = {"baseline": {**base, "drift": 1.0}}
    for name, fn in conditions.items():
        emb_c, _ = enc.embed(entries, data_dir=target_dir, batch_size=args.batch_size,
                             num_workers=args.num_workers, corrupt=fn)
        drift = float((emb_clean * emb_c).sum(1).mean())   # both are L2-normalized
        m = task_metrics(emb_c, lab, gal, prb, n_true, args.seed)
        rows[name] = {**m, "drift": round(drift, 4)}
        log.info(f"[{enc.name}] {name:8}: drift={drift:.3f} "
                 f"Rank-1={m['rank1']} (Δ{m['rank1']-base['rank1']:+.3f}) "
                 f"HDBSCAN_ARI={m['hdbscan_ari']} kmeans_ARI={m['kmeans_ari']}")
    return {"encoder": enc.name, "spec": spec, "embed_dim": int(emb_clean.shape[1]),
            "conditions": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 — encoder robustness ablations.")
    ap.add_argument("--target-dir", default=str(config.TARGET_DIR))
    ap.add_argument("--encoders", nargs="+", default=["dinov2", "imagenet"])
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "cmpd300_source.pt"))
    ap.add_argument("--keep", type=float, default=0.55, help="central fraction kept (border test)")
    ap.add_argument("--angle", type=float, default=20.0, help="rotation degrees")
    ap.add_argument("--sigma", type=float, default=1.0, help="border noise std (normalized space)")
    ap.add_argument("--seed", type=int, default=config.REID_SEED)
    ap.add_argument("--max-per-id", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-examples", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_output_dirs()
    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        log.error(f"target-dir does not exist: {target_dir}"); sys.exit(1)

    entries, id_map = entries_from_folders(target_dir, max_per_id=args.max_per_id)
    lab_all = np.array([e["label"] for e in entries])
    n_true = len(id_map)
    gal, prb = single_shot_idx(lab_all, seed=args.seed)
    log.info(f"target={target_dir} | {n_true} ids | {len(entries)} imgs | "
             f"gallery={len(gal)} probe={len(prb)}")
    log.info(f"conditions: border(keep={args.keep}) rotation({args.angle}°) both")

    conditions = make_conditions(keep=args.keep, angle=args.angle, sigma=args.sigma)
    if args.save_examples:
        save_examples(entries, target_dir, build_encoder(args.encoders[0], ckpt=args.ckpt),
                      conditions, config.RESULTS_DIR)

    results = {"target_dir": str(target_dir), "n_ids": n_true, "n_images": len(entries),
               "params": {"keep": args.keep, "angle": args.angle, "sigma": args.sigma},
               "encoders": []}
    for spec in args.encoders:
        try:
            results["encoders"].append(
                run_encoder(spec, entries, target_dir, conditions, gal, prb, n_true, args))
        except Exception as exc:  # noqa: BLE001
            log.error(f"encoder '{spec}' failed: {exc}")
            results["encoders"].append({"encoder": spec, "error": str(exc)})

    out = Path(args.out) if args.out else config.RESULTS_DIR / "12_perturbation_summary.json"
    save_json(results, out)
    log.info(f"summary saved to {out}")
    _print_table(results)


def _print_table(results: dict) -> None:
    print("\n" + "=" * 88)
    print("STAGE 3 — PERTURBATION TESTS (drift = cosine to clean; 1.0 = fully invariant)")
    print(f"target: {results['target_dir']}  |  {results['n_ids']} cattle, "
          f"{results['n_images']} imgs  |  params: {results['params']}")
    print("=" * 88)
    for e in results["encoders"]:
        if "error" in e:
            print(f"\n{e['encoder']}: ERROR {e['error'][:70]}"); continue
        print(f"\n{e['encoder']}  (dim {e['embed_dim']})")
        print(f"  {'condition':10} {'drift':>7} {'Rank-1':>8} {'HDBSCAN ARI':>12} {'kmeans ARI':>11}")
        print("  " + "-" * 52)
        for name, r in e["conditions"].items():
            print(f"  {name:10} {r['drift']:>7} {r['rank1']:>8} "
                  f"{str(r['hdbscan_ari']):>12} {str(r['kmeans_ari']):>11}")
    print("\n" + "=" * 88)
    print("Reading: high drift + small Rank-1/ARI drop under 'rotation' → encoder is pose-"
          "invariant. A big drop → rotation-sensitivity, a candidate cause of over-"
          "segmentation.\n'border': small drop → uses the muzzle; big drop → relied on the "
          "surroundings.")


if __name__ == "__main__":
    main()
