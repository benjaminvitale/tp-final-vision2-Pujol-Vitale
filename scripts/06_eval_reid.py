"""06_eval_reid.py — Harness de re-ID (Fase 6): sanity + gap + baseline ImageNet.

- SANITY intra-CMPD300 (`--source-dir`): valida el harness (identidades vistas → Rank-1 alto).
- GAP crudo hocico→cara (`--target-dir`): encoder de hocico sobre caras de Ahmed, sin adaptar.
- Con `--compare-imagenet`: además evalúa un ResNet-50 de ImageNet PURO sobre EL MISMO split
  gallery/probe del target. Sirve para detectar si el número está inflado por fotos parecidas
  dentro de cada individuo (fuga por sesión): si ImageNet puro da casi igual que tu encoder,
  el rendimiento es "gratis" y no mide reconocimiento de hocico.

Uso:
    python scripts/06_eval_reid.py --source-dir .../train --target-dir .../ahmed_subset \\
                                   --max-per-id 10 --compare-imagenet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.reid.embeddings import EmbeddingExtractor
from src.reid.eval_reid import rank_metrics
from src.reid.reid_dataset import entries_from_folders, split_gallery_probe
from src.utils import get_logger, save_json


def score(extractor, gal, prb, root, batch_size):
    ge, gl = extractor.embed(gal, data_dir=root, batch_size=batch_size)
    pe, pl = extractor.embed(prb, data_dir=root, batch_size=batch_size)
    return rank_metrics(pe, pl, ge, gl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Harness de re-ID (Fase 6).")
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "cmpd300_source.pt"))
    ap.add_argument("--source-dir", default=None, help="CMPD300/train para el sanity.")
    ap.add_argument("--target-dir", default=None, help="caras de Ahmed para el gap.")
    ap.add_argument("--compare-imagenet", action="store_true",
                    help="además evaluar ImageNet puro sobre el mismo split del target.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-images", type=int, default=2)
    ap.add_argument("--max-per-id", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    log = get_logger("reid.eval")
    config.ensure_output_dirs()
    if not args.source_dir and not args.target_dir:
        log.error("Pasá --source-dir y/o --target-dir."); sys.exit(1)

    source = EmbeddingExtractor.from_checkpoint(Path(args.ckpt))
    results = {"ckpt": args.ckpt}

    # ---- SANITY intra-CMPD300 ----
    if args.source_dir:
        log.info("== SANITY intra-CMPD300 (identidades VISTAS → plomería) ==")
        entries, _ = entries_from_folders(Path(args.source_dir), max_per_id=args.max_per_id)
        gal, prb, info = split_gallery_probe(entries, seed=args.seed, min_images=args.min_images)
        m = score(source, gal, prb, Path(args.source_dir), args.batch_size)
        results["sanity_cmpd300"] = {**m, **info, "nota": "leakage; solo valida el harness"}
        log.info(f"  source -> Rank-1={m['rank1']:.4f} mAP={m['mAP']:.4f} (esperado ALTO)")

    # ---- GAP sobre Ahmed (mismo split para todos los encoders) ----
    if args.target_dir:
        log.info("== GAP crudo hocico→cara sobre Ahmed (sin adaptar) ==")
        entries, _ = entries_from_folders(Path(args.target_dir), max_per_id=args.max_per_id)
        gal, prb, info = split_gallery_probe(entries, seed=args.seed, min_images=args.min_images)
        log.info(f"  {info['n_ids_used']} individuos | gallery={info['n_gallery']} "
                 f"probe={info['n_probe']} (descartados {info['n_ids_dropped_lt_min']})")

        m_src = score(source, gal, prb, Path(args.target_dir), args.batch_size)
        results["gap_ahmed_source"] = {**m_src, **info,
                                       "encoder": source.name, "nota": "encoder de hocico, sin adaptar"}
        log.info(f"  source(hocico) -> Rank-1={m_src['rank1']:.4f} mAP={m_src['mAP']:.4f}")

        if args.compare_imagenet:
            imagenet = EmbeddingExtractor.from_imagenet()
            m_in = score(imagenet, gal, prb, Path(args.target_dir), args.batch_size)
            results["gap_ahmed_imagenet"] = {**m_in, "encoder": "imagenet_resnet50",
                                             "nota": "baseline tonto (sin hocico); mismo split"}
            log.info(f"  imagenet(puro) -> Rank-1={m_in['rank1']:.4f} mAP={m_in['mAP']:.4f}")

    out = config.RESULTS_DIR / "06_reid_summary.json"
    save_json(results, out)
    log.info(f"resumen guardado en {out}")

    # ---- resumen legible + interpretación ----
    print("\n" + "=" * 66)
    print("FASE 6 — RE-ID")
    print("=" * 66)
    if "sanity_cmpd300" in results:
        s = results["sanity_cmpd300"]
        print(f"SANITY CMPD300 (plomería)   : Rank-1={s['rank1']:.3f}  mAP={s['mAP']:.3f}")
    if "gap_ahmed_source" in results:
        g = results["gap_ahmed_source"]
        print(f"Ahmed — encoder de hocico   : Rank-1={g['rank1']:.3f}  mAP={g['mAP']:.3f}")
    if "gap_ahmed_imagenet" in results:
        i = results["gap_ahmed_imagenet"]
        print(f"Ahmed — ImageNet PURO       : Rank-1={i['rank1']:.3f}  mAP={i['mAP']:.3f}")
        d = results["gap_ahmed_source"]["rank1"] - i["rank1"]
        print("-" * 66)
        print(f"Ventaja del encoder de hocico sobre ImageNet: {d:+.3f} en Rank-1")
        if d < 0.05:
            print("⚠ Ventaja chica: el número parece 'gratis' (fotos parecidas por individuo),")
            print("  no reconocimiento de hocico. El gap crudo NO es confiable así.")
        else:
            print("✓ El encoder de hocico aporta sobre ImageNet: hay señal real.")
    print("=" * 66)


if __name__ == "__main__":
    main()
