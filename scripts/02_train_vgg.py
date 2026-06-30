"""02_train_vgg.py — Fase 3: replicación completa con VGG16_BN.

PAPER: VGG16_BN, backbone congelado (solo FC), 3 variantes de optimización de
desbalance, 5 semillas, accuracy media ± std en test.

Variantes:
  (a) ce        : Cross-Entropy, sin augmentation
  (b) ce_aug    : Cross-Entropy + data augmentation
  (c) wce       : Weighted Cross-Entropy, sin augmentation

Para cada (variante × semilla): entrena, evalúa en test (global + por clase) y
agrega media ± std. Guarda tabla resumen y CSV por clase del mejor run de cada
variante (con foco en las clases con 4 imágenes).

Uso:
    python scripts/02_train_vgg.py                 # replicación completa (5 semillas, 50 épocas)
    python scripts/02_train_vgg.py --seeds 0 1     # subset de semillas
    python scripts/02_train_vgg.py --epochs 50
    python scripts/02_train_vgg.py --smoke         # pipeline rápido: 1 semilla, 2 épocas, subset
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.evaluate import evaluate_checkpoint  # noqa: E402
from src.train import RunConfig, train_one_run  # noqa: E402
from src.utils import get_device, get_logger, load_json, save_json  # noqa: E402

VARIANTS = {
    "ce":     {"loss_kind": "ce",  "use_aug": False},
    "ce_aug": {"loss_kind": "ce",  "use_aug": True},
    "wce":    {"loss_kind": "wce", "use_aug": False},
    "wce_aug": {"loss_kind": "wce", "use_aug": True},
}

# IDs de las clases con 4 imágenes (las críticas del paper). Para reporte focalizado.
SMALL_CLASSES = ["cattle_2100", "cattle_3420", "cattle_4549", "cattle_5208",
                 "cattle_5355", "cattle_5630", "cattle_5925", "cattle_8050"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.REPLICATE_SEEDS))
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--model", default="vgg16_bn")
    ap.add_argument("--smoke", action="store_true",
                    help="Pipeline rápido: 1 semilla, 2 épocas, subset chico, sin pesos ImageNet.")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignorar progreso previo (de una corrida cortada) y arrancar de cero.")
    args = ap.parse_args()

    log = get_logger("02_train_vgg")
    device = get_device()
    config.ensure_output_dirs()

    # Smoke: validar el pipeline punta a punta, no la ciencia.
    smoke = args.smoke
    seeds = [0] if smoke else args.seeds
    epochs = 2 if smoke else args.epochs
    max_train = 32 if smoke else None
    max_val = 16 if smoke else None
    max_test = 16 if smoke else None
    pretrained = not smoke  # en smoke evitamos bajar 528 MB

    log.info(f"device={device} | model={args.model} | variants={args.variants} | "
             f"seeds={seeds} | epochs={epochs} | smoke={smoke}")

    out = config.RESULTS_DIR / ("02_vgg_summary_smoke.json" if smoke else "02_vgg_summary.json")
    # Progreso intermedio: se reescribe después de cada (variante, semilla) para poder
    # retomar el sweep si el entorno se desconecta a mitad de camino (p.ej. Colab/Kaggle).
    progress_path = out.with_suffix(".progress.json")

    results: dict[str, list[dict]] = {v: [] for v in args.variants}
    done: set[tuple[str, int]] = set()

    if not smoke and not args.fresh and progress_path.exists():
        prev = load_json(progress_path)
        for variant, runs in prev.get("results", {}).items():
            if variant in results:
                results[variant] = list(runs)
                done.update((variant, r["seed"]) for r in runs)
        if done:
            log.info(f"Retomando sweep: {len(done)} corrida(s) ya completadas "
                     f"({progress_path}). Usar --fresh para ignorar y arrancar de cero.")

    for variant in args.variants:
        spec = VARIANTS[variant]
        for seed in seeds:
            if (variant, seed) in done:
                log.info(f"[{variant} s{seed}] ya completada, salteando.")
                continue
            rc = RunConfig(
                model_name=args.model, loss_kind=spec["loss_kind"], use_aug=spec["use_aug"],
                seed=seed, epochs=epochs, pretrained=pretrained,
                max_train=max_train, max_val=max_val,
                num_workers=0 if smoke else config.NUM_WORKERS,
                tag=f"{args.model}_{variant}_s{seed}" + ("_smoke" if smoke else ""),
            )
            run = train_one_run(rc, device=device)
            # Evaluar en test el mejor checkpoint de esta corrida.
            csv_path = config.RESULTS_DIR / f"perclass_{rc.tag}.csv"
            ev = evaluate_checkpoint(run["checkpoint"], device=device,
                                     max_test=max_test, save_csv=csv_path)
            log.info(f"[{variant} s{seed}] val={run['best_val_acc']:.4f} "
                     f"test_global={ev['global_acc']:.4f} test_balanced={ev['balanced_acc']:.4f}")
            results[variant].append({
                "seed": seed,
                "best_val_acc": run["best_val_acc"],
                "test_global_acc": ev["global_acc"],
                "test_balanced_acc": ev["balanced_acc"],
                "ms_per_image": ev["ms_per_image"],
                "checkpoint": run["checkpoint"],
                "perclass_csv": str(csv_path),
            })
            if not smoke:
                save_json({"model": args.model, "epochs": epochs, "seeds": seeds,
                          "device": device, "results": results}, progress_path)

    # ---- Resumen media ± std por variante ----
    summary = {}
    for variant, runs in results.items():
        gaccs = [r["test_global_acc"] for r in runs]
        baccs = [r["test_balanced_acc"] for r in runs]
        summary[variant] = {
            "n_runs": len(runs),
            "test_global_mean": round(statistics.mean(gaccs), 4),
            "test_global_std": round(statistics.pstdev(gaccs), 4) if len(gaccs) > 1 else 0.0,
            "test_balanced_mean": round(statistics.mean(baccs), 4),
            "test_balanced_std": round(statistics.pstdev(baccs), 4) if len(baccs) > 1 else 0.0,
            "runs": runs,
        }

    save_json({"model": args.model, "epochs": epochs, "seeds": seeds,
               "device": device, "smoke": smoke, "summary": summary}, out)

    # ---- Reporte ----
    print("\n" + "=" * 68)
    print(f"RESUMEN REPLICACIÓN — {args.model}" + (" [SMOKE]" if smoke else ""))
    print("=" * 68)
    print(f"{'variante':10} | {'test global (mean±std)':24} | {'balanced (mean±std)':22}")
    print("-" * 68)
    for v in args.variants:
        s = summary[v]
        print(f"{v:10} | {s['test_global_mean']:.4f} ± {s['test_global_std']:.4f}        "
              f"| {s['test_balanced_mean']:.4f} ± {s['test_balanced_std']:.4f}")
    print("=" * 68)
    print(f"Resumen: {out}")
    if not smoke:
        print("\nValidación de éxito esperada (paper): mejor variante ~96-98%+, y "
              "ce_aug/wce deben mejorar la accuracy de las clases con 4 imágenes vs ce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
