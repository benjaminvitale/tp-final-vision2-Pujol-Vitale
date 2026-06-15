"""03_train_resnet.py — Fase 4: backbone propio ResNet-50.

PAPER: misma receta que la replicación VGG (resolución 300x300, [0,1] crudo, SGD
mom=0.9, lr=0.001, StepLR(7, 0.1), 50 épocas). ResNet-50 NO es el modelo que replica
el 98.7% (ese es VGG16_BN); lo entrenamos como backbone propio para reutilizar en la
fase futura de domain adaptation.

Corre DOS modos (ver plan.md §Fase 4):
  - freeze   : freeze_backbone=True  → solo FC, como el paper.
  - finetune : freeze_backbone=False → fine-tune completo.

Para cada (modo × semilla): entrena, evalúa en test (global + balanced + por clase) y
agrega media ± std. El MEJOR run (por val accuracy, sin mirar test → sin fuga) se
copia a un checkpoint canónico en outputs/checkpoints/ — ese es el backbone que
reutiliza domain adaptation.

NOSOTROS (el plan deja abierto, ver más abajo): por defecto loss=ce, sin augmentation
y 1 semilla (este es nuestro backbone, no el número de replicación del paper). Todo
override-able por flags si se quiere un backbone más fuerte (p.ej. --aug).

Uso:
    python scripts/03_train_resnet.py                      # freeze + finetune, seed 0, 50 épocas
    python scripts/03_train_resnet.py --modes freeze       # solo modo paper
    python scripts/03_train_resnet.py --aug --loss wce     # backbone más fuerte
    python scripts/03_train_resnet.py --seeds 0 1 2        # varias semillas → mean±std
    python scripts/03_train_resnet.py --smoke              # pipeline rápido en CPU (subset, 2 épocas)
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.evaluate import evaluate_checkpoint  # noqa: E402
from src.train import RunConfig, train_one_run  # noqa: E402
from src.utils import get_device, get_logger, load_json, save_json  # noqa: E402

MODEL = "resnet50"

# modo -> freeze_backbone.
MODES = {
    "freeze":   True,    # PAPER: solo FC.
    "finetune": False,   # NOSOTROS: fine-tune completo (mejor backbone para DA).
}

# IDs de las 8 clases con 4 imágenes (ver DEVIATIONS.md). Reporte focalizado: no
# esconder el peor caso detrás del promedio (principio de integridad de CLAUDE.md).
SMALL_CLASSES = ["cattle_2100", "cattle_3420", "cattle_4549", "cattle_5208",
                 "cattle_5355", "cattle_5630", "cattle_5925", "cattle_8050"]


def _small_class_accs(ev: dict) -> dict[str, float | None]:
    """Accuracy en test de las 8 clases con 4 imágenes (None si no hay muestras)."""
    try:
        label_map = load_json(config.SPLITS_DIR / "label_map.json")  # nombre -> idx
    except FileNotFoundError:
        return {}
    out: dict[str, float | None] = {}
    for name in SMALL_CLASSES:
        idx = label_map.get(name)
        if idx is None:
            continue
        n = ev["per_class_total"][idx]
        a = ev["per_class_acc"][idx]
        out[name] = None if (n == 0 or a != a) else round(a, 4)  # a!=a → NaN
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()),
                    choices=list(MODES.keys()),
                    help="freeze (paper) y/o finetune (full fine-tune).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Semillas. Default 1 sola (este es el backbone, no la replicación).")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--loss", default="ce", choices=["ce", "wce"],
                    help="Loss base. Default ce (receta base del paper).")
    ap.add_argument("--aug", action="store_true",
                    help="Data augmentation en train (off por defecto).")
    ap.add_argument("--smoke", action="store_true",
                    help="Pipeline rápido: 1 semilla, 2 épocas, subset chico, sin pesos ImageNet.")
    args = ap.parse_args()

    log = get_logger("03_train_resnet")
    device = get_device()
    config.ensure_output_dirs()

    # Smoke: validar el pipeline punta a punta, no la ciencia. Subset más chico que
    # en VGG porque el full fine-tune de ResNet-50 en CPU es pesado.
    smoke = args.smoke
    seeds = [0] if smoke else args.seeds
    epochs = 2 if smoke else args.epochs
    max_train = 16 if smoke else None
    max_val = 8 if smoke else None
    max_test = 8 if smoke else None
    pretrained = not smoke  # en smoke evitamos bajar 98 MB de pesos ImageNet

    log.info(f"device={device} | model={MODEL} | modes={args.modes} | "
             f"loss={args.loss} | aug={args.aug} | seeds={seeds} | epochs={epochs} | smoke={smoke}")

    results: dict[str, list[dict]] = {m: [] for m in args.modes}
    # Mejor run global por VAL accuracy (selección sin fuga de test) → backbone canónico.
    best_overall: dict | None = None

    for mode in args.modes:
        freeze = MODES[mode]
        for seed in seeds:
            tag = (f"{MODEL}_{mode}_{args.loss}_{'aug' if args.aug else 'noaug'}_s{seed}"
                   + ("_smoke" if smoke else ""))
            rc = RunConfig(
                model_name=MODEL, loss_kind=args.loss, use_aug=args.aug,
                seed=seed, freeze_backbone=freeze, epochs=epochs, pretrained=pretrained,
                max_train=max_train, max_val=max_val,
                num_workers=0 if smoke else config.NUM_WORKERS,
                tag=tag,
            )
            run = train_one_run(rc, device=device)
            csv_path = config.RESULTS_DIR / f"perclass_{rc.tag}.csv"
            ev = evaluate_checkpoint(run["checkpoint"], device=device,
                                     max_test=max_test, save_csv=csv_path)
            small = _small_class_accs(ev)
            log.info(f"[{mode} s{seed}] val={run['best_val_acc']:.4f} "
                     f"test_global={ev['global_acc']:.4f} test_balanced={ev['balanced_acc']:.4f} "
                     f"| {ev['ms_per_image']} ms/img")
            if small:
                log.info(f"[{mode} s{seed}] small-class acc (4 imgs): {small}")

            entry = {
                "mode": mode,
                "seed": seed,
                "freeze_backbone": freeze,
                "best_val_acc": run["best_val_acc"],
                "test_global_acc": ev["global_acc"],
                "test_balanced_acc": ev["balanced_acc"],
                "ms_per_image": ev["ms_per_image"],
                "small_class_acc": small,
                "checkpoint": run["checkpoint"],
                "perclass_csv": str(csv_path),
            }
            results[mode].append(entry)
            # Selección por val acc (NO por test → sin fuga).
            if best_overall is None or run["best_val_acc"] > best_overall["best_val_acc"]:
                best_overall = entry

    # ---- Resumen media ± std por modo ----
    summary = {}
    for mode, runs in results.items():
        gaccs = [r["test_global_acc"] for r in runs]
        baccs = [r["test_balanced_acc"] for r in runs]
        summary[mode] = {
            "n_runs": len(runs),
            "test_global_mean": round(statistics.mean(gaccs), 4),
            "test_global_std": round(statistics.pstdev(gaccs), 4) if len(gaccs) > 1 else 0.0,
            "test_balanced_mean": round(statistics.mean(baccs), 4),
            "test_balanced_std": round(statistics.pstdev(baccs), 4) if len(baccs) > 1 else 0.0,
            "runs": runs,
        }

    # ---- Backbone canónico: copiar el mejor checkpoint (por val acc) ----
    backbone_name = "resnet50_backbone_smoke.pt" if smoke else "resnet50_backbone.pt"
    backbone_path = config.CHECKPOINTS_DIR / backbone_name
    shutil.copyfile(best_overall["checkpoint"], backbone_path)
    log.info(f"Backbone canónico (mejor por val acc: {best_overall['mode']} "
             f"s{best_overall['seed']}, val={best_overall['best_val_acc']:.4f}) → {backbone_path}")

    out = config.RESULTS_DIR / ("03_resnet_summary_smoke.json" if smoke else "03_resnet_summary.json")
    save_json({
        "model": MODEL, "epochs": epochs, "seeds": seeds, "loss": args.loss,
        "aug": args.aug, "device": device, "smoke": smoke,
        "backbone_checkpoint": str(backbone_path),
        "best_run": {k: best_overall[k] for k in
                     ("mode", "seed", "best_val_acc", "test_global_acc", "test_balanced_acc")},
        "summary": summary,
    }, out)

    # ---- Reporte ----
    print("\n" + "=" * 70)
    print(f"RESUMEN BACKBONE PROPIO — {MODEL}" + (" [SMOKE]" if smoke else ""))
    print("=" * 70)
    print(f"{'modo':10} | {'test global (mean±std)':24} | {'balanced (mean±std)':22}")
    print("-" * 70)
    for m in args.modes:
        s = summary[m]
        print(f"{m:10} | {s['test_global_mean']:.4f} ± {s['test_global_std']:.4f}        "
              f"| {s['test_balanced_mean']:.4f} ± {s['test_balanced_std']:.4f}")
    print("=" * 70)
    print(f"Mejor run (por val acc): {best_overall['mode']} s{best_overall['seed']} "
          f"| val={best_overall['best_val_acc']:.4f} test={best_overall['test_global_acc']:.4f}")
    print(f"Backbone para domain adaptation: {backbone_path}")
    print(f"Resumen: {out}")
    if not smoke:
        print("\nNota: ResNet-50 NO replica el 98.7% del paper (ese es VGG16_BN). Acá "
              "interesa la calidad del backbone para reusar en domain adaptation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
