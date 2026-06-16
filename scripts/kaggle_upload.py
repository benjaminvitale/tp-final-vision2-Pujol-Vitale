"""kaggle_upload.py — Sube a Kaggle el dataset de imágenes y los pesos ImageNet.

OPERATIVO (no es parte del pipeline de entrenamiento). Crea/actualiza dos Kaggle
Datasets para que el notebook los adjunte (Add Input) y `config.py` los autodetecte
en `/kaggle/input/`:

  - imágenes : `data_local/`            → preserva la carpeta BeefCattle_Muzzle_Individualized/
                                          → /kaggle/input/<slug>/BeefCattle_Muzzle_Individualized/
  - pesos    : `imagenet-pretrained/`   → los .pth directos bajo el slug
                                          → /kaggle/input/<slug>/vgg16_bn-...pth

Las rutas se derivan de `config.py` (single source of truth), no se hardcodean.

Credenciales (NO commitear): KAGGLE_USERNAME + KAGGLE_KEY en el entorno, o
`~/.kaggle/kaggle.json`. Crear el token en kaggle.com → Settings → API → Create New Token.

Correr desde el checkout local que tiene la data (no desde Kaggle ni desde el worktree):

    pip install kagglehub
    export KAGGLE_USERNAME=tu_usuario KAGGLE_KEY=xxxxxxxx
    python scripts/kaggle_upload.py --user tu_usuario
    python scripts/kaggle_upload.py --user tu_usuario --only weights --version-notes "v2"

El slug no afecta la autodetección (config.py busca por nombre de carpeta/archivo),
pero conviene uno estable para reusar el dataset entre notebooks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _require_auth() -> None:
    """Falla temprano y claro si no hay credenciales de Kaggle."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return
    raise SystemExit(
        "Faltan credenciales de Kaggle. Exportá KAGGLE_USERNAME + KAGGLE_KEY, o poné "
        "~/.kaggle/kaggle.json (kaggle.com → Settings → API → Create New Token)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"),
                    help="Usuario de Kaggle (o exportá KAGGLE_USERNAME).")
    ap.add_argument("--images-slug", default="cattle-muzzle-db",
                    help="Slug del dataset de imágenes.")
    ap.add_argument("--weights-slug", default="cattle-imagenet-pretrained",
                    help="Slug del dataset de pesos ImageNet.")
    ap.add_argument("--only", choices=["images", "weights"],
                    help="Subir solo uno (default: ambos).")
    ap.add_argument("--version-notes", default=None,
                    help="Notas de versión (si el dataset ya existe, crea una nueva versión).")
    args = ap.parse_args()

    if not args.user:
        raise SystemExit("Indicá --user o exportá KAGGLE_USERNAME.")
    _require_auth()

    import kagglehub  # import tardío: solo si realmente se va a subir.

    # Rutas derivadas de config (single source of truth).
    images_dir = config.DATA_DIR.parent          # data_local/ (contiene la carpeta del dataset)
    weights_dir = config.PRETRAINED_DIR           # imagenet-pretrained/ (contiene los .pth)

    jobs: list[tuple[str, str, Path]] = []
    if args.only in (None, "images"):
        if not config.DATA_DIR.is_dir():
            raise SystemExit(f"No encuentro el dataset de imágenes en {config.DATA_DIR}.")
        jobs.append(("imágenes", f"{args.user}/{args.images_slug}", images_dir))
    if args.only in (None, "weights"):
        if weights_dir is None or not Path(weights_dir).is_dir():
            raise SystemExit(
                "No encuentro los pesos (config.PRETRAINED_DIR es None). Bajá los .pth a "
                "imagenet-pretrained/ primero (ver README) o seteá CATTLE_PRETRAINED_DIR."
            )
        jobs.append(("pesos", f"{args.user}/{args.weights_slug}", Path(weights_dir)))

    for label, handle, path in jobs:
        print(f"\n=== Subiendo {label}: {handle}  ←  {path} ===")
        kwargs = {"version_notes": args.version_notes} if args.version_notes else {}
        kagglehub.dataset_upload(handle, str(path), **kwargs)
        print(f"OK → https://www.kaggle.com/datasets/{handle}")

    print("\nListo. En el notebook de Kaggle: Add Input → buscá los slugs de arriba.")
    print("config.py los autodetecta solo (por nombre de carpeta/archivo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
