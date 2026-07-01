"""download_zenodo.py — Descarga (reanudable) de un record de Zenodo.

Utilidad pensada para datasets grandes que NO entran en disco local: se corre
directamente EN LA VM. Descarga todos los archivos del record, verifica el
checksum md5 publicado por Zenodo y, opcionalmente, descomprime los .zip.

Uso:
    # Cows Frontal Face Dataset (record 10535934), descomprimiendo el zip:
    python scripts/download_zenodo.py --record 10535934 --dest ~/data/cows_face --extract

    # Reintentar / forzar re-descarga:
    python scripts/download_zenodo.py --record 10535934 --dest ~/data/cows_face --extract --force

Notas:
  - Usa `wget -c` (REANUDABLE): si se corta el SSH o la descarga, volvé a correr
    el MISMO comando y sigue desde donde quedó. Conviene correrlo dentro de `tmux`.
  - Verifica md5 contra el que publica Zenodo; aborta si no coincide.
  - Sin dependencias externas: urllib + hashlib (stdlib) y `wget` del sistema.
  - OJO DE DOMINIO: el record 10535934 es "Cows Frontal Face Dataset" (CARAS
    frontales), una modalidad DISTINTA al muzzle (hocico) del pipeline principal.
    Por eso se baja a una carpeta aparte (p.ej. ~/data/cows_face), no a DATA_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://zenodo.org/api/records/{record}"


def fetch_record(record: str) -> dict:
    """Devuelve el JSON del record (metadata + lista de archivos)."""
    with urllib.request.urlopen(API.format(record=record), timeout=60) as r:
        return json.load(r)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """md5 de un archivo, leyendo en bloques (sirve para archivos de varios GB)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Descarga reanudable con wget (-c). Lanza si wget falla."""
    cmd = ["wget", "-c", "--tries=10", "--timeout=60", "-O", str(dest), url]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default="10535934", help="ID del record de Zenodo.")
    ap.add_argument("--dest", default="~/data/zenodo", help="Carpeta destino.")
    ap.add_argument("--extract", action="store_true", help="Descomprimir los .zip.")
    ap.add_argument("--force", action="store_true",
                    help="Re-descargar aunque el md5 ya coincida.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    rec = fetch_record(args.record)
    title = rec.get("metadata", {}).get("title", "?")
    files = rec.get("files", [])
    print(f"Record {args.record}: {title}")
    print(f"{len(files)} archivo(s) -> {dest}\n")

    for f in files:
        key = f["key"]
        url = f["links"]["self"]
        want_md5 = f.get("checksum", "").split(":")[-1]
        out = dest / key

        if out.is_file() and not args.force and want_md5 and md5sum(out) == want_md5:
            print(f"[OK] {key}: ya estaba descargado y verificado.")
        else:
            print(f"[>>] {key}: descargando (reanudable)...")
            download(url, out)
            if want_md5:
                got = md5sum(out)
                if got != want_md5:
                    print(f"[ERROR] {key}: md5 esperado {want_md5}, obtenido {got}. "
                          f"Volvé a correr el comando para reintentar.", file=sys.stderr)
                    return 1
                print(f"[OK] {key}: md5 verificado.")

        if args.extract and out.suffix.lower() == ".zip":
            print(f"[++] {key}: descomprimiendo en {dest} ...")
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
            print(f"[OK] {key}: descomprimido.")

    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())