"""reid_dataset.py — entries + split gallery/probe por identidad (Fase 6).

Dos formas de partir gallery/probe:

- `split_gallery_probe`: al azar dentro de cada individuo. Simple, pero si las fotos de un
  individuo son casi idénticas (misma ráfaga), infla el número (el modelo matchea gemelas).

- `split_gallery_probe_by_session`: agrupa por SESIÓN y nunca parte una sesión entre gallery
  y probe → el probe se matchea contra fotos de OTRA sesión, no contra su gemela. Es el split
  honesto cuando el dataset tiene varias tomas por individuo (p.ej. Ahmed).

Sesión = el nombre del archivo sin el sufijo `-NN` de enumeración dentro de la ráfaga.
Ej: `IMG20230727070509-00.jpg`, `-01`, `-02` → misma sesión `IMG20230727070509`.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BURST_SUFFIX = re.compile(r"-\d+$")   # "-00", "-01", ... al final del stem


def session_id(rel_path: str) -> str:
    """ID de sesión = stem del archivo sin el sufijo `-NN` de la ráfaga."""
    return _BURST_SUFFIX.sub("", Path(rel_path).stem)


def entries_from_folders(root: Path, max_per_id: int | None = None) -> tuple[list[dict], dict]:
    """<root>/<id>/*.img → (entries [{path,label}], id_map {carpeta: entero})."""
    root = Path(root)
    id_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    id_map = {name: i for i, name in enumerate(id_names)}
    entries: list[dict] = []
    for name in id_names:
        imgs = sorted(f for f in (root / name).iterdir()
                      if f.is_file() and f.suffix.lower() in IMG_EXTS)
        if max_per_id is not None:
            imgs = imgs[:max_per_id]
        for f in imgs:
            entries.append({"path": (Path(name) / f.name).as_posix(), "label": id_map[name]})
    return entries, id_map


def split_gallery_probe(entries: list[dict], seed: int = 0, min_images: int = 2,
                        gallery_frac: float = 0.5) -> tuple[list[dict], list[dict], dict]:
    """Split al azar por individuo (no honesto si hay fotos gemelas)."""
    by_label: dict[int, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    rng = random.Random(seed)
    gallery, probe, used, dropped = [], [], 0, 0
    for lab, items in sorted(by_label.items()):
        if len(items) < min_images:
            dropped += 1
            continue
        items = items[:]; rng.shuffle(items)
        n_gal = min(max(1, round(len(items) * gallery_frac)), len(items) - 1)
        gallery += items[:n_gal]; probe += items[n_gal:]; used += 1
    info = {"split": "random", "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped": dropped, "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info


def split_gallery_probe_by_session(entries: list[dict], seed: int = 0, min_sessions: int = 2,
                                   gallery_frac: float = 0.5) -> tuple[list[dict], list[dict], dict]:
    """Split HONESTO: por sesión. Una sesión entera va a gallery O a probe, nunca partida.

    Individuos con < min_sessions sesiones se descartan (no se pueden separar).
    """
    # label -> {session_id: [entries]}
    by_label: dict[int, dict[str, list[dict]]] = {}
    for e in entries:
        by_label.setdefault(e["label"], {}).setdefault(session_id(e["path"]), []).append(e)

    rng = random.Random(seed)
    gallery, probe, used, dropped, tot_sessions = [], [], 0, 0, 0
    for lab, sessions in sorted(by_label.items()):
        sids = list(sessions.keys())
        if len(sids) < min_sessions:
            dropped += 1
            continue
        rng.shuffle(sids)
        n_gal = min(max(1, round(len(sids) * gallery_frac)), len(sids) - 1)
        gal_sids = set(sids[:n_gal])
        for sid in sids:
            (gallery if sid in gal_sids else probe).extend(sessions[sid])
        used += 1
        tot_sessions += len(sids)
    info = {"split": "by_session", "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped_lt_min_sessions": dropped, "min_sessions": min_sessions,
            "avg_sessions_per_id": round(tot_sessions / max(used, 1), 2),
            "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info
