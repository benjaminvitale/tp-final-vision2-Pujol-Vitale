"""reid_dataset.py — entries + gallery/probe split by identity (Phase 6).

Splits:
- `split_gallery_probe`: random split within each individual.
- `split_gallery_probe_by_session`: groups by SESSION (filename timestamp) and does not
  split a session between gallery and probe (avoids matching twin photos from the same burst).

`gallery_shots`: if passed (e.g. 1 = single-shot), the gallery gets exactly that number
of images (or sessions) per individual and the rest goes to probe. Single-shot reduces
burst-photo leakage: with only one reference per individual it is harder to match by photo
similarity instead of biometrics.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BURST_SUFFIX = re.compile(r"-\d+$")   # "-00", "-01", ... at the end of the stem


def session_id(rel_path: str) -> str:
    """Session ID = file stem without the burst suffix `-NN`."""
    return _BURST_SUFFIX.sub("", Path(rel_path).stem)


def session_stats(entries: list[dict]) -> dict:
    """Diagnose whether a session-based split is meaningful for this dataset.

    The session split (and the anti-burst control) only works if filenames actually
    encode a capture group: several photos share a session stem and each individual
    spans MORE THAN ONE session. If every photo maps to its own session, `session_id()`
    is not finding real sessions and the split degenerates to a per-image split — the
    metrics would then measure photo similarity, not biometrics. This function surfaces
    that before we trust any session-based number.

    Returns per-dataset counts plus `session_split_is_meaningful` (True when a good
    fraction of individuals have >1 session and sessions bundle >1 photo on average).
    """
    by_label: dict[int, dict[str, int]] = {}
    for e in entries:
        by_label.setdefault(e["label"], {}).setdefault(session_id(e["path"]), 0)
        by_label[e["label"]][session_id(e["path"])] += 1

    n_ids = len(by_label)
    sessions_per_id = [len(s) for s in by_label.values()]
    imgs_per_session = [c for s in by_label.values() for c in s.values()]
    ids_multi_session = sum(1 for k in sessions_per_id if k > 1)
    n_total_sessions = sum(sessions_per_id)
    n_total_imgs = sum(imgs_per_session)

    frac_multi = ids_multi_session / max(n_ids, 1)
    avg_imgs_per_session = n_total_imgs / max(n_total_sessions, 1)
    # Meaningful if most individuals span >1 session AND sessions actually bundle photos.
    meaningful = frac_multi >= 0.5 and avg_imgs_per_session >= 1.5

    return {
        "n_ids": n_ids,
        "n_images": n_total_imgs,
        "n_sessions": n_total_sessions,
        "avg_sessions_per_id": round(n_total_sessions / max(n_ids, 1), 2),
        "avg_images_per_session": round(avg_imgs_per_session, 2),
        "ids_with_multi_session": ids_multi_session,
        "frac_ids_multi_session": round(frac_multi, 3),
        "max_images_in_one_session": max(imgs_per_session) if imgs_per_session else 0,
        "session_split_is_meaningful": bool(meaningful),
    }


def one_per_session_indices(entries: list[dict], seed: int = 0) -> list[int]:
    """Indices of one representative photo per (individual, session).

    Anti-burst control for CLUSTERING: keeping a single photo per session forces any
    recovered cluster to span DIFFERENT sessions of the same animal, so ARI/NMI measure
    biometric identity rather than near-duplicate burst photos clustering trivially.
    Deterministic given `seed`.
    """
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[int]] = {}
    for i, e in enumerate(entries):
        groups.setdefault((e["label"], session_id(e["path"])), []).append(i)
    keep = [rng.choice(idxs) for idxs in groups.values()]
    return sorted(keep)


def session_single_shot_indices(entries: list[dict], seed: int = 0,
                                min_sessions: int = 2) -> tuple[list[int], list[int]]:
    """Index-based single-shot-by-session split → (gallery_idx, probe_idx).

    Same protocol as `split_gallery_probe_by_session(gallery_shots=1)` but returns
    positional indices so precomputed embeddings can be reused for retrieval without
    re-embedding. One SESSION per individual goes to the gallery, the rest to probe.
    Individuals with fewer than `min_sessions` sessions are dropped.
    """
    by_label: dict[int, dict[str, list[int]]] = {}
    for i, e in enumerate(entries):
        by_label.setdefault(e["label"], {}).setdefault(session_id(e["path"]), []).append(i)
    rng = random.Random(seed)
    gallery, probe = [], []
    for _lab, sessions in sorted(by_label.items()):
        sids = list(sessions.keys())
        if len(sids) < min_sessions:
            continue
        rng.shuffle(sids)
        gal_sid = sids[0]
        for sid in sids:
            (gallery if sid == gal_sid else probe).extend(sessions[sid])
    return sorted(gallery), sorted(probe)


def entries_from_folders(root: Path, max_per_id: int | None = None) -> tuple[list[dict], dict]:
    """<root>/<id>/*.img → (entries [{path,label}], id_map {folder: int})."""
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
                        gallery_frac: float = 0.5,
                        gallery_shots: int | None = None) -> tuple[list[dict], list[dict], dict]:
    """Random split per individual. `gallery_shots` fixes how many images go to gallery/id."""
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
        if gallery_shots is not None:
            n_gal = min(gallery_shots, len(items) - 1)   # at least 1 goes to probe
        else:
            n_gal = min(max(1, round(len(items) * gallery_frac)), len(items) - 1)
        gallery += items[:n_gal]; probe += items[n_gal:]; used += 1
    info = {"split": "single_shot" if gallery_shots == 1 else "random",
            "gallery_shots": gallery_shots, "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped": dropped, "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info


def split_gallery_probe_by_session(entries: list[dict], seed: int = 0, min_sessions: int = 2,
                                   gallery_frac: float = 0.5,
                                   gallery_shots: int | None = None
                                   ) -> tuple[list[dict], list[dict], dict]:
    """Session-based split. `gallery_shots` = how many SESSIONS go to gallery per individual."""
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
        if gallery_shots is not None:
            n_gal = min(gallery_shots, len(sids) - 1)
        else:
            n_gal = min(max(1, round(len(sids) * gallery_frac)), len(sids) - 1)
        gal_sids = set(sids[:n_gal])
        for sid in sids:
            (gallery if sid in gal_sids else probe).extend(sessions[sid])
        used += 1; tot_sessions += len(sids)
    info = {"split": "by_session_single" if gallery_shots == 1 else "by_session",
            "gallery_shots": gallery_shots, "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped_lt_min_sessions": dropped, "min_sessions": min_sessions,
            "avg_sessions_per_id": round(tot_sessions / max(used, 1), 2),
            "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info
