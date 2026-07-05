"""11_session_diag.py — Stage 3, Task 0 (GATE): does session structure exist?

The Zenodo Muzzle DB names files `cattle_XXXX_DSCF####.jpg`, where `DSCF####` is the
camera's frame counter. Consecutive frame numbers ≈ one continuous capture (a session).
This script asks the ONE question that gates the rest of the analysis:

    Is each animal photographed in ONE session (contiguous DSCF block) or in SEVERAL?

- One session per animal  → an honest cross-session split is IMPOSSIBLE on this target;
  the ~2x over-segmentation cannot be session fragmentation (there is only one session),
  so it is a representation problem, and the anti-leakage control must be near-duplicate
  removal (pHash), not a session split.
- Several sessions per animal → the session-based tasks (purity, disjoint retrieval) are
  valid and can proceed.

Reads filenames only — no model, no GPU. Run it before anything else.

Usage:
    python scripts/11_session_diag.py                       # uses config.TARGET_DIR
    python scripts/11_session_diag.py --target-dir /path/to/BeefCattle_Muzzle_Individualized
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils import get_logger, save_json

log = get_logger("session.diag")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GAPS = (1, 2, 3, 5, 10, 20, 50, 100)


def capture_index(stem: str) -> int | None:
    """The last integer run in the filename stem = the DSCF frame counter.

    'cattle_5507_DSCF0311' → 311 (ignores the 5507 id prefix). Returns None if the name
    carries no number (then no capture-order information is available).
    """
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None


def n_runs(nums: list[int], gap: int) -> int:
    """Number of maximal runs of capture indices where consecutive gap <= `gap`."""
    nums = sorted(nums)
    runs = 1
    for a, b in zip(nums, nums[1:]):
        if b - a > gap:
            runs += 1
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Task 0 — session-structure gate.")
    ap.add_argument("--target-dir", default=str(config.TARGET_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.target_dir)
    if not root.is_dir():
        log.error(f"target-dir does not exist: {root}")
        return 1

    by_id: dict[str, list[int]] = defaultdict(list)
    n_missing = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                ci = capture_index(f.stem)
                if ci is None:
                    n_missing += 1
                else:
                    by_id[d.name].append(ci)

    if not by_id:
        log.error("No numbered filenames found — cannot derive capture order.")
        return 1

    n_animals = len(by_id)
    print("\n" + "=" * 74)
    print("STAGE 3 — TASK 0: SESSION-STRUCTURE GATE (from DSCF frame counter)")
    print(f"target: {root}")
    print(f"animals: {n_animals} | filenames without a number: {n_missing}")
    print("=" * 74)

    print("\nHow many 'sessions' (runs of ~consecutive DSCF) does each animal have?")
    gap_rows = []
    for gap in GAPS:
        s = [n_runs(v, gap) for v in by_id.values()]
        multi = sum(1 for x in s if x >= 2)
        row = {"gap": gap, "mean_sessions_per_animal": round(statistics.mean(s), 2),
               "pct_animals_multi_session": round(100 * multi / n_animals, 1),
               "max_sessions": max(s)}
        gap_rows.append(row)
        print(f"  gap={gap:>3} | sesiones/animal medio={row['mean_sessions_per_animal']:>5} "
              f"| % con >=2 sesiones={row['pct_animals_multi_session']:>5}% "
              f"| max={row['max_sessions']}")

    # Is the counter GLOBAL (one continuous shoot → disjoint contiguous blocks per animal)?
    ranges = sorted((min(v), max(v), name) for name, v in by_id.items())
    overlaps = sum(1 for (lo, hi, _), (lo2, _, _) in zip(ranges, ranges[1:]) if lo2 <= hi)
    print(f"\nDSCF ranges overlapping the next animal's: {overlaps}/{len(ranges) - 1}")
    print("first 8 [min,max] ranges per animal (sorted by min):")
    for lo, hi, name in ranges[:8]:
        print(f"  {name}: [{lo},{hi}]  (width {hi - lo + 1}, {len(by_id[name])} photos)")

    print("\nexamples (sorted DSCF indices):")
    for name in list(by_id)[:6]:
        print(f"  {name}: {sorted(by_id[name])}")

    # --- heuristic verdict (a hint, not a hard gate) ---
    row10 = next(r for r in gap_rows if r["gap"] == 10)
    frac_overlap = overlaps / max(len(ranges) - 1, 1)
    multi_session = row10["pct_animals_multi_session"] >= 30 and frac_overlap >= 0.2
    print("\n" + "-" * 74)
    if multi_session:
        print("READING: MULTI-SESSION likely → session-based tasks (purity, disjoint "
              "retrieval) are valid. Proceed with the brief as written.")
    else:
        print("READING: ~ONE SESSION PER ANIMAL likely (contiguous DSCF, disjoint ranges) "
              "→ cross-session split is not possible here. The ~2x over-segmentation is a "
              "REPRESENTATION problem, not session leakage; anti-leakage control → pHash "
              "near-duplicate removal. Confirm by eye with the ranges/examples above.")
    print("-" * 74)

    report = {"target_dir": str(root), "n_animals": n_animals,
              "filenames_without_number": n_missing, "gap_sweep": gap_rows,
              "range_overlaps": overlaps, "n_range_pairs": len(ranges) - 1,
              "heuristic_multi_session": bool(multi_session)}
    out = Path(args.out) if args.out else config.RESULTS_DIR / "11_session_diag.json"
    config.ensure_output_dirs()
    save_json(report, out)
    print(f"\nreport saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
