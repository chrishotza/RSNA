from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PLANE_IDS = {
    "UNKNOWN": 0,
    "SAGITTAL": 1,
    "CORONAL": 2,
    "AXIAL": 3,
}


@dataclass(frozen=True)
class SliceRecord:
    sop_instance_uid: str
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, float, float, float, float, float] | None
    instance_number: int | None


def _vec(values: Sequence[float], n: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"expected finite vector of length {n}")
    return a


def physical_slice_coordinate(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
) -> float:
    """Project ImagePositionPatient onto the DICOM slice normal."""
    pos = _vec(image_position, 3)
    ori = _vec(image_orientation, 6)
    row = ori[:3]
    col = ori[3:]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        raise ValueError("degenerate ImageOrientationPatient")
    normal = normal / norm
    return float(np.dot(pos, normal))


def sort_slice_records(records: Sequence[SliceRecord]) -> list[SliceRecord]:
    """Deterministic anatomical sorting with geometry-first fallback.

    Geometry is used only when every slice has a valid physical coordinate.
    Otherwise the entire series falls back to InstanceNumber, never a mixture
    of incomparable coordinate systems.
    """
    if not records:
        raise ValueError("empty series")

    coords: list[float] = []
    geometry_ok = True
    for r in records:
        if r.image_position is None or r.image_orientation is None:
            geometry_ok = False
            break
        try:
            coords.append(physical_slice_coordinate(r.image_position, r.image_orientation))
        except ValueError:
            geometry_ok = False
            break

    if geometry_ok:
        paired = list(zip(records, coords))
        paired.sort(key=lambda rc: (rc[1], rc[0].sop_instance_uid))
        return [r for r, _ in paired]

    if not all(r.instance_number is not None for r in records):
        raise ValueError("series lacks complete geometry and complete InstanceNumber fallback")

    return sorted(records, key=lambda r: (int(r.instance_number), r.sop_instance_uid))


def deterministic_anchor_centers(
    n_slices: int,
    n_anchors: int = 12,
    low_fraction: float = 0.04,
    high_fraction: float = 0.96,
) -> list[int]:
    """Evenly cover a series while staying deterministic on short stacks."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if n_anchors <= 0:
        raise ValueError("n_anchors must be positive")
    if not (0.0 <= low_fraction <= high_fraction <= 1.0):
        raise ValueError("invalid anchor fractions")

    if n_slices == 1:
        return [0]

    lo = low_fraction * (n_slices - 1)
    hi = high_fraction * (n_slices - 1)
    raw = np.linspace(lo, hi, num=min(n_anchors, n_slices))
    centers = np.rint(raw).astype(int)
    centers = np.clip(centers, 0, n_slices - 1)

    # Preserve order while dropping duplicates created on short stacks.
    seen = set()
    out = []
    for c in centers.tolist():
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def adjacent_triplet(center: int, n_slices: int) -> tuple[int, int, int]:
    """Return a 2.5D triplet with edge replication."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if not (0 <= center < n_slices):
        raise ValueError("center outside series")
    return (
        max(0, center - 1),
        center,
        min(n_slices - 1, center + 1),
    )


def canonical_plane(value: str | None) -> str:
    s = (value or "").strip().upper()
    if s.startswith("SAG"):
        return "SAGITTAL"
    if s.startswith("COR"):
        return "CORONAL"
    if s.startswith("AX"):
        return "AXIAL"
    return "UNKNOWN"


def _binary_flag(value: bool | int | float | str | None) -> int:
    """Map only explicit truthy protocol values to 1; unknown/NaN -> 0."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "t"}:
            return 1
        if text in {"0", "false", "no", "n", "f", "", "nan", "none", "<na>", "unknown"}:
            return 0
        try:
            value = float(text)
        except ValueError:
            return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(number):
        return 0
    return 1 if number == 1.0 else 0


def acquisition_type_id(
    anatomical_plane: str | None,
    fluid_sensitive: bool | int | float | str | None,
    fat_suppression: bool | int | float | str | None,
) -> int:
    """Stable compact ID: plane x fluid x fat suppression.

    Unknown protocol flags map to 0 rather than Python's dangerous bool(NaN).
    IDs occupy [0, 15], leaving room in the model vocabulary for future
    sequence metadata without changing existing IDs.
    """
    plane_id = PLANE_IDS[canonical_plane(anatomical_plane)]
    fluid = _binary_flag(fluid_sensitive)
    fat = _binary_flag(fat_suppression)
    return plane_id * 4 + fluid * 2 + fat


def build_series_window_manifest(
    sorted_sops: Sequence[str],
    n_anchors: int = 12,
) -> list[dict[str, object]]:
    centers = deterministic_anchor_centers(len(sorted_sops), n_anchors=n_anchors)
    out: list[dict[str, object]] = []
    for rank, center in enumerate(centers):
        a, b, c = adjacent_triplet(center, len(sorted_sops))
        out.append(
            {
                "anchor_rank": rank,
                "center_index": center,
                "sop_triplet": [sorted_sops[a], sorted_sops[b], sorted_sops[c]],
            }
        )
    return out
