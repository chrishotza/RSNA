"""Rank-based blending helpers for the 12-column macro-AUC task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def rank_percentile(values: np.ndarray) -> np.ndarray:
    """Convert one prediction column to percentile ranks."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("rank_percentile expects a 1-D array")
    if not np.isfinite(x).all():
        raise ValueError("predictions contain NaN or inf")
    return pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)


def rank_blend(
    parent: np.ndarray,
    candidate: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend two prediction columns after rank conversion.

    ROC-AUC is invariant to strictly monotone transforms, so percentile ranks
    remove scale differences before the blend.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")

    p = rank_percentile(parent)
    c = rank_percentile(candidate)

    return (1.0 - weight) * p + weight * c


def per_label_rank_corr(
    parent: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Spearman-style correlation per target after rank conversion."""
    p = np.asarray(parent, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if p.shape != c.shape or p.ndim != 2:
        raise ValueError("parent and candidate must have the same 2-D shape")

    out = np.empty(p.shape[1], dtype=float)
    for j in range(p.shape[1]):
        pr = rank_percentile(p[:, j])
        cr = rank_percentile(c[:, j])
        out[j] = np.corrcoef(pr, cr)[0, 1]
    return out


@dataclass(frozen=True)
class BlendDecision:
    weights: np.ndarray
    corr_to_parent: np.ndarray
    mean_weight: float
    max_weight: float


def make_decision(
    parent: np.ndarray,
    candidate: np.ndarray,
    weights: Iterable[float],
) -> BlendDecision:
    """Build a logged, auditable per-label blend decision."""
    w = np.asarray(list(weights), dtype=float)
    if parent.ndim != 2 or candidate.shape != parent.shape:
        raise ValueError("parent and candidate must have identical 2-D shapes")
    if w.shape != (parent.shape[1],):
        raise ValueError("one blend weight is required per label")
    if ((w < 0.0) | (w > 1.0)).any():
        raise ValueError("blend weights must be in [0, 1]")

    corr = per_label_rank_corr(parent, candidate)
    return BlendDecision(
        weights=w,
        corr_to_parent=corr,
        mean_weight=float(w.mean()),
        max_weight=float(w.max()),
    )
