from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected [N,Q]")
    if not np.isfinite(x).all():
        raise ValueError("non-finite predictions")
    return pd.DataFrame(x).rank(method="average", pct=True).to_numpy(np.float64)


@dataclass(frozen=True)
class OrthoRouterConfig:
    agreement_threshold: float = 0.70
    disagreement_threshold: float = 0.10
    agreement_temperature: float = 0.08
    disagreement_temperature: float = 0.05
    displacement_cap: float = 0.20


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def ortho_route(
    anchor: np.ndarray,
    specialist_b: np.ndarray,
    specialist_c: np.ndarray,
    cfg: OrthoRouterConfig = OrthoRouterConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Route two orthogonal public specialists into a stronger anchor in rank space.

    Specialists are allowed to move the anchor only when:
    - they agree with each other;
    - their consensus materially disagrees with the anchor.

    The displacement is hard-capped so a weak specialist cannot dominate.
    """
    a = rank01(anchor)
    b = rank01(specialist_b)
    c = rank01(specialist_c)
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("all arms must have identical [N,Q] shape")

    specialist = np.median(np.stack([b, c], axis=0), axis=0)
    agreement = 1.0 - np.abs(b - c)
    disagreement = np.abs(specialist - a)

    gate_agree = _sigmoid(
        (agreement - cfg.agreement_threshold) / cfg.agreement_temperature
    )
    gate_diff = _sigmoid(
        (disagreement - cfg.disagreement_threshold) / cfg.disagreement_temperature
    )
    gate = gate_agree * gate_diff

    delta = specialist - a
    delta = np.clip(delta, -cfg.displacement_cap, cfg.displacement_cap)
    routed = np.clip(a + gate * delta, 0.0, 1.0)
    final = rank01(routed)

    audit = {
        "anchor_rank": a,
        "specialist_rank": specialist,
        "agreement": agreement,
        "disagreement": disagreement,
        "gate": gate,
        "raw_routed": routed,
    }
    return final, audit


def validate_submission_frame(frame: pd.DataFrame, labels: list[str], uids: list[str]) -> None:
    expected = ["StudyInstanceUID", *labels]
    if frame.columns.tolist() != expected:
        raise ValueError(f"column mismatch: {frame.columns.tolist()}")
    got = frame["StudyInstanceUID"].astype(str).tolist()
    if got != list(map(str, uids)):
        raise ValueError("UID order mismatch")
    arr = frame[labels].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite output")
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError("prediction outside [0,1]")
