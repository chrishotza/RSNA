from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TARGETS = [
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion",
    "Synovitis","Baker's","Contusion","Fracture",
]

PLANES=("SAGITTAL","CORONAL","AXIAL","UNKNOWN")

@dataclass(frozen=True)
class ViewPolicy:
    n_anchors: int
    low_fraction: float
    high_fraction: float
    plane_multiplier: dict[str,float]
    fluid_multiplier: float
    fat_suppression_multiplier: float

    def validate(self)->None:
        if self.n_anchors < 1:
            raise ValueError("n_anchors must be positive")
        if not (0.0 <= self.low_fraction < self.high_fraction <= 1.0):
            raise ValueError("invalid depth span")
        for p in PLANES:
            if p not in self.plane_multiplier:
                raise ValueError(f"missing plane {p}")
            if self.plane_multiplier[p] < 0:
                raise ValueError("negative multiplier")
        if self.fluid_multiplier <= 0 or self.fat_suppression_multiplier <= 0:
            raise ValueError("sequence multipliers must be positive")


# Pre-registered A6 view families. These are acquisition/sampling priors only;
# learned target attention still decides what evidence matters.
A6_POLICIES: dict[str,ViewPolicy] = {
    "baseline_balanced": ViewPolicy(
        n_anchors=12, low_fraction=.04, high_fraction=.96,
        plane_multiplier={"SAGITTAL":1.0,"CORONAL":1.0,"AXIAL":1.0,"UNKNOWN":.5},
        fluid_multiplier=1.0, fat_suppression_multiplier=1.0,
    ),
    "pf_axial_dense": ViewPolicy(
        n_anchors=20, low_fraction=.08, high_fraction=.92,
        plane_multiplier={"SAGITTAL":.55,"CORONAL":.70,"AXIAL":1.85,"UNKNOWN":.25},
        fluid_multiplier=1.10, fat_suppression_multiplier=1.10,
    ),
    "synovitis_fluid_dense": ViewPolicy(
        n_anchors=20, low_fraction=.03, high_fraction=.97,
        plane_multiplier={"SAGITTAL":1.15,"CORONAL":1.05,"AXIAL":1.35,"UNKNOWN":.25},
        fluid_multiplier=1.80, fat_suppression_multiplier=1.45,
    ),
    "lateral_compartment_dense": ViewPolicy(
        n_anchors=18, low_fraction=.06, high_fraction=.94,
        plane_multiplier={"SAGITTAL":1.45,"CORONAL":1.55,"AXIAL":.65,"UNKNOWN":.25},
        fluid_multiplier=1.15, fat_suppression_multiplier=1.10,
    ),
}

TARGET_POLICY = {
    "PF OA":"pf_axial_dense",
    "Synovitis":"synovitis_fluid_dense",
    "Lateral OA":"lateral_compartment_dense",
    "Lateral Meniscus":"lateral_compartment_dense",
}


def acquisition_score(
    *,
    plane:str,
    fluid_sensitive:bool,
    fat_suppression:bool,
    policy:ViewPolicy,
)->float:
    policy.validate()
    p=(plane or "UNKNOWN").upper()
    if p not in policy.plane_multiplier:
        p="UNKNOWN"
    s=float(policy.plane_multiplier[p])
    if fluid_sensitive:
        s*=policy.fluid_multiplier
    if fat_suppression:
        s*=policy.fat_suppression_multiplier
    return s


def deterministic_weighted_centers(
    n_slices:int,
    policy:ViewPolicy,
)->list[int]:
    """Deterministic depth coverage for one series.

    The view change is the span and density, not stochastic TTA. This keeps the
    experiment reproducible and avoids conflating sampling diversity with randomness.
    """
    policy.validate()
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if n_slices == 1:
        return [0]
    import numpy as np
    k=min(policy.n_anchors,n_slices)
    lo=policy.low_fraction*(n_slices-1)
    hi=policy.high_fraction*(n_slices-1)
    raw=np.linspace(lo,hi,num=k)
    pts=np.clip(np.rint(raw).astype(int),0,n_slices-1).tolist()
    out=[];seen=set()
    for x in pts:
        if x not in seen:
            seen.add(x);out.append(x)
    return out


def policy_for_target(target:str)->ViewPolicy:
    if target not in TARGETS:
        raise KeyError(target)
    return A6_POLICIES[TARGET_POLICY.get(target,"baseline_balanced")]


def target_acquisition_matrix(series_rows: Iterable[dict]) -> dict[str,list[float]]:
    """Return deterministic target-specific acquisition weights for a study."""
    rows=list(series_rows)
    out={t:[] for t in TARGETS}
    for t in TARGETS:
        pol=policy_for_target(t)
        for r in rows:
            out[t].append(acquisition_score(
                plane=str(r.get("Anatomical_Plane","UNKNOWN")),
                fluid_sensitive=bool(r.get("Fluid_Sensitive",False)),
                fat_suppression=bool(r.get("Fat_Suppression",False)),
                policy=pol,
            ))
    return out
