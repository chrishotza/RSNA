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


def decode_acquisition_type_id(type_id:int)->tuple[str,bool,bool]:
    """Inverse of a3_feature_bank.acquisition_type_id for IDs 0..15."""
    i=int(type_id)
    if not 0 <= i <= 15:
        raise ValueError("acquisition type id outside [0,15]")
    plane_id=i//4
    rem=i%4
    fluid=bool(rem//2)
    fat=bool(rem%2)
    plane={0:"UNKNOWN",1:"SAGITTAL",2:"CORONAL",3:"AXIAL"}[plane_id]
    return plane,fluid,fat


def attention_log_prior_from_token_types(token_types, strength:float=0.35):
    """Build [B,Q,T] additive attention-logit priors from acquisition IDs.

    strength=0 is exactly neutral. We normalize each target's acquisition
    multipliers by its median positive value and take log, so the prior acts
    multiplicatively on attention odds rather than hard-routing tokens.
    """
    import numpy as np
    tt=np.asarray(token_types)
    if tt.ndim != 2:
        raise ValueError("token_types must be [B,T]")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    B,T=tt.shape
    out=np.zeros((B,len(TARGETS),T),dtype=np.float32)
    for b in range(B):
        for t in range(T):
            plane,fluid,fat=decode_acquisition_type_id(int(tt[b,t]))
            for q,target in enumerate(TARGETS):
                p=policy_for_target(target)
                w=acquisition_score(
                    plane=plane,fluid_sensitive=fluid,
                    fat_suppression=fat,policy=p,
                )
                out[b,q,t]=float(w)
    for q in range(len(TARGETS)):
        vals=out[:,q,:]
        pos=vals[vals>0]
        med=float(np.median(pos)) if len(pos) else 1.0
        out[:,q,:]=np.log(np.clip(vals/max(med,1e-8),1e-4,1e4))*float(strength)
    return out
