from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.a3_feature_bank import (
    SliceRecord,
    adjacent_triplet,
    acquisition_type_id,
    canonical_plane,
    sort_slice_records,
)
from src.a6_view_policy import A6_POLICIES


MAX_ANCHORS = max(p.n_anchors for p in A6_POLICIES.values())


def dense_union_centers(n_slices:int)->list[int]:
    """Union of all A6 policy depth grids for a series.

    One shared feature bank is used by all specialists. We retain the union of
    their deterministic depth grids so no specialist is forced to reconstruct
    missing views from an A3-era six-anchor bank.
    """
    if n_slices<=0:
        raise ValueError("n_slices must be positive")
    if n_slices==1:
        return [0]
    pts=set()
    for p in A6_POLICIES.values():
        k=min(p.n_anchors,n_slices)
        lo=p.low_fraction*(n_slices-1)
        hi=p.high_fraction*(n_slices-1)
        vals=np.linspace(lo,hi,k)
        pts.update(np.clip(np.rint(vals).astype(int),0,n_slices-1).tolist())
    return sorted(pts)


def series_manifest(sorted_sops:Sequence[str])->list[dict]:
    centers=dense_union_centers(len(sorted_sops))
    rows=[]
    for rank,c in enumerate(centers):
        a,b,d=adjacent_triplet(c,len(sorted_sops))
        rows.append({
            "anchor_rank":rank,
            "center_index":c,
            "relative_depth":0.0 if len(sorted_sops)==1 else float(c/(len(sorted_sops)-1)),
            "sop_triplet":[sorted_sops[a],sorted_sops[b],sorted_sops[d]],
        })
    return rows


def select_policy_token_mask(
    token_type_ids:Sequence[int],
    relative_depths:Sequence[float],
    target:str,
)->np.ndarray:
    """Deterministic token eligibility mask for one target policy.

    This is intentionally permissive: plane/sequence preference is handled by
    the attention prior. The hard mask only enforces the target's registered
    depth span, preventing out-of-policy edge windows from entering that target.
    """
    from src.a6_view_policy import policy_for_target
    p=policy_for_target(target)
    tt=np.asarray(token_type_ids)
    rd=np.asarray(relative_depths,float)
    if tt.ndim!=1 or rd.shape!=tt.shape:
        raise ValueError("token_type_ids/relative_depths mismatch")
    return (rd>=p.low_fraction)&(rd<=p.high_fraction)


def expected_bank_density(n_slices:int)->dict:
    c=dense_union_centers(n_slices)
    return {
        "n_slices":int(n_slices),
        "n_centers":len(c),
        "coverage_fraction":float(len(c)/max(n_slices,1)),
        "min_center":int(min(c)),
        "max_center":int(max(c)),
    }


@dataclass(frozen=True)
class A6SeriesDescriptor:
    uid:str
    series_uid:str
    plane:str
    fluid_sensitive:bool
    fat_suppression:bool

    @property
    def type_id(self)->int:
        return acquisition_type_id(
            self.plane,
            self.fluid_sensitive,
            self.fat_suppression,
        )

    @property
    def canonical_plane(self)->str:
        return canonical_plane(self.plane)
