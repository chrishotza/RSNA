from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

LABELS = [
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion",
    "Synovitis","Baker's","Contusion","Fracture",
]

@dataclass(frozen=True)
class A6Variant:
    name: str
    policy: str
    focus_targets: tuple[str,...]
    attention_prior_strength: float
    focus_loss_multiplier: float

A6_VARIANTS = {
    "A6-PF": A6Variant(
        name="A6-PF",
        policy="pf_axial_dense",
        focus_targets=("PF OA",),
        attention_prior_strength=0.35,
        focus_loss_multiplier=2.0,
    ),
    "A6-SYN": A6Variant(
        name="A6-SYN",
        policy="synovitis_fluid_dense",
        focus_targets=("Synovitis",),
        attention_prior_strength=0.40,
        focus_loss_multiplier=2.0,
    ),
    "A6-LAT": A6Variant(
        name="A6-LAT",
        policy="lateral_compartment_dense",
        focus_targets=("Lateral OA","Lateral Meniscus"),
        attention_prior_strength=0.35,
        focus_loss_multiplier=1.75,
    ),
}


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    norm=lambda s:"".join(ch.lower() for ch in str(s) if ch.isalnum())
    by={norm(c):c for c in df.columns}
    uid=by.get(norm("StudyInstanceUID"))
    if uid is None:
        raise ValueError("missing StudyInstanceUID")
    out=pd.DataFrame({"StudyInstanceUID":df[uid].astype(str)})
    for lab in LABELS:
        c=by.get(norm(lab))
        if c is None:
            raise ValueError(f"missing label {lab}")
        out[lab]=pd.to_numeric(df[c],errors="coerce")
    return out


def prepare_clean_a6_targets(
    competition_train: pd.DataFrame,
    pseudo_labels: pd.DataFrame,
) -> dict:
    """Split gold and non-gold before any A6 training target is created.

    Gold studies are identified only by finite expert labels in competition train.csv.
    They are returned for later evaluation, never merged into weak targets.
    """
    gold=_canonicalize(competition_train)
    pseudo=_canonicalize(pseudo_labels).drop_duplicates("StudyInstanceUID",keep="last")

    gvals=gold[LABELS].to_numpy(np.float32)
    gold_row=np.isfinite(gvals).any(axis=1)

    gold_uids=gold.loc[gold_row,"StudyInstanceUID"].astype(str).tolist()
    train_uids=gold.loc[~gold_row,"StudyInstanceUID"].astype(str).tolist()

    p=pseudo.set_index("StudyInstanceUID").reindex(train_uids)
    if len(p)!=len(train_uids):
        raise RuntimeError("pseudo-label reindex drift")
    weak=p[LABELS].to_numpy(np.float32)
    weak=np.where(np.isfinite(weak),weak,0.5).astype(np.float32)

    if not np.isfinite(weak).all():
        raise RuntimeError("non-finite A6 weak targets")
    if ((weak<0)|(weak>1)).any():
        raise RuntimeError("A6 weak target outside [0,1]")
    if set(train_uids) & set(gold_uids):
        raise RuntimeError("gold leakage into A6 train UIDs")

    return {
        "train_uids":train_uids,
        "train_targets":weak,
        "gold_uids":gold_uids,
        "gold_values":gvals[gold_row],
        "n_train":len(train_uids),
        "n_gold":len(gold_uids),
    }


def target_loss_weights(variant: A6Variant) -> np.ndarray:
    w=np.ones(len(LABELS),dtype=np.float32)
    for t in variant.focus_targets:
        if t not in LABELS:
            raise ValueError(f"unknown target {t}")
        w[LABELS.index(t)]=float(variant.focus_loss_multiplier)
    return w


def weighted_point_loss_elements(loss_elements, variant: A6Variant):
    """Apply fixed target emphasis to per-example/per-target loss elements."""
    import torch
    if loss_elements.ndim != 2 or loss_elements.shape[1] != len(LABELS):
        raise ValueError("expected [B,12] loss elements")
    w=torch.as_tensor(
        target_loss_weights(variant),
        dtype=loss_elements.dtype,
        device=loss_elements.device,
    )
    return loss_elements*w[None,:]
