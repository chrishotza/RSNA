from __future__ import annotations

import numpy as np
import torch

LABELS=[
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion",
    "Synovitis","Baker's","Contusion","Fracture",
]
SLOT_SIZES=[26,22,18,12,18]
SLOT_NAMES=["SAG_FS","SAG_NONFS","COR_FS","COR_NONFS","AXIAL"]
SAGITTAL_SLOTS={0,1}

# Pre-registered inference-only priors for the Apex 0.949 CoAtNet.
# Values are multiplicative attention-odds weights, converted to additive log-biases.
TARGET_SLOT_ODDS={
    "PF OA": {
        "SAG_FS":0.70,"SAG_NONFS":0.60,"COR_FS":0.85,"COR_NONFS":0.80,"AXIAL":1.80
    },
    "Synovitis": {
        "SAG_FS":1.35,"SAG_NONFS":0.70,"COR_FS":1.30,"COR_NONFS":0.70,"AXIAL":1.45
    },
    "Lateral OA": {
        "SAG_FS":1.30,"SAG_NONFS":1.05,"COR_FS":1.45,"COR_NONFS":1.25,"AXIAL":0.70
    },
    "Lateral Meniscus": {
        "SAG_FS":1.45,"SAG_NONFS":1.20,"COR_FS":1.35,"COR_NONFS":1.15,"AXIAL":0.55
    },
}


def slot_index_for_depth(D:int)->np.ndarray:
    if int(D)!=sum(SLOT_SIZES):
        raise ValueError(f"expected D={sum(SLOT_SIZES)}, got {D}")
    b=np.cumsum([0]+SLOT_SIZES)
    idx=np.searchsorted(b,np.arange(D),side="right")-1
    return idx.astype(np.int64)


def centers_for_full_coverage(mask:np.ndarray,D:int=96,k:int=94)->list[int]:
    valid=np.where(np.asarray(mask)>0)[0]
    if len(valid)<3:
        valid=np.arange(min(3,D))
    lo,hi=int(valid.min()),int(valid.max())
    cs=[c for c in range(lo+1,hi) if c-1>=lo and c+1<=hi]
    if not cs:
        cs=[max(1,min((lo+hi)//2,D-2))]
    idx=np.linspace(0,len(cs)-1,k).round().astype(int)
    return [cs[i] for i in idx]


def build_target_window_log_prior(
    centers:list[int],
    *,
    D:int=96,
    strength:float=1.0,
)->np.ndarray:
    """Return [Q,K] additive attention priors for Apex window attention."""
    if strength<0:
        raise ValueError("strength must be non-negative")
    sidx=slot_index_for_depth(D)
    c=np.asarray(centers,dtype=int)
    if c.ndim!=1 or ((c<0)|(c>=D)).any():
        raise ValueError("invalid window centers")
    slots=sidx[c]
    out=np.zeros((len(LABELS),len(c)),dtype=np.float32)
    for q,lab in enumerate(LABELS):
        weights=TARGET_SLOT_ODDS.get(lab)
        if not weights:
            continue
        odds=np.asarray([weights[SLOT_NAMES[int(s)]] for s in slots],dtype=np.float32)
        # Median-normalize so the intervention changes relative attention only.
        med=float(np.median(odds))
        out[q]=np.log(np.clip(odds/max(med,1e-8),1e-4,1e4))*float(strength)
    return out


def apex_head_with_prior(model,feats,log_prior:torch.Tensor|None=None):
    """Bit-compatible Apex head when log_prior is None/zero.

    Expects the public RaptorClassifier interface: norm, att, clsW, clsb.
    """
    h=model.norm(feats)
    logits_att=model.att(h)  # [B,K,Q]
    if log_prior is not None:
        if log_prior.ndim==2:
            log_prior=log_prior[None,:,:]
        # incoming [B,Q,K] -> [B,K,Q]
        if log_prior.shape[1:]==(model.n,feats.shape[1]):
            log_prior=log_prior.permute(0,2,1)
        if log_prior.shape != logits_att.shape and log_prior.shape[0]!=1:
            raise ValueError(f"log_prior shape {tuple(log_prior.shape)} != attention {tuple(logits_att.shape)}")
        logits_att=logits_att+log_prior.to(device=logits_att.device,dtype=logits_att.dtype)
    a=torch.softmax(logits_att,dim=1)
    pooled=torch.einsum("bkn,bkf->bnf",a,h)
    return (pooled*model.clsW).sum(-1)+model.clsb


def prior_for_mask(mask,*,strength:float=1.0,k:int=94)->torch.Tensor:
    m=np.asarray(mask)
    D=len(m)
    centers=centers_for_full_coverage(m,D=D,k=k)
    p=build_target_window_log_prior(centers,D=D,strength=strength)
    return torch.from_numpy(p)
