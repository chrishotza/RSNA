from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.a3_training import (
    FeatureBankDataset,
    collate_feature_bank,
    macro_auc,
    binary_auc,
)
from src.a3_uam_seq import UAMSeqConfig, UAMSeqHead, unstated_aware_weight
from src.a6_training import A6_VARIANTS, A6Variant, LABELS, target_loss_weights
from src.a6_view_policy import attention_log_prior_from_token_types


@dataclass(frozen=True)
class A6TrainConfig:
    batch_size: int = 64
    epochs: int = 8
    patience: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.02
    point_weight: float = 0.35
    rank_weight: float = 0.65
    min_target_gap: float = 0.35
    temperature: float = 1.0
    projection_dim: int = 384
    metadata_dim: int = 48
    hidden_dim: int = 256
    gru_layers: int = 2
    dropout: float = 0.15
    metadata_vocab: int = 32
    seed: int = 20261009


def assert_training_records_are_gold_free(records: list[dict], gold_uids: Iterable[str]) -> None:
    gold=set(map(str,gold_uids))
    seen=[str(r["uid"]) for r in records]
    overlap=sorted(set(seen)&gold)
    if overlap:
        raise RuntimeError(f"A6 gold leakage: {len(overlap)} training UIDs overlap gold: {overlap[:5]}")
    if len(seen)!=len(set(seen)):
        raise RuntimeError("duplicate training UID in A6 records")


def _target_weight_tensor(variant:A6Variant, device, dtype):
    return torch.as_tensor(target_loss_weights(variant),device=device,dtype=dtype)


def a6_point_loss(logits,targets,variant:A6Variant):
    if logits.shape!=targets.shape or logits.ndim!=2:
        raise ValueError("logits/targets must be [B,12]")
    reliability=unstated_aware_weight(targets)
    per=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    tw=_target_weight_tensor(variant,logits.device,logits.dtype)[None,:]
    w=reliability*tw
    denom=w.sum()
    if float(denom.detach().cpu())<=0:
        return logits.sum()*0.0
    return (per*w).sum()/denom


def a6_pairwise_auc_loss(logits,targets,variant:A6Variant,min_target_gap=.35,temperature=1.0):
    if logits.shape!=targets.shape or logits.ndim!=2:
        raise ValueError("logits/targets must be [B,12]")
    if temperature<=0:
        raise ValueError("temperature must be positive")
    reliability=unstated_aware_weight(targets)
    tw=_target_weight_tensor(variant,logits.device,logits.dtype)
    total=logits.sum()*0.0
    denom_total=logits.new_zeros(())
    for q in range(logits.shape[1]):
        y=targets[:,q]; z=logits[:,q]; w=reliability[:,q]
        dy=y[:,None]-y[None,:]
        valid=dy>=min_target_gap
        if not valid.any():
            continue
        pw=torch.sqrt(torch.clamp(w[:,None]*w[None,:],min=0.0))
        pw=pw*valid.to(pw.dtype)*tw[q]
        denom=pw.sum()
        if float(denom.detach().cpu())<=0:
            continue
        dz=z[:,None]-z[None,:]
        total=total+(F.softplus(-dz/temperature)*pw).sum()
        denom_total=denom_total+denom
    if float(denom_total.detach().cpu())<=0:
        return logits.sum()*0.0
    return total/denom_total


def a6_objective(logits,targets,variant:A6Variant,cfg:A6TrainConfig):
    point=a6_point_loss(logits,targets,variant)
    rank=a6_pairwise_auc_loss(
        logits,targets,variant,
        min_target_gap=cfg.min_target_gap,
        temperature=cfg.temperature,
    )
    total=cfg.point_weight*point+cfg.rank_weight*rank
    return total,{"point_bce":point.detach(),"pair_rank":rank.detach()}


def build_a6_model(feature_dim:int,cfg:A6TrainConfig)->UAMSeqHead:
    return UAMSeqHead(UAMSeqConfig(
        feature_dim=feature_dim,
        projection_dim=cfg.projection_dim,
        metadata_dim=cfg.metadata_dim,
        hidden_dim=cfg.hidden_dim,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        metadata_vocab=cfg.metadata_vocab,
        num_targets=12,
        use_target_type_router=True,
    ))


def torch_attention_prior(token_types:torch.Tensor,variant:A6Variant)->torch.Tensor:
    arr=token_types.detach().cpu().numpy()
    prior=attention_log_prior_from_token_types(
        arr,
        strength=float(variant.attention_prior_strength),
    )
    return torch.as_tensor(prior,device=token_types.device,dtype=torch.float32)


def high_conf_binary(targets:np.ndarray,lo=.10,hi=.90)->np.ndarray:
    y=np.full_like(targets,np.nan,dtype=np.float32)
    y[targets<=lo]=0.0
    y[targets>=hi]=1.0
    return y


def auc_missing(y_true,y_score):
    per=[]
    for q in range(y_true.shape[1]):
        m=np.isfinite(y_true[:,q])
        if m.sum()<2 or len(np.unique(y_true[m,q]))<2:
            per.append(np.nan)
        else:
            per.append(binary_auc(y_true[m,q],y_score[m,q]))
    per=np.asarray(per,float)
    return float(np.nanmean(per)),per


def focus_metric(targets:np.ndarray,pred:np.ndarray,variant:A6Variant)->dict:
    y=high_conf_binary(targets)
    macro,per=auc_missing(y,pred)
    focus_idx=[LABELS.index(t) for t in variant.focus_targets]
    vals=[per[i] for i in focus_idx if np.isfinite(per[i])]
    focus=float(np.mean(vals)) if vals else float("nan")
    return {
        "weak_macro_auc":macro,
        "focus_auc":focus,
        "per_target":dict(zip(LABELS,[None if not np.isfinite(x) else float(x) for x in per])),
    }


def selection_score(metrics:dict)->float:
    """Pre-registered A6 model-selection criterion.

    Focus matters most, but the global weak-label macro is retained to reject a
    specialist that destroys general ordering.
    """
    f=metrics["focus_auc"]; m=metrics["weak_macro_auc"]
    if not np.isfinite(f) or not np.isfinite(m):
        return float("-inf")
    return 0.75*float(f)+0.25*float(m)


def _predict_loader(model,loader,variant,device):
    pred=[]; target=[]; uids=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            feat=batch["features"].to(device)
            typ=batch["token_types"].to(device)
            mask=batch["token_mask"].to(device)
            prior=torch_attention_prior(typ,variant)
            with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                z=model(feat,typ,mask,attention_log_prior=prior)
            pred.append(z.float().cpu().numpy())
            target.append(batch["targets"].numpy())
            uids.extend(batch["uids"])
    return uids,np.concatenate(pred),np.concatenate(target)


def train_one_fold(
    records:list[dict],
    *,
    fold:int,
    variant_name:str,
    gold_uids:Iterable[str],
    cfg:A6TrainConfig=A6TrainConfig(),
    device:torch.device|None=None,
):
    if variant_name not in A6_VARIANTS:
        raise KeyError(variant_name)
    variant=A6_VARIANTS[variant_name]
    assert_training_records_are_gold_free(records,gold_uids)
    tr=[r for r in records if int(r["fold"])!=int(fold)]
    va=[r for r in records if int(r["fold"])==int(fold)]
    if not tr or not va:
        raise RuntimeError("empty A6 fold split")

    feature_dim=int(np.asarray(tr[0]["features"]).shape[-1])
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g=torch.Generator().manual_seed(cfg.seed+int(fold))
    train_loader=DataLoader(
        FeatureBankDataset(tr),batch_size=cfg.batch_size,shuffle=True,
        collate_fn=collate_feature_bank,num_workers=0,generator=g,
    )
    val_loader=DataLoader(
        FeatureBankDataset(va),batch_size=cfg.batch_size,shuffle=False,
        collate_fn=collate_feature_bank,num_workers=0,
    )

    model=build_a6_model(feature_dim,cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best_state=None;best_score=float("-inf");bad=0;history=[]

    for epoch in range(cfg.epochs):
        model.train();losses=[]
        for batch in train_loader:
            feat=batch["features"].to(device)
            typ=batch["token_types"].to(device)
            mask=batch["token_mask"].to(device)
            y=batch["targets"].to(device)
            prior=torch_attention_prior(typ,variant)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                z=model(feat,typ,mask,attention_log_prior=prior)
                loss,parts=a6_objective(z,y,variant,cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt);scaler.update()
            losses.append(float(loss.detach().cpu()))

        ids,pred,target=_predict_loader(model,val_loader,variant,device)
        metrics=focus_metric(target,pred,variant)
        score=selection_score(metrics)
        history.append({
            "epoch":epoch,
            "loss":float(np.mean(losses)),
            "selection_score":score,
            **metrics,
        })
        if score>best_score+1e-5:
            best_score=score
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad=0
        else:
            bad+=1
            if bad>=cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("A6 fold failed to select a checkpoint")
    model.load_state_dict(best_state,strict=True)
    ids,pred,target=_predict_loader(model,val_loader,variant,device)
    final=focus_metric(target,pred,variant)
    return {
        "fold":int(fold),
        "variant":variant_name,
        "uids":ids,
        "pred":pred,
        "targets":target,
        "metrics":final,
        "selection_score":selection_score(final),
        "state_dict":best_state,
        "model_config":model.cfg.__dict__,
        "history":history,
    }


def assemble_oof(fold_results:list[dict],expected_uids:Iterable[str]):
    by_uid={}
    targets={}
    for r in fold_results:
        for uid,p,y in zip(r["uids"],r["pred"],r["targets"]):
            if uid in by_uid:
                raise RuntimeError(f"duplicate OOF prediction for {uid}")
            by_uid[uid]=p
            targets[uid]=y
    expected=list(map(str,expected_uids))
    if set(by_uid)!=set(expected):
        miss=sorted(set(expected)-set(by_uid))
        extra=sorted(set(by_uid)-set(expected))
        raise RuntimeError(f"OOF coverage mismatch missing={miss[:5]} extra={extra[:5]}")
    return (
        np.stack([by_uid[u] for u in expected]),
        np.stack([targets[u] for u in expected]),
    )
