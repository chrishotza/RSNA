#!/usr/bin/env python3
"""Evaluate Apex original head vs A6 attention-prior adapter on a clean holdout.

Input NPZ:
  features: [N,K,F] pre-head Apex window features
  centers:  [N,K] center indices in the 96-slice canonical volume
  truth:    [N,12]
  clsW:     [12,F]
  clsb:     [12]
  norm_weight/norm_bias: LayerNorm parameters
  att_*: serialized two-layer Apex attention MLP parameters

This tool intentionally evaluates only the head intervention. It never fits a
parameter on gold labels.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS=[
 "ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
 "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"
]
PRIOR_TARGETS={"PF OA","Synovitis","Lateral OA","Lateral Meniscus"}

def aucs(y,p):
    out=[]
    for j in range(12):
        if len(np.unique(y[:,j]))<2: out.append(np.nan)
        else: out.append(float(roc_auc_score(y[:,j],p[:,j])))
    return np.asarray(out,float)

def macro(y,p): return float(np.nanmean(aucs(y,p)))

def softmax(x,axis):
    x=x-np.max(x,axis=axis,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=axis,keepdims=True)

def layernorm(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=((x-m)**2).mean(-1,keepdims=True)
    return (x-m)/np.sqrt(v+eps)*w+b

def tanh(x): return np.tanh(x)

def build_prior(centers,strength):
    from src.a6_apex_attention import build_target_window_log_prior
    return np.stack([
        build_target_window_log_prior(list(map(int,row)),D=96,strength=strength)
        for row in centers
    ])

def head(z,prior=None):
    feat=np.asarray(z["features"],np.float64)
    nw=np.asarray(z["norm_weight"],np.float64)
    nb=np.asarray(z["norm_bias"],np.float64)
    h=layernorm(feat,nw,nb)
    w0=np.asarray(z["att_0_weight"],np.float64)
    b0=np.asarray(z["att_0_bias"],np.float64)
    w3=np.asarray(z["att_3_weight"],np.float64)
    b3=np.asarray(z["att_3_bias"],np.float64)
    hidden=tanh(np.einsum("bkf,hf->bkh",h,w0)+b0)
    logits=np.einsum("bkh,qh->bkq",hidden,w3)+b3
    if prior is not None:
        logits=logits+np.transpose(prior,(0,2,1))
    a=softmax(logits,axis=1)
    pooled=np.einsum("bkq,bkf->bqf",a,h)
    clsW=np.asarray(z["clsW"],np.float64)
    clsb=np.asarray(z["clsb"],np.float64)
    out=(pooled*clsW[None,:,:]).sum(-1)+clsb[None,:]
    return 1/(1+np.exp(-out)),a

def bootstrap(y,a,b,n=5000,seed=20261009):
    rng=np.random.default_rng(seed)
    d=[]
    N=len(y)
    for _ in range(n):
        idx=rng.integers(0,N,N)
        aa=[];bb=[]
        for j in range(12):
            yy=y[idx,j]
            if len(np.unique(yy))<2: continue
            aa.append(roc_auc_score(yy,a[idx,j]))
            bb.append(roc_auc_score(yy,b[idx,j]))
        if aa: d.append(float(np.mean(bb)-np.mean(aa)))
    x=np.asarray(d,float)
    return {
      "mean":float(x.mean()),"p025":float(np.quantile(x,.025)),
      "p50":float(np.quantile(x,.5)),"p975":float(np.quantile(x,.975)),
      "p_gt_0":float(np.mean(x>0))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--strength",type=float,default=1.0)
    ap.add_argument("--out",default="artifacts/validation/a6_apex_attention_eval.json")
    a=ap.parse_args()

    z=np.load(a.npz,allow_pickle=False)
    y=np.asarray(z["truth"],float)
    centers=np.asarray(z["centers"],int)
    assert y.ndim==2 and y.shape[1]==12
    assert centers.shape[:1]==y.shape[:1]

    p0,att0=head(z,None)
    zero,_=head(z,build_prior(centers,0.0))
    parity=float(np.max(np.abs(p0-zero)))
    if parity>1e-12:
        raise RuntimeError(f"zero-prior parity failed: {parity}")

    prior=build_prior(centers,a.strength)
    p1,att1=head(z,prior)
    a0=aucs(y,p0); a1=aucs(y,p1); delta=a1-a0

    focus=[LABELS.index(x) for x in PRIOR_TARGETS]
    report={
      "schema":"a6_apex_attention_eval_v1",
      "n":int(len(y)),
      "strength":a.strength,
      "zero_prior_max_abs_diff":parity,
      "parent_macro":float(np.nanmean(a0)),
      "adapter_macro":float(np.nanmean(a1)),
      "macro_delta":float(np.nanmean(a1)-np.nanmean(a0)),
      "focus_mean_delta":float(np.nanmean(delta[focus])),
      "per_target":{
        lab:{"parent_auc":float(a0[j]),"adapter_auc":float(a1[j]),"delta":float(delta[j])}
        for j,lab in enumerate(LABELS)
      },
      "paired_bootstrap":bootstrap(y,p0,p1),
      "mean_abs_attention_change":float(np.mean(np.abs(att1-att0))),
      "promotion_gates":{
        "zero_prior_exact": parity<=1e-12,
        "focus_positive": float(np.nanmean(delta[focus]))>0,
        "macro_nonnegative": float(np.nanmean(delta))>=0,
      }
    }
    report["promote"]=bool(all(report["promotion_gates"].values()))
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["promote"]: raise SystemExit(2)

if __name__=="__main__": main()
