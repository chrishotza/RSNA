#!/usr/bin/env python3
"""Strict internal promotion gate for A5-family candidates.

This module is deliberately stdlib + numpy/pandas/sklearn only and performs no
leaderboard access. It evaluates paired predictions on the same gold studies,
reports per-target and macro AUC deltas, study-level bootstrap uncertainty,
target jackknife sensitivity, and guards against post-hoc target cherry-picking.

Expected NPZ keys:
  uids, truth, original, <candidate>, [uncertainty]
Shapes are [n_study, 12] except uids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = [
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion","Synovitis",
    "Baker's","Contusion","Fracture",
]

def rank01(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    out=np.empty_like(x)
    for j in range(x.shape[1]):
        order=np.argsort(x[:,j],kind="mergesort")
        ranks=np.empty(len(x),float)
        # average ties, 0..1
        vals=x[order,j]
        i=0
        while i<len(vals):
            k=i+1
            while k<len(vals) and vals[k]==vals[i]:
                k+=1
            r=(i+k-1)/2
            ranks[order[i:k]]=r
            i=k
        out[:,j]=r/max(len(x)-1,1)
    return out

def per_target_auc(y,p):
    vals=[]
    for j in range(y.shape[1]):
        if len(np.unique(y[:,j]))<2:
            vals.append(np.nan)
        else:
            vals.append(float(roc_auc_score(y[:,j],p[:,j])))
    return np.asarray(vals,float)

def macro_auc(y,p):
    return float(np.nanmean(per_target_auc(y,p)))

def paired_bootstrap(y,a,b,n=10000,seed=20261009):
    rng=np.random.default_rng(seed)
    diffs=[]
    N=len(y)
    for _ in range(n):
        idx=rng.integers(0,N,N)
        aa=[];bb=[]
        for j in range(y.shape[1]):
            yy=y[idx,j]
            if len(np.unique(yy))<2: continue
            aa.append(roc_auc_score(yy,a[idx,j]))
            bb.append(roc_auc_score(yy,b[idx,j]))
        if aa:
            diffs.append(float(np.mean(bb)-np.mean(aa)))
    d=np.asarray(diffs,float)
    return {
        "n_valid":int(len(d)),
        "mean":float(d.mean()),
        "median":float(np.median(d)),
        "p025":float(np.quantile(d,.025)),
        "p05":float(np.quantile(d,.05)),
        "p95":float(np.quantile(d,.95)),
        "p975":float(np.quantile(d,.975)),
        "p_gt_0":float(np.mean(d>0)),
    }

def study_jackknife(y,a,b):
    full=macro_auc(y,b)-macro_auc(y,a)
    vals=[]
    for i in range(len(y)):
        keep=np.ones(len(y),bool);keep[i]=False
        vals.append(macro_auc(y[keep],b[keep])-macro_auc(y[keep],a[keep]))
    v=np.asarray(vals)
    return {
        "full_delta":float(full),
        "min_leave_one_out":float(v.min()),
        "max_leave_one_out":float(v.max()),
        "mean_leave_one_out":float(v.mean()),
        "sign_flips":int(np.sum(np.sign(v)!=np.sign(full))),
    }

def target_leave_one_out(y,a,b):
    da=per_target_auc(y,a)
    db=per_target_auc(y,b)
    delta=db-da
    valid=np.isfinite(delta)
    macro=float(np.mean(delta[valid]))
    vals={}
    idx=np.where(valid)[0]
    for j in idx:
        rest=idx[idx!=j]
        vals[LABELS[j]]=float(np.mean(delta[rest])) if len(rest) else np.nan
    return {
        "macro_delta":macro,
        "min_without_one_target":float(np.nanmin(list(vals.values()))),
        "max_without_one_target":float(np.nanmax(list(vals.values()))),
        "without_target":vals,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--candidate",default="eam12")
    ap.add_argument("--parent",default="original")
    ap.add_argument("--out",default="artifacts/validation/a5_promotion_report.json")
    ap.add_argument("--bootstrap",type=int,default=10000)
    args=ap.parse_args()

    path=Path(args.npz)
    z=np.load(path,allow_pickle=False)
    y=np.asarray(z["truth"],float)
    a=np.asarray(z[args.parent],float)
    b=np.asarray(z[args.candidate],float)
    uids=np.asarray(z["uids"])

    assert y.shape==a.shape==b.shape
    assert y.shape[1]==12 and len(y)>=40
    assert len(set(map(str,uids)))==len(uids)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert ((a>=0)&(a<=1)).all() and ((b>=0)&(b<=1)).all()

    # AUC depends only on ordering. Evaluate raw and rank-space to catch tie/scale effects.
    auc_a=per_target_auc(y,a); auc_b=per_target_auc(y,b)
    delta=auc_b-auc_a
    raw_parent=float(np.nanmean(auc_a))
    raw_cand=float(np.nanmean(auc_b))
    rank_parent=macro_auc(y,rank01(a))
    rank_cand=macro_auc(y,rank01(b))

    boot=paired_bootstrap(y,a,b,args.bootstrap)
    sj=study_jackknife(y,a,b)
    tj=target_leave_one_out(y,a,b)

    improved=np.isfinite(delta)&(delta>0)
    degraded=np.isfinite(delta)&(delta<0)

    # Strict pre-registered gate for an inference-only replacement.
    # Not "proof" of hidden LB improvement: it is an internal promotion rule.
    gates={
        "macro_positive": raw_cand > raw_parent,
        "bootstrap_directional": boot["p_gt_0"] >= .80,
        "not_single_target_effect": tj["min_without_one_target"] > -0.005,
        "broad_target_behavior": int(improved.sum()) >= int(degraded.sum()),
        "study_stability": sj["sign_flips"] <= max(2,int(.05*len(y))),
        "rank_consistency": (rank_cand-rank_parent) >= -0.002,
    }

    report={
        "schema":"a5_internal_promotion_v1",
        "source_npz":str(path),
        "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_studies":int(len(y)),
        "parent":args.parent,
        "candidate":args.candidate,
        "parent_macro_auc":raw_parent,
        "candidate_macro_auc":raw_cand,
        "macro_delta":raw_cand-raw_parent,
        "rank_macro_delta":rank_cand-rank_parent,
        "per_target":{
            lab:{
                "parent_auc":float(auc_a[j]),
                "candidate_auc":float(auc_b[j]),
                "delta":float(delta[j]),
            } for j,lab in enumerate(LABELS)
        },
        "targets_improved":int(improved.sum()),
        "targets_degraded":int(degraded.sum()),
        "paired_bootstrap":boot,
        "study_leave_one_out":sj,
        "target_leave_one_out":tj,
        "promotion_gates":gates,
        "promote":bool(all(gates.values())),
        "warning":"Gold58 is a guardrail, not a leaderboard. Do not tune target weights on this report and then quote this same report as unbiased validation."
    }

    if "uncertainty" in z.files:
        unc=np.asarray(z["uncertainty"],float)
        if unc.shape==y.shape and np.isfinite(unc).all():
            report["uncertainty_summary"]={
                lab:{
                    "mean":float(unc[:,j].mean()),
                    "p90":float(np.quantile(unc[:,j],.9)),
                    "max":float(unc[:,j].max()),
                } for j,lab in enumerate(LABELS)
            }

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))
    if not report["promote"]:
        raise SystemExit(2)

if __name__=="__main__":
    main()
