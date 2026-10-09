#!/usr/bin/env python3
"""Target-swap validation gate for RSNA A5/0.97 experiments.

Compares parent and specialist/candidate predictions on the same gold studies.
Selection uses paired study bootstrap per target (default 400 resamples,
win fraction >= 0.90), reports Holm-adjusted p-values, and performs two-way
split-half cross-fitting so a target chosen on one half is scored on the other.

This is a guardrail against using the 58 gold studies as a private leaderboard.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS=[
    "ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
    "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"
]

def auc(y,s):
    if len(np.unique(y))<2: return np.nan
    return float(roc_auc_score(y,s))

def per_target(y,s):
    return np.asarray([auc(y[:,j],s[:,j]) for j in range(y.shape[1])],float)

def macro(y,s):
    return float(np.nanmean(per_target(y,s)))

def bootstrap_target(y,a,b,n=400,seed=20261009):
    rng=np.random.default_rng(seed)
    N=len(y); T=y.shape[1]
    d=np.full((n,T),np.nan,float)
    for k in range(n):
        idx=rng.integers(0,N,N)
        for j in range(T):
            aa=auc(y[idx,j],a[idx,j]); bb=auc(y[idx,j],b[idx,j])
            if np.isfinite(aa) and np.isfinite(bb): d[k,j]=bb-aa
    rows=[]
    for j in range(T):
        x=d[:,j]; x=x[np.isfinite(x)]
        rows.append({
            "target":LABELS[j],
            "delta":float(auc(y[:,j],b[:,j])-auc(y[:,j],a[:,j])),
            "boot_win_frac":float(np.mean(x>0)) if len(x) else np.nan,
            "ci_lo":float(np.quantile(x,.025)) if len(x) else np.nan,
            "ci_hi":float(np.quantile(x,.975)) if len(x) else np.nan,
            "p_one_sided":float(np.mean(x<=0)) if len(x) else np.nan,
        })
    return rows

def holm_adjust(pvals):
    p=np.asarray(pvals,float)
    out=np.full_like(p,np.nan)
    ok=np.isfinite(p)
    idx=np.where(ok)[0]
    order=idx[np.argsort(p[idx])]
    running=0.0
    m=len(order)
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*p[i])
        running=max(running,adj)
        out[i]=running
    return out

def apply_swaps(a,b,selected):
    out=a.copy()
    out[:,selected]=b[:,selected]
    return out

def split_half_crossfit(y,a,b,folds,select_rule,n_boot):
    if folds is None:
        # deterministic alternating split; only a fallback when canonical fold ids unavailable
        folds=np.arange(len(y))%2
    folds=np.asarray(folds).astype(int)%2
    gains=[]; details=[]
    for train_half,test_half in [(0,1),(1,0)]:
        tr=folds==train_half; te=folds==test_half
        rows=bootstrap_target(y[tr],a[tr],b[tr],n=n_boot,seed=20261009+train_half)
        sel=np.asarray([r["boot_win_frac"]>=select_rule for r in rows],bool)
        pa=macro(y[te],a[te])
        pb=macro(y[te],apply_swaps(a[te],b[te],sel))
        gains.append(pb-pa)
        details.append({
            "select_half":int(train_half),
            "score_half":int(test_half),
            "selected_targets":[LABELS[j] for j in np.where(sel)[0]],
            "parent_macro":pa,
            "swapped_macro":pb,
            "heldout_gain":pb-pa,
        })
    return details,float(np.mean(gains))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--parent",default="parent")
    ap.add_argument("--candidate",default="candidate")
    ap.add_argument("--fold-key",default="fold")
    ap.add_argument("--select-rule",type=float,default=.90)
    ap.add_argument("--bootstrap",type=int,default=400)
    ap.add_argument("--out",default="artifacts/validation/target_swap_gate.json")
    args=ap.parse_args()

    p=Path(args.npz); z=np.load(p,allow_pickle=False)
    y=np.asarray(z["truth"],float)
    a=np.asarray(z[args.parent],float)
    b=np.asarray(z[args.candidate],float)
    assert y.shape==a.shape==b.shape and y.shape[1]==12
    assert len(y)>=40 and np.isfinite(a).all() and np.isfinite(b).all()

    rows=bootstrap_target(y,a,b,args.bootstrap)
    holm=holm_adjust([r["p_one_sided"] for r in rows])
    selected=[]
    for j,r in enumerate(rows):
        r["p_holm"]=float(holm[j]) if np.isfinite(holm[j]) else np.nan
        r["survives_holm"]=bool(np.isfinite(holm[j]) and holm[j]<.05)
        r["selected"]=bool(r["boot_win_frac"]>=args.select_rule)
        selected.append(r["selected"])
    selected=np.asarray(selected,bool)

    swapped=apply_swaps(a,b,selected)
    folds=np.asarray(z[args.fold_key]) if args.fold_key in z.files else None
    split_details,split_gain=split_half_crossfit(
        y,a,b,folds,args.select_rule,args.bootstrap
    )

    parent_macro=macro(y,a)
    cand_macro=macro(y,b)
    swap_macro=macro(y,swapped)

    # Null calibration: permute candidate study rows independently per target.
    rng=np.random.default_rng(20261009)
    null_selected=[]; null_gain=[]
    for k in range(200):
        bp=b.copy()
        for j in range(12): bp[:,j]=bp[rng.permutation(len(bp)),j]
        rr=bootstrap_target(y,a,bp,n=100,seed=90000+k)
        ss=np.asarray([r["boot_win_frac"]>=args.select_rule for r in rr],bool)
        null_selected.append(int(ss.sum()))
        null_gain.append(macro(y,apply_swaps(a,bp,ss))-parent_macro)

    report={
      "schema":"rsna_target_swap_gate_v1",
      "source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
      "n_studies":int(len(y)),
      "parent":args.parent,
      "candidate":args.candidate,
      "select_rule":args.select_rule,
      "bootstrap":args.bootstrap,
      "parent_macro":parent_macro,
      "candidate_macro":cand_macro,
      "selected_swap_macro_same_data":swap_macro,
      "selected_swap_gain_same_data":swap_macro-parent_macro,
      "per_target":rows,
      "selected_targets":[LABELS[j] for j in np.where(selected)[0]],
      "split_half":split_details,
      "split_half_mean_gain":split_gain,
      "null_calibration":{
        "mean_targets_selected":float(np.mean(null_selected)),
        "p_any_target_selected":float(np.mean(np.asarray(null_selected)>0)),
        "p_null_gain_ge_observed_same_data":float(np.mean(np.asarray(null_gain)>=(swap_macro-parent_macro))),
      },
      "promotion_gates":{
        "at_least_one_target":bool(selected.any()),
        "same_data_macro_positive":bool(swap_macro>parent_macro),
        "split_half_positive":bool(split_gain>0),
        "null_calibration_ok":bool(np.mean(np.asarray(null_selected)>0)<.20),
      },
      "warning":"Target selection and same-data gain are optimistic. Promotion requires positive split-half transfer; Holm is reported as an additional high bar."
    }
    report["promote"]=bool(all(report["promotion_gates"].values()))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))
    if not report["promote"]: raise SystemExit(2)

if __name__=="__main__":
    main()
