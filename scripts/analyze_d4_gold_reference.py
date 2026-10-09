#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def per_auc(y,p):
    out=[]
    for j in range(12):
        out.append(float(roc_auc_score(y[:,j],p[:,j])) if len(np.unique(y[:,j]))==2 else np.nan)
    return np.asarray(out,float)

def macro(y,p): return float(np.nanmean(per_auc(y,p)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out",required=True)
    ap.add_argument("--bootstrap",type=int,default=10000)
    a=ap.parse_args()
    z=np.load(a.npz,allow_pickle=False)
    y=np.asarray(z["truth"],float)
    p=np.asarray(z["probability_mean"],float)
    u=np.asarray(z["study_uids"]).astype(str)
    auc=per_auc(y,p)
    rng=np.random.default_rng(20261009)
    boots=[]
    for _ in range(a.bootstrap):
        idx=rng.integers(0,len(y),len(y))
        vals=[]
        for j in range(12):
            yy=y[idx,j]
            if len(np.unique(yy))<2: continue
            vals.append(roc_auc_score(yy,p[idx,j]))
        if vals: boots.append(float(np.mean(vals)))
    b=np.asarray(boots)
    loo=[]
    for i in range(len(y)):
        keep=np.ones(len(y),bool);keep[i]=False
        loo.append(macro(y[keep],p[keep]))
    positives={lab:int(y[:,j].sum()) for j,lab in enumerate(LABELS)}
    report={
      "schema":"d4_gold58_reference_analysis_v1",
      "n":int(len(y)),
      "macro_auc":float(np.nanmean(auc)),
      "per_target_auc":dict(zip(LABELS,map(float,auc))),
      "positives":positives,
      "bootstrap_macro":{
        "mean":float(b.mean()),"p025":float(np.quantile(b,.025)),
        "p50":float(np.quantile(b,.5)),"p975":float(np.quantile(b,.975))
      },
      "leave_one_study_out":{
        "min":float(np.min(loo)),"max":float(np.max(loo)),
        "mean":float(np.mean(loo))
      },
      "priority_targets_by_low_auc":[
        {"target":LABELS[j],"auc":float(auc[j]),"positives":positives[LABELS[j]]}
        for j in np.argsort(auc)
      ],
      "warning":"This public D4 gold reference is a development surface, not independent model-selection evidence."
    }
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
