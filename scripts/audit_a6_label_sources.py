#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def norm(s): return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def canonical(df):
    by={norm(c):c for c in df.columns}
    uid=by.get(norm("StudyInstanceUID"))
    if uid is None: return None
    out=pd.DataFrame({"StudyInstanceUID":df[uid].astype(str)})
    for lab in LABELS:
        exact=by.get(norm(lab))
        conf=None
        for c in df.columns:
            nc=norm(c)
            if norm(lab) in nc and "confidence" in nc:
                conf=c;break
        c=exact or conf
        if c is None: return None
        out[lab]=pd.to_numeric(df[c],errors="coerce")
    return out

def find_best(root):
    candidates=[]
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv",".parquet"}: continue
        try:
            raw=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
            f=canonical(raw)
            if f is None: continue
            finite=np.isfinite(f[LABELS].to_numpy(float))
            vals=f[LABELS].to_numpy(float)[finite]
            if not len(vals): continue
            if np.nanmin(vals)<0 or np.nanmax(vals)>1: continue
            candidates.append((int(finite.sum()),int(f.StudyInstanceUID.nunique()),p,f))
        except Exception: pass
    if not candidates: raise RuntimeError(f"no canonical label table in {root}")
    candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
    return candidates[0]

def score(gold,frame):
    f=frame.drop_duplicates("StudyInstanceUID",keep="last").set_index("StudyInstanceUID").reindex(gold.StudyInstanceUID)
    pred=f[LABELS].to_numpy(float)
    truth=gold[LABELS].to_numpy(float)
    out={}
    for j,lab in enumerate(LABELS):
        m=np.isfinite(truth[:,j]) & np.isfinite(pred[:,j])
        if m.sum()<2 or len(np.unique(truth[m,j]))<2:
            out[lab]=None
        else:
            out[lab]=float(roc_auc_score(truth[m,j],pred[m,j]))
    vals=[x for x in out.values() if x is not None]
    return out,float(np.mean(vals)),int(np.isfinite(pred).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--competition",required=True)
    ap.add_argument("--source",action="append",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    train=pd.read_csv(Path(a.competition)/"train.csv",dtype={"StudyInstanceUID":str})
    gold=train.dropna(subset=LABELS,how="all").copy()
    if len(gold)!=58: raise RuntimeError(f"gold count drift {len(gold)}")
    gold_uids=set(gold.StudyInstanceUID.astype(str))

    report={"gold_count":58,"sources":{}}
    per_target_best={lab:None for lab in LABELS}
    for spec in a.source:
        name,root=spec.split("=",1)
        coverage,uids,p,f=find_best(root)
        per,macro,finite=score(gold,f)
        gold_rows=int(f.StudyInstanceUID.astype(str).isin(gold_uids).sum())
        non_gold=f[~f.StudyInstanceUID.astype(str).isin(gold_uids)]
        report["sources"][name]={
            "file":str(p),"coverage_cells":coverage,"uids":uids,
            "gold_rows_present":gold_rows,
            "non_gold_rows_after_hard_exclusion":int(non_gold.StudyInstanceUID.nunique()),
            "gold_macro_auc":macro,"per_target_auc":per,
        }
        for lab in LABELS:
            v=per.get(lab)
            if v is None: continue
            cur=per_target_best[lab]
            if cur is None or v>cur["auc"]:
                per_target_best[lab]={"source":name,"auc":v}
    report["per_target_best_observed"]=per_target_best
    report["warning"]="Gold scores are descriptive provenance evidence. A6 training always hard-excludes the 58 gold UIDs before loading weak targets."
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main()
