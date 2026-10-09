#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

REQ=["submission.csv","submission_global30.csv","submission_targetaware.csv","d4_parent_submission.csv"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.root)
    miss=[x for x in REQ if not (root/x).is_file()]
    if miss: raise SystemExit("missing outputs: "+repr(miss))
    report={}
    for fn in REQ:
        p=root/fn
        d=pd.read_csv(p)
        v=d.iloc[:,1:].to_numpy(float)
        if d.shape[1]!=13 or not np.isfinite(v).all():
            raise SystemExit("invalid output "+fn)
        report[fn]={
            "shape":list(d.shape),
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
            "min":float(v.min()),"max":float(v.max())
        }
    g=pd.read_csv(root/"submission_global30.csv").iloc[:,1:].to_numpy(float)
    t=pd.read_csv(root/"submission_targetaware.csv").iloc[:,1:].to_numpy(float)
    diff=np.abs(g-t)
    report["targetaware_vs_global30"]={
        "changed_cells":int((diff>1e-12).sum()),
        "max_abs_diff":float(diff.max())
    }
    Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
