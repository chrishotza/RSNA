#!/usr/bin/env python3
"""Build a private D4-lite + independent ConvNeXt hybrid notebook.

The D4 notebook is copied byte-for-structure and the ConvNeXt reader cells are
appended after D4 has published its verified submission.csv. No D4 model cell is
modified. The reader produces a global-30 blend and a pre-registered target-aware
variant based only on clean ConvNeXt gold58 evidence.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re
from pathlib import Path

TARGET_WEIGHTS=[.45,.40,.45,.25,.45,.20,.05,.15,.05,.45,.45,.20]

def union(a,b):
    out=[]
    for x in (a or [])+(b or []):
        if x not in out: out.append(x)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--d4",required=True)
    ap.add_argument("--cnxt",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--kernel-id",default="chrishotza/rsna-a5-d4-cnxt-hybrid")
    a=ap.parse_args()

    d4dir=Path(a.d4); cdir=Path(a.cnxt); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    d4p=next(d4dir.glob("*.ipynb")); cp=next(cdir.glob("*.ipynb"))
    d4=json.loads(d4p.read_text()); cn=json.loads(cp.read_text())
    dm=json.loads((d4dir/"kernel-metadata.json").read_text())
    cm=json.loads((cdir/"kernel-metadata.json").read_text())

    if len(d4["cells"]) < 30 or len(cn["cells"]) < 40:
        raise SystemExit("upstream notebook cell contract drift")

    reader=[copy.deepcopy(cn["cells"][i]) for i in range(35,40)]
    joined="\n".join("".join(c.get("source",[])) for c in reader)
    required=["own_src/preprocess.py","own_src/knee.py","own_src/infer.py",
              "cnxt_v0_fold0.pt","submission.csv"]
    miss=[x for x in required if x not in joined]
    if miss: raise SystemExit(f"reader cell contract drift: {miss}")

    prep='''# A5 hybrid adapter: verified D4-lite parent already exists as submission.csv.
RUN_STACK = True
print("[A5-HYBRID] attaching clean independent ConvNeXt reader to D4-lite parent")
'''
    reader.insert(0,{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                     "source":prep.splitlines(keepends=True)})

    final="".join(reader[-1]["source"])
    # Detect actual weight variable instead of assuming its spelling.
    pats=[r"OWN_W\s*=\s*[0-9.]+",r"_o_w\s*=\s*[0-9.]+",r"OWN_WEIGHT\s*=\s*[0-9.]+"]
    found=[]
    for pat in pats:
        found.extend((pat,m.group(0)) for m in re.finditer(pat,final))
    if len(found)!=1:
        raise SystemExit(f"expected one reader weight assignment, got {found}")
    pat,_=found[0]
    var=re.match(r"[A-Za-z_][A-Za-z0-9_]*",found[0][1]).group(0)
    final=re.sub(pat,f"{var} = 0.30",final,count=1)

    anchor='_o_pub, _o_bak = "/kaggle/working/submission.csv", "/kaggle/working/_public_stack.csv"'
    if anchor not in final:
        raise SystemExit("ConvNeXt parent backup anchor drift")
    final=final.replace(anchor,anchor+'\nimport shutil as _a5_shutil\n'
                        '_a5_shutil.copy(_o_pub, "/kaggle/working/d4_parent_submission.csv")',1)
    reader[-1]["source"]=final.splitlines(keepends=True)

    target=r'''
# Pre-registered target-aware blend from CLEAN ConvNeXt holdout evidence only.
import pandas as _a5pd, numpy as _a5np
_A5_LABS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
_A5_W=_a5np.asarray([.45,.40,.45,.25,.45,.20,.05,.15,.05,.45,.45,.20],float)
_p=_a5pd.read_csv("/kaggle/working/d4_parent_submission.csv",dtype={"StudyInstanceUID":str})
_g=_a5pd.read_csv("/kaggle/working/submission.csv",dtype={"StudyInstanceUID":str})
_pr=_p[_A5_LABS].rank(method="average",pct=True).to_numpy(float)
_gr=_g[_A5_LABS].to_numpy(float)
# Exact global blend is g=.70*parent_rank+.30*reader_rank.
_rr=(_gr-.70*_pr)/.30
if not _a5np.isfinite(_rr).all():
    raise RuntimeError("reader rank recovery non-finite")
_rr=_a5np.clip(_rr,0,1)
_t=(1-_A5_W)*_pr+_A5_W*_rr
_o=_p.copy(); _o[_A5_LABS]=_t
_o.to_csv("/kaggle/working/submission_targetaware.csv",index=False)
_g.to_csv("/kaggle/working/submission_global30.csv",index=False)
print("[A5-HYBRID] emitted global30 + targetaware",dict(zip(_A5_LABS,_A5_W.tolist())))
'''
    reader.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                   "source":target.splitlines(keepends=True)})

    hybrid=copy.deepcopy(d4)
    hybrid["cells"].extend(reader)

    meta=copy.deepcopy(dm)
    meta["id"]=a.kernel_id
    meta["title"]="RSNA A5 D4 ConvNeXt Hybrid"
    meta["code_file"]="rsna-a5-d4-cnxt-hybrid.ipynb"
    meta["is_private"]=True
    meta["enable_gpu"]=True
    meta["enable_internet"]=False
    for k in ("dataset_sources","kernel_sources","competition_sources","model_sources"):
        meta[k]=union(dm.get(k,[]),cm.get(k,[]))

    nbout=out/meta["code_file"]
    nbout.write_text(json.dumps(hybrid))
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))

    audit={
      "d4_sha256":hashlib.sha256(d4p.read_bytes()).hexdigest(),
      "cnxt_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),
      "d4_cells":len(d4["cells"]),"hybrid_cells":len(hybrid["cells"]),
      "reader_cells_appended":len(reader),
      "weight_variable":var,"global_weight":.30,
      "target_weights":TARGET_WEIGHTS,
      "datasets_added":[x for x in meta.get("dataset_sources",[]) if x not in dm.get("dataset_sources",[])],
      "kernel_sources_added":[x for x in meta.get("kernel_sources",[]) if x not in dm.get("kernel_sources",[])],
      "model_sources_added":[x for x in meta.get("model_sources",[]) if x not in dm.get("model_sources",[])],
    }
    print(json.dumps(audit,indent=2))
    (out/"build_audit.json").write_text(json.dumps(audit,indent=2)+"\n")

if __name__=="__main__":
    main()
