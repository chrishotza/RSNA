from __future__ import annotations

import argparse
import json
from pathlib import Path


LOAD_BLOCK = r'''
def find_a3_training_checkpoint(fold):
    name=f"a3_a3-2_fold{fold}.pt"
    hits=[]
    for root in (Path("/kaggle/input"),):
        if root.is_dir():
            hits.extend(root.rglob(name))
    hits=sorted(set(p.resolve() for p in hits if p.is_file()))
    if len(hits)!=1:
        raise RuntimeError(f"expected exactly one {name}, got {hits}")
    return hits[0]

def infer_fold_from_checkpoint(test_records,fold,device):
    ckpt_path=find_a3_training_checkpoint(fold)
    ck=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    if ck.get("variant")!="A3-2" or int(ck.get("fold",-1))!=fold:
        raise RuntimeError(f"A3-2 checkpoint identity mismatch: {ckpt_path}")
    model=build_model(
        2048,VARIANTS["A3-2"],
        projection_dim=384,metadata_dim=48,hidden_dim=256,
        gru_layers=2,metadata_vocab=32,dropout=0.15,
    ).to(device)
    state=ck.get("state_dict")
    if not isinstance(state,dict):
        raise RuntimeError(f"A3-2 checkpoint missing state_dict: {ckpt_path}")
    model.load_state_dict(state,strict=True)
    ids,pred=predict(model,test_records,device)
    log(f"loaded A3-2 fold={fold} checkpoint={ckpt_path}")
    del model,ck,state
    gc.collect(); torch.cuda.empty_cache()
    return ids,pred
'''


MAIN_BLOCK = r'''
def main():
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    log(f"GPU={torch.cuda.get_device_name(0)}")
    comp=find_comp_root()
    test=pd.read_csv(comp/"test.csv",dtype={"StudyInstanceUID":str})
    test_series=pd.read_csv(
        comp/"test_series.csv",
        dtype={"StudyInstanceUID":str,"SeriesInstanceUID":str},
    )
    test_uids=test.StudyInstanceUID.astype(str).tolist()
    if not test_uids or len(test_uids)!=len(set(test_uids)):
        raise RuntimeError("invalid test UID identity")

    # Only hidden/test feature extraction happens in the submission kernel.
    test_records=build_records(comp,test_series,test_uids,"test")
    if len(test_records)!=len(test_uids):
        raise RuntimeError("A3 test feature-bank cardinality drift")

    device=torch.device("cuda:0")
    fold_preds=[]
    ckpt_paths=[]
    for fold in range(5):
        ids,p=infer_fold_from_checkpoint(test_records,fold,device)
        if ids!=test_uids:
            raise RuntimeError(f"A3-2 fold {fold} test UID order drift")
        if p.shape!=(len(test_uids),12) or not np.isfinite(p).all():
            raise RuntimeError(f"A3-2 fold {fold} invalid prediction matrix")
        fold_preds.append(p)
        ckpt_paths.append(str(find_a3_training_checkpoint(fold)))

    logits=np.mean(np.stack(fold_preds,axis=0),axis=0)
    ranked=rank01(logits)
    sub=pd.DataFrame(ranked,columns=LABELS)
    sub.insert(0,"StudyInstanceUID",test_uids)
    if sub.shape!=(len(test_uids),13):
        raise RuntimeError("submission shape drift")
    arr=sub[LABELS].to_numpy(float)
    if not np.isfinite(arr).all() or ((arr<0)|(arr>1)).any():
        raise RuntimeError("invalid A3 submission values")

    out=WORK/"submission.csv"
    sub.to_csv(out,index=False)
    receipt={
        "experiment":"EXP-A3",
        "model":"A3-RANK A3-2",
        "variant":"rank-first, no target-type router",
        "training_kernel":"chrishotza/rsna-exp-a3-a3-rank-training",
        "folds":5,
        "anchors_per_series":ANCHORS_PER_SERIES,
        "crop_mm":CROP_MM,
        "img":IMG,
        "rad_sha256":RAD_SHA256,
        "checkpoints":ckpt_paths,
        "studies":len(test_uids),
        "submission_sha256":sha256(out),
        "elapsed_seconds":time.time()-START,
    }
    (WORK/"a3_submission_receipt.json").write_text(json.dumps(receipt,indent=2))
    log(f"COMPLETE submission={out} sha={receipt['submission_sha256']}")

if __name__=="__main__":
    main()
'''


def build(source: Path, output_py: Path, output_ipynb: Path) -> None:
    text=source.read_text(encoding="utf-8")

    a=text.index("def train_fold(")
    b=text.index("def rank01(",a)
    text=text[:a]+LOAD_BLOCK+"\n\n"+text[b:]

    m=text.index("def main():", text.index("def rank01("))
    text=text[:m]+MAIN_BLOCK+"\n"
    compile(text,str(output_py),"exec")

    output_py.parent.mkdir(parents=True,exist_ok=True)
    output_py.write_text(text,encoding="utf-8")
    nb={
      "cells":[
        {"cell_type":"markdown","metadata":{},"source":[
          "# EXP-A3 — A3-2 inference-only submission\n",
          "Consumes frozen A3-2 fold checkpoints from the completed training kernel.\n"
        ]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)}
      ],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},
      "nbformat":4,"nbformat_minor":5
    }
    output_ipynb.write_text(json.dumps(nb),encoding="utf-8")
    print(f"built {output_py} and {output_ipynb}; bytes={len(text)}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output-py",type=Path,required=True)
    ap.add_argument("--output-ipynb",type=Path,required=True)
    args=ap.parse_args()
    build(args.source,args.output_py,args.output_ipynb)


if __name__=="__main__":
    main()
