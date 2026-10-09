from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def notebook_code(path: Path) -> str:
    if path.suffix == ".py":
        return path.read_text(encoding="utf-8", errors="replace")
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )


def find_code(root: Path) -> Path:
    files = sorted(list(root.glob("*.py")) + list(root.glob("*.ipynb")))
    if not files:
        raise FileNotFoundError(f"no notebook source in {root}")
    return files[0]


def extract_literal_assignment(code: str, name: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} literal assignment not found")


def strip_terminal_main(source: str) -> str:
    markers = [
        "\n_t0=time.time()\nmain()",
        "\n_t0 = time.time()\nmain()",
        "\nif __name__ == '__main__':",
        '\nif __name__ == "__main__":',
    ]
    cuts = [source.find(m) for m in markers if source.find(m) >= 0]
    if cuts:
        return source[: min(cuts)]
    return source


A4_TAIL = r'''
# ============================================================================
# EXP-A4 ORTHO -- exact-public 3-arm inference router
# ============================================================================
import gc as _a4_gc
import hashlib as _a4_hashlib
import json as _a4_json
import time as _a4_time
from pathlib import Path as _A4Path
import numpy as _a4_np
import pandas as _a4_pd
import torch as _a4_torch

_A4_START = _a4_time.time()
_A4_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

def _a4_log(msg):
    print(f"[A4-ORTHO {_a4_time.time()-_A4_START:7.1f}s] {msg}", flush=True)

def _a4_rank01(x):
    x=_a4_np.asarray(x,dtype=_a4_np.float64)
    if x.ndim!=2 or not _a4_np.isfinite(x).all():
        raise RuntimeError("invalid A4 arm predictions")
    return _a4_pd.DataFrame(x).rank(method="average",pct=True).to_numpy(_a4_np.float64)

def _a4_sigmoid(x):
    x=_a4_np.clip(_a4_np.asarray(x,dtype=_a4_np.float64),-40,40)
    return 1.0/(1.0+_a4_np.exp(-x))

def _a4_route(anchor,bio,dino):
    a,b,c=map(_a4_rank01,(anchor,bio,dino))
    specialist=_a4_np.median(_a4_np.stack([b,c],axis=0),axis=0)
    agreement=1.0-_a4_np.abs(b-c)
    disagreement=_a4_np.abs(specialist-a)
    gate=_a4_sigmoid((agreement-0.70)/0.08)*_a4_sigmoid((disagreement-0.10)/0.05)
    delta=_a4_np.clip(specialist-a,-0.20,0.20)
    raw=_a4_np.clip(a+gate*delta,0.0,1.0)
    final=_a4_rank01(raw)
    return final,{"agreement":agreement,"disagreement":disagreement,"gate":gate}

# Raptor source above has already executed and written the public anchor.
_anchor_path=_A4Path("/kaggle/working/submission.csv")
if not _anchor_path.is_file():
    raise RuntimeError("A4 anchor Raptor did not produce submission.csv")
_anchor=_a4_pd.read_csv(_anchor_path,dtype={"StudyInstanceUID":str})
if _anchor.columns.tolist()!=["StudyInstanceUID",*_A4_LABELS]:
    raise RuntimeError("A4 anchor schema drift")

# V36 definitions are executed in a private namespace; its original main() was stripped.
_A4_V36_NS={"__name__":"_a4_v36_defs"}
exec(compile(_A4_V36_DEFS,"<a4-v36-defs>","exec"),_A4_V36_NS)

build_index=_A4_V36_NS["build_file_and_series_index"]
DatasetCls=_A4_V36_NS["RSNAKnee5WindowDataset"]
BioCls=_A4_V36_NS["Track5A_BiomedCLIPCrossViewModel"]
DinoCls=_A4_V36_NS["RSNADinov2SlotModel"]
pool_windows=_A4_V36_NS["apply_target_pooling"]
DataLoader=_A4_V36_NS["DataLoader"]

dev=_a4_torch.device("cuda" if _a4_torch.cuda.is_available() else "cpu")
if dev.type!="cuda":
    raise RuntimeError("A4 requires CUDA")

file_map,series_map=build_index(root_dirs=["/kaggle/input"])
test_series_path=file_map.get("test_series.csv")
if not test_series_path:
    raise FileNotFoundError("test_series.csv not found")
series_df=_a4_pd.read_csv(test_series_path,dtype={"StudyInstanceUID":str,"SeriesInstanceUID":str})
test_df=series_df[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
dataset=DatasetCls(test_df,series_df,series_map,num_windows=5)
loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2)

p_bio=file_map.get("loop5_v3_track5a_fold0.pth")
p_dino=file_map.get("best_dinov3_finetuned_384.pth")
if not p_bio or not p_dino:
    raise FileNotFoundError(f"A4 specialist checkpoints missing: bio={p_bio} dino={p_dino}")

bio=BioCls(num_classes=12,d_model=512).to(dev)
bio.load_state_dict(_a4_torch.load(p_bio,map_location=dev,weights_only=False),strict=True)
bio.eval()
dino=DinoCls(model_name="vit_small_patch14_dinov2.lvd142m",num_classes=12,pretrained=False).to(dev)
st=_a4_torch.load(p_dino,map_location=dev,weights_only=False)
if any(k.startswith("backbone.") for k in st.keys()) or "view_embeds" in st:
    dino.load_state_dict(st,strict=True)
else:
    dino.backbone.load_state_dict(st,strict=True)
dino.eval()
_a4_log("specialists loaded strict=True")

uids=[]
bio_rows=[]
dino_rows=[]
for study_ids,s_5w,c_5w,a_5w in loader:
    uids.extend(map(str,study_ids))
    B,NW=s_5w.shape[:2]
    bw,dw=[],[]
    for w in range(NW):
        s=s_5w[:,w].to(dev,dtype=_a4_torch.float32)
        c=c_5w[:,w].to(dev,dtype=_a4_torch.float32)
        a=a_5w[:,w].to(dev,dtype=_a4_torch.float32)
        dummy=_a4_torch.zeros(B,8,384,384,device=dev)
        with _a4_torch.no_grad(), _a4_torch.amp.autocast("cuda"):
            pb=_a4_torch.sigmoid(bio(s,c,a,dummy,dummy,dummy)).float()
            pd=_a4_torch.sigmoid(dino(s,c,a)).float()
        bw.append(pb.cpu().numpy())
        dw.append(pd.cpu().numpy())
    bio_rows.append(pool_windows(bw))
    dino_rows.append(pool_windows(dw))

bio_pred=_a4_np.vstack(bio_rows)
dino_pred=_a4_np.vstack(dino_rows)
special=_a4_pd.DataFrame({"StudyInstanceUID":uids})
for j,t in enumerate(_A4_LABELS):
    special[t+"__bio"]=bio_pred[:,j]
    special[t+"__dino"]=dino_pred[:,j]

special=special.set_index("StudyInstanceUID").reindex(_anchor["StudyInstanceUID"]).reset_index()
if special.isna().any().any():
    raise RuntimeError("A4 specialist UID alignment drift")

bio_aligned=_a4_np.column_stack([special[t+"__bio"].to_numpy() for t in _A4_LABELS])
dino_aligned=_a4_np.column_stack([special[t+"__dino"].to_numpy() for t in _A4_LABELS])
anchor_values=_anchor[_A4_LABELS].to_numpy(_a4_np.float64)
final,audit=_a4_route(anchor_values,bio_aligned,dino_aligned)

out=_anchor[["StudyInstanceUID"]].copy()
out[_A4_LABELS]=final
arr=out[_A4_LABELS].to_numpy(_a4_np.float64)
if not _a4_np.isfinite(arr).all() or ((arr<0)|(arr>1)).any():
    raise RuntimeError("A4 invalid output range")
out.to_csv("/kaggle/working/submission.csv",index=False)

receipt={
    "experiment":"EXP-A4",
    "codename":"A4-ORTHO",
    "inference_only":True,
    "arms":{
        "anchor":"dreaddevelopment raptor_ft_coatnet_v4_full.pt",
        "specialist_b":"loop5_v3_track5a_fold0.pth",
        "specialist_c":"best_dinov3_finetuned_384.pth",
    },
    "router":{
        "agreement_threshold":0.70,
        "disagreement_threshold":0.10,
        "agreement_temperature":0.08,
        "disagreement_temperature":0.05,
        "displacement_cap":0.20,
    },
    "studies":len(out),
    "gate_mean":float(audit["gate"].mean()),
    "gate_max":float(audit["gate"].max()),
    "elapsed_seconds":_a4_time.time()-_A4_START,
}
_A4Path("/kaggle/working/a4_ortho_receipt.json").write_text(_a4_json.dumps(receipt,indent=2))
_a4_log(f"COMPLETE submission.csv rows={len(out)} gate_mean={receipt['gate_mean']:.4f}")
'''


def build(raptor_root: Path, v36_root: Path, output: Path) -> None:
    raptor = notebook_code(find_code(raptor_root))
    v36_outer = notebook_code(find_code(v36_root))
    v36 = strip_terminal_main(extract_literal_assignment(v36_outer, "_V36_SRC"))

    # Public Raptor notebook may write relative submission.csv. Force working-path identity.
    raptor = raptor.replace(
        'out.to_csv("submission.csv", index=False)',
        'out.to_csv("/kaggle/working/submission.csv", index=False)',
    )
    raptor = raptor.replace(
        "out.to_csv('submission.csv', index=False)",
        "out.to_csv('/kaggle/working/submission.csv', index=False)",
    )

    combined = (
        raptor
        + "\n\n# === A4 embedded exact V36 definitions ===\n"
        + "_A4_V36_DEFS = " + repr(v36) + "\n"
        + A4_TAIL
    )
    compile(combined, "EXP-A4-ortho-submit.py", "exec")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EXP-A4 — ORTHO\n",
                    "Inference-only public three-arm disagreement router.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": combined.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(nb), encoding="utf-8")
    print(f"built {output} code_bytes={len(combined)}")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--raptor-root",type=Path,required=True)
    ap.add_argument("--v36-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    build(args.raptor_root,args.v36_root,args.output)


if __name__=="__main__":
    main()
