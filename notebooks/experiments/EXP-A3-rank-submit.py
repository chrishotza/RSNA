from __future__ import annotations

import gc, hashlib, json, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from pydicom.pixel_data_handlers.util import apply_modality_lut
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from src.a3_feature_bank import SliceRecord, acquisition_type_id, build_series_window_manifest, sort_slice_records
from src.a3_training import VARIANTS, FeatureBankDataset, build_model, collate_feature_bank, compute_objective
from src.a3_uam_seq import LABELS, SEED, seed_everything, uid_fold

START=time.time()
WORK=Path("/kaggle/working")
WORK.mkdir(parents=True,exist_ok=True)
IMG=224
CROP_MM=130.0
ANCHORS_PER_SERIES=6
ENCODER_BATCH=96
TRAIN_BATCH=64
EPOCHS=5
PATIENCE=1
LR=3e-4
WEIGHT_DECAY=0.02
TIME_BUDGET=8.6*3600
RAD_SHA256="08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
VARIANT=VARIANTS["A3-3"]

def log(msg):
    print(f"[A3-SUB {time.time()-START:8.1f}s] {msg}",flush=True)

def deadline(tag):
    if time.time()-START>TIME_BUDGET:
        raise TimeoutError(f"A3 time budget exceeded during {tag}")

def sha256(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def find_comp_root():
    for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (p/"train.csv").is_file() and (p/"test.csv").is_file():
            return p
    raise FileNotFoundError("competition root not found")

def norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def canonicalize(df):
    by={norm(c):c for c in df.columns}
    uid=by.get(norm("StudyInstanceUID"))
    if uid is None: return None
    out=pd.DataFrame({"StudyInstanceUID":df[uid].astype(str)})
    for label in LABELS:
        c=by.get(norm(label))
        if c is None: return None
        out[label]=pd.to_numeric(df[c],errors="coerce")
    return out

def discover_pseudo():
    roots=[Path("/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels"),Path("/kaggle/input/rsna-knee-llm-labels")]
    cand=[]
    for root in roots:
        if not root.is_dir(): continue
        for p in root.rglob("*.csv"):
            try:
                f=canonicalize(pd.read_csv(p))
                if f is None: continue
                a=f[LABELS].to_numpy(float)
                finite=np.isfinite(a)
                vals=a[finite]
                if len(vals) and vals.min()>=0 and vals.max()<=1:
                    cand.append((int(finite.sum()),int(f.StudyInstanceUID.nunique()),p,f))
            except Exception:
                pass
    if not cand: raise RuntimeError("no pseudo-label CSV satisfies UID+12-target contract")
    cand.sort(key=lambda x:(x[0],x[1]),reverse=True)
    _,n,p,f=cand[0]
    if n<4000: raise RuntimeError(f"pseudo-label coverage suspicious: {n}")
    log(f"pseudo labels={p} studies={n}")
    return f.drop_duplicates("StudyInstanceUID",keep="last")

def prepare_targets(train,pseudo):
    gold=canonicalize(train)
    if gold is None: raise RuntimeError("train target contract changed")
    p=pseudo.set_index("StudyInstanceUID").reindex(gold.StudyInstanceUID).reset_index()
    weak=p[LABELS].to_numpy(np.float32)
    weak=np.where(np.isfinite(weak),weak,0.5).astype(np.float32)
    gv=gold[LABELS].to_numpy(np.float32)
    gm=np.isfinite(gv)
    target=np.where(gm,gv,weak).astype(np.float32)
    if not np.isfinite(target).all(): raise RuntimeError("non-finite training targets")
    return gold.StudyInstanceUID.tolist(),target,gm

class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self,x):
        return self.backbone(x).mean(dim=(2,3))

def load_encoder(device):
    candidates=[Path("/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt"),Path("/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt")]
    valid=[p for p in candidates if p.is_file() and sha256(p)==RAD_SHA256]
    if len(valid)!=1: raise RuntimeError(f"expected one pinned Rad encoder, got {valid}")
    model=RadEncoder()
    state=torch.load(valid[0],map_location="cpu",weights_only=True)
    if set(state)!=set(model.state_dict()): raise RuntimeError("Rad encoder key drift")
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False).to(device)
    log(f"Rad encoder verified {valid[0]}")
    return model

def header(path):
    h=pydicom.dcmread(path,stop_before_pixels=True)
    ipp=getattr(h,"ImagePositionPatient",None)
    iop=getattr(h,"ImageOrientationPatient",None)
    inst=getattr(h,"InstanceNumber",None)
    return SliceRecord(
        sop_instance_uid=Path(path).stem,
        image_position=tuple(float(x) for x in ipp) if ipp is not None and len(ipp)==3 else None,
        image_orientation=tuple(float(x) for x in iop) if iop is not None and len(iop)==6 else None,
        instance_number=int(inst) if inst is not None else None,
    )

def read_slice(path):
    d=pydicom.dcmread(path)
    a=apply_modality_lut(d.pixel_array,d).astype(np.float32)
    if str(getattr(d,"PhotometricInterpretation",""))=="MONOCHROME1": a=a.max()-a
    a=np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)
    lo,hi=np.percentile(a,[1,99])
    if hi<=lo: lo,hi=float(a.min()),float(a.max())
    a=np.zeros_like(a,np.float32) if hi<=lo else np.clip((a-lo)/(hi-lo),0,1)
    ps=getattr(d,"PixelSpacing",None)
    spacing=float(ps[0]) if ps is not None and len(ps)>=1 and float(ps[0])>0 else 0.5
    side=max(8,min(int(round(CROP_MM/spacing)),min(a.shape)))
    y0=(a.shape[0]-side)//2; x0=(a.shape[1]-side)//2
    return cv2.resize(a[y0:y0+side,x0:x0+side],(IMG,IMG),interpolation=cv2.INTER_AREA).astype(np.float32)

@torch.inference_mode()
def encode_triplets(model,triplets,device):
    out=[]
    for start in range(0,len(triplets),ENCODER_BATCH):
        deadline("Rad encoding")
        x=np.stack(triplets[start:start+ENCODER_BATCH]).astype(np.float32)
        t=torch.from_numpy(x).to(device,non_blocking=True).mul_(2).sub_(1)
        with torch.autocast("cuda",dtype=torch.float16,enabled=True):
            f=model(t)
        f=f.float().cpu().numpy()
        if not np.isfinite(f).all(): raise RuntimeError("non-finite feature")
        out.append(f.astype(np.float16))
    return np.concatenate(out,axis=0)

def build_records(comp,series_df,uids,split,targets=None,gold_mask=None):
    device=torch.device("cuda:0")
    enc=load_encoder(device)
    grouped={str(uid):g.to_dict("records") for uid,g in series_df.groupby("StudyInstanceUID",sort=False)}
    records=[]
    for i,uid in enumerate(map(str,uids)):
        deadline(f"{split} feature bank")
        feats=[]; types=[]
        for row in grouped.get(uid,[]):
            suid=str(row["SeriesInstanceUID"])
            sdir=comp/f"{split}_series"/uid/suid
            files=sorted(sdir.glob("*.dcm"))
            if not files: continue
            hs=[]; by={}
            for f in files:
                try:
                    h=header(f); hs.append(h); by[h.sop_instance_uid]=f
                except Exception:
                    continue
            if not hs: continue
            try: ordered=sort_slice_records(hs)
            except Exception: continue
            sops=[x.sop_instance_uid for x in ordered]
            manifest=build_series_window_manifest(sops,n_anchors=ANCHORS_PER_SERIES)
            trip=[]
            for m in manifest:
                try: trip.append(np.stack([read_slice(by[s]) for s in m["sop_triplet"]],axis=0))
                except Exception: pass
            if not trip: continue
            f=encode_triplets(enc,trip,device)
            feats.append(f)
            type_id=acquisition_type_id(row.get("Anatomical_Plane",""),row.get("Fluid_Sensitive",0),row.get("Fat_Suppression",0))
            types.extend([type_id]*len(f))
        if not feats: raise RuntimeError(f"{split} study {uid} produced zero valid tokens")
        rec={
            "uid":uid,
            "features":np.concatenate(feats,axis=0),
            "token_types":np.asarray(types,dtype=np.int16),
            "targets":np.zeros(12,np.float32) if targets is None else targets[i].astype(np.float32),
            "gold_mask":np.zeros(12,bool) if gold_mask is None else gold_mask[i].astype(bool),
            "fold":uid_fold(uid,5),
        }
        records.append(rec)
        if (i+1)%100==0: log(f"{split} bank {i+1}/{len(uids)}")
    del enc; gc.collect(); torch.cuda.empty_cache()
    return records

def loader(records,shuffle=False):
    return DataLoader(FeatureBankDataset(records),batch_size=TRAIN_BATCH,shuffle=shuffle,num_workers=0,collate_fn=collate_feature_bank,drop_last=False)

def predict(model,records,device):
    model.eval(); ids=[]; out=[]
    with torch.inference_mode():
        for b in loader(records,False):
            x=b["features"].to(device); t=b["token_types"].to(device); m=b["token_mask"].to(device)
            with torch.autocast("cuda",dtype=torch.float16,enabled=True):
                z=model(x,t,m)
            out.append(z.float().cpu().numpy()); ids.extend(b["uids"])
    return ids,np.concatenate(out)

def train_fold(train_records,test_records,fold,device):
    tr=[r for r in train_records if int(r["fold"])!=fold]
    va=[r for r in train_records if int(r["fold"])==fold]
    model=build_model(2048,VARIANT,projection_dim=384,metadata_dim=48,hidden_dim=256,gru_layers=2,metadata_vocab=32,dropout=0.15).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    best=None; best_loss=float("inf"); bad=0
    for epoch in range(EPOCHS):
        deadline(f"fold {fold} epoch {epoch}")
        model.train(); losses=[]
        for b in loader(tr,True):
            x=b["features"].to(device); t=b["token_types"].to(device); m=b["token_mask"].to(device)
            y=b["targets"].to(device); g=b["gold_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=True):
                z=model(x,t,m); loss,_=compute_objective(z,y,g,VARIANT)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        model.eval(); vals=[]
        with torch.inference_mode():
            for b in loader(va,False):
                x=b["features"].to(device); t=b["token_types"].to(device); m=b["token_mask"].to(device)
                y=b["targets"].to(device); g=b["gold_mask"].to(device)
                with torch.autocast("cuda",dtype=torch.float16,enabled=True):
                    z=model(x,t,m); loss,_=compute_objective(z,y,g,VARIANT)
                vals.append(float(loss.detach().cpu()))
        vl=float(np.mean(vals)); tl=float(np.mean(losses))
        log(f"fold={fold} epoch={epoch} train={tl:.5f} val={vl:.5f}")
        if vl<best_loss-1e-5:
            best_loss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if best is None: raise RuntimeError(f"fold {fold} produced no checkpoint")
    model.load_state_dict(best,strict=True)
    torch.save({"variant":"A3-3","fold":fold,"state_dict":best,"config":model.cfg.__dict__,"seed":SEED},WORK/f"a3_fold{fold}.pt")
    ids,pred=predict(model,test_records,device)
    del model,opt,scaler; gc.collect(); torch.cuda.empty_cache()
    return ids,pred

def rank01(x):
    return pd.DataFrame(x).rank(method="average",pct=True).to_numpy(np.float64)

def main():
    seed_everything(SEED)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    log(f"GPU={torch.cuda.get_device_name(0)}")
    comp=find_comp_root()
    train=pd.read_csv(comp/"train.csv",dtype={"StudyInstanceUID":str})
    test=pd.read_csv(comp/"test.csv",dtype={"StudyInstanceUID":str})
    train_series=pd.read_csv(comp/"train_series.csv",dtype={"StudyInstanceUID":str,"SeriesInstanceUID":str})
    test_series=pd.read_csv(comp/"test_series.csv",dtype={"StudyInstanceUID":str,"SeriesInstanceUID":str})
    pseudo=discover_pseudo()
    train_uids,targets,gold_mask=prepare_targets(train,pseudo)
    test_uids=test.StudyInstanceUID.astype(str).tolist()
    train_records=build_records(comp,train_series,train_uids,"train",targets,gold_mask)
    test_records=build_records(comp,test_series,test_uids,"test")
    device=torch.device("cuda:0")
    fold_preds=[]
    for fold in range(5):
        ids,p=train_fold(train_records,test_records,fold,device)
        if ids!=test_uids: raise RuntimeError("test UID order drift")
        fold_preds.append(p)
    logits=np.mean(np.stack(fold_preds,axis=0),axis=0)
    ranked=rank01(logits)
    sub=pd.DataFrame(ranked,columns=LABELS)
    sub.insert(0,"StudyInstanceUID",test_uids)
    if sub.shape!=(len(test_uids),13): raise RuntimeError("submission shape drift")
    if not np.isfinite(sub[LABELS].to_numpy()).all(): raise RuntimeError("non-finite submission")
    out=WORK/"submission.csv"
    sub.to_csv(out,index=False)
    receipt={
        "experiment":"EXP-A3",
        "model":"A3-RANK A3-3",
        "variant":"rank-first + target-specific acquisition router",
        "folds":5,
        "epochs_max":EPOCHS,
        "anchors_per_series":ANCHORS_PER_SERIES,
        "crop_mm":CROP_MM,
        "img":IMG,
        "rad_sha256":RAD_SHA256,
        "studies":len(test_uids),
        "submission_sha256":sha256(out),
        "elapsed_seconds":time.time()-START,
    }
    (WORK/"a3_submission_receipt.json").write_text(json.dumps(receipt,indent=2))
    log(f"COMPLETE submission={out} sha={receipt['submission_sha256']}")

if __name__=="__main__":
    main()
