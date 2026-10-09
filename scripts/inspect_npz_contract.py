#!/usr/bin/env python3
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
import numpy as np

src=Path(sys.argv[1]); out=Path(sys.argv[2])
with np.load(src,allow_pickle=False) as z:
    report={
      "sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
      "keys":list(z.files),
      "arrays":{}
    }
    for k in z.files:
        a=np.asarray(z[k])
        item={"shape":list(a.shape),"dtype":str(a.dtype)}
        flat=a.reshape(-1)
        if a.dtype.kind in "iufb":
            item["finite"]=bool(np.isfinite(a).all())
            item["min"]=float(np.nanmin(a)) if a.size else None
            item["max"]=float(np.nanmax(a)) if a.size else None
            item["preview"]=flat[:20].tolist()
        else:
            item["preview"]=[str(x) for x in flat[:20]]
        report["arrays"][k]=item
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
