"""Make a self-contained Kaggle notebook with the Raptor source and A6 runner."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
raptor=(root/"external/src/rsna_knee/raptor.py").read_text()
exporter=(root/"scripts/a6_export_raptor_visual_teacher.py").read_text()
# Raptor has relative package imports; install tiny source package in notebook FS.
cells=[
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
"import os, pathlib, subprocess, sys\n",
"subprocess.run([sys.executable,'-m','pip','install','-q','timm','pydicom','opencv-python-headless'],check=True)\n",
"pathlib.Path('/kaggle/working/rsna_knee').mkdir(exist_ok=True)\n",
"pathlib.Path('/kaggle/working/rsna_knee/__init__.py').write_text('')\n",
"src="+repr(raptor)+"\n",
"pathlib.Path('/kaggle/working/rsna_knee/raptor.py').write_text(src)\n",
"pathlib.Path('/kaggle/working/a6_export.py').write_text("+repr(exporter)+")\n",
"sys.path.insert(0,'/kaggle/working')\n"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
"from pathlib import Path\n",
"import subprocess,sys\n",
"root=Path('/kaggle/input')\n",
"def locate(name):\n",
"  matches=list(root.rglob(name))\n",
"  if len(matches)!=1: raise RuntimeError(f'Expected one {name}, found {len(matches)}')\n",
"  return matches[0]\n",
"train=locate('train.csv').parent\n",
"weights=locate('raptor_ft_coatnet_v5_full_swa.pt')\n",
"cmd=[sys.executable,'/kaggle/working/a6_export.py','--competition-root',str(train),'--weights',str(weights),'--source-root','/kaggle/working','--output','/kaggle/working/a6_result','--limit','0']\n",
"print('Running A6 full 4349-study teacher extraction; no competition submission')\n",
"subprocess.run(cmd,check=True)\n"]}
]
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
out=root/"kaggle/a6_teacher_export";out.mkdir(parents=True,exist_ok=True)
(out/"a6_smoke.ipynb").write_text(json.dumps(notebook))
print("Generated",out/"a6_smoke.ipynb")
