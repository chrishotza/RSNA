import json,subprocess,sys
from pathlib import Path

def test_registry_rejects_contaminated(tmp_path):
    reg={"arms":{"bad":{"gold58_allowed":False,"provenance":"UNKNOWN"},"good":{"gold58_allowed":True,"provenance":"CLEAN"}}}
    p=tmp_path/"r.json";p.write_text(json.dumps(reg))
    ok=subprocess.run([sys.executable,"scripts/check_arm_provenance.py","--registry",str(p),"--arm","good","--purpose","gold58"])
    assert ok.returncode==0
    bad=subprocess.run([sys.executable,"scripts/check_arm_provenance.py","--registry",str(p),"--arm","bad","--purpose","gold58"])
    assert bad.returncode!=0
