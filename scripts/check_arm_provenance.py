#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry",default="validation/A5_ARM_REGISTRY.json")
    ap.add_argument("--arm",action="append",required=True)
    ap.add_argument("--purpose",choices=["gold58","public_parent"],required=True)
    a=ap.parse_args()
    reg=json.loads(Path(a.registry).read_text())
    for name in a.arm:
        if name not in reg["arms"]:
            raise SystemExit(f"unknown arm: {name}")
        arm=reg["arms"][name]
        if a.purpose=="gold58" and not arm.get("gold58_allowed",False):
            raise SystemExit(f"REFUSE {name}: provenance={arm.get('provenance')} cannot enter gold58 gate")
        print(f"ALLOW {name}: purpose={a.purpose} provenance={arm.get('provenance')}")
if __name__=="__main__":
    main()
