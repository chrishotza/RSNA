$ErrorActionPreference = "Stop"
$Notebook = Join-Path $PSScriptRoot "..\notebooks\A0_public_0941\rsna-knee-blend-full.ipynb"
if (-not (Test-Path $Notebook)) { throw "A0 notebook missing: $Notebook" }
python -c "import json,sys; p=r'$Notebook'; n=json.load(open(p,encoding='utf-8')); c=n['cells']; assert len(c)==25; s=lambda i: ''.join(c[i].get('source',[])); assert 'APPENDED ARM BLEND' in s(23); assert '_DINOV2_MATCHED_MEMBERS' in s(24); assert 'V18_CALIBRATOR_APPLIED' in s(24); print('A0 notebook integrity: PASS')"
if ($LASTEXITCODE -ne 0) { throw "A0 integrity check failed" }
