$ErrorActionPreference = "Stop"

$required = @(
    "vendor/anchor/README.md",
    "vendor/anchor/LICENSE",
    "vendor/anchor/train/train_armA.py",
    "vendor/anchor/cache/build_cache.py",
    "vendor/anchor/submit/make_blend_cell.py",
    "vendor/anchor/submit/push_submission.py"
)

$missing = @()
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Host "ANCHOR INVALID"
    $missing | ForEach-Object { Write-Host "MISSING: $_" }
    exit 1
}

Write-Host "ANCHOR OK"
Write-Host "Required files present: $($required.Count)"
