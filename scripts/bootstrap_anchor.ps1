$ErrorActionPreference = "Stop"

# Run from the RSNA repository root.
# This imports the public 0.941 anchor under vendor/anchor without replacing
# our research files. The anchor's own license and attribution stay with it.

$AnchorUrl = "https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection.git"
$RemoteName = "rsna-anchor"
$Prefix = "vendor/anchor"

git rev-parse --show-toplevel | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Run this script from inside a Git repository."
}

$existing = git remote get-url $RemoteName 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add $RemoteName $AnchorUrl
} elseif ($existing.Trim() -ne $AnchorUrl) {
    git remote set-url $RemoteName $AnchorUrl
}

git fetch --depth 1 $RemoteName main

if (Test-Path $Prefix) {
    throw "$Prefix already exists. Refusing to overwrite the anchor."
}

git subtree add --prefix=$Prefix "$RemoteName/main" --squash

Write-Host ""
Write-Host "Anchor imported under $Prefix"
Write-Host "Next: run scripts/verify_anchor.ps1"
