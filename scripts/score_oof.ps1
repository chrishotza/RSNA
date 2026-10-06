param(
    [Parameter(Mandatory=$true)]
    [string]$OofCsv,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OofCsv)) {
    throw "OOF file not found: $OofCsv"
}

& $Python -m src.proxy_score score --oof $OofCsv
if ($LASTEXITCODE -ne 0) {
    throw "OOF scoring failed with exit code $LASTEXITCODE"
}
