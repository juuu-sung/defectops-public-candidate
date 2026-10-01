[CmdletBinding()]
param(
    [string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip')
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
$checkpoint = Join-Path $projectPath 'artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt'
foreach ($path in @($defectPython, $Archive, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input missing: $path. Preserve existing files; do not retrain."
    }
}
Set-Location $projectPath
& $defectPython -c 'import psutil'
if ($LASTEXITCODE -ne 0) {
    throw 'Memory monitor missing. Run: & ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements\mpi-normal-diversity-monitor.txt"'
}
Write-Host 'Normal-only cohort preparation: 1086 candidate photos -> 64 selected photos.'
Write-Host 'No model fit, threshold calculation, test image decoding or anomaly scoring.'
Write-Host 'Close other GPU jobs normally. First 16 normals must pass feature/replay/resource preflight.'
Write-Host 'The same command reuses verified descriptors. Do not delete output after errors.'
& $defectPython 'scripts\prepare_mpi_normal_diversity.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw 'Normal diversity preparation stopped. Send the error; keep all files. Do not start training.'
}
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-preparation-v1.zip'
Write-Host "Share this ZIP for selection review: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
