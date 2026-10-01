[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
$checkpoint = Join-Path $projectPath 'artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt'
$selection = Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-preparation-v1.zip'
foreach ($path in @($defectPython, $Archive, $checkpoint, $selection)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input missing: $path. Preserve existing files; do not rerun selection."
    }
}
Set-Location $projectPath
& $defectPython -c 'import psutil'
if ($LASTEXITCODE -ne 0) {
    throw 'Memory monitor missing. Install requirements\mpi-normal-diversity-monitor.txt with the project Python.'
}
Write-Host 'C-stage: locked 64-photo bank -> development128/calibration169 -> reload297.'
Write-Host 'No test scoring, model promotion or automatic PASS. Close other GPU jobs normally.'
Write-Host 'Budget: cumulative 4 hours, PyTorch reserved 7 GiB, process RSS 20 GiB.'
Write-Host 'Verified phases/photo scores are reusable. Forced termination without a cost receipt requires review.'
& $defectPython 'scripts\fit_mpi_normal_diversity.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw 'Normal diversity fit stopped. Keep all files and send the error. Do not run test evaluation.'
}
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-fit-v1.zip'
Write-Host "Share this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
