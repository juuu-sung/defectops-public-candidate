[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'CPU read-only: 8 new false positives, 1 resolved normal, 5 defects; all12 regions.'
Write-Host 'No GPU, training, inference, new calibration or threshold changes. Preserve all prior results.'
& $defectPython 'scripts\review_mpi_patchcore_balanced_maps.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Paired map review stopped. Keep outputs and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-balanced-map-review-v1.zip'
Write-Host "Send this file: $handoff"
Write-Host 'To view locally, extract this NEW ZIP to a separate folder and open index.html. Do not alter old outputs.'
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
