[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'CPU saved-map review: 15 original false alerts, 241 normal controls, all55 observed defects.'
Write-Host 'No GPU/model loading, inference, training, threshold changes or border cropping.'
& $defectPython 'scripts\review_mpi_patchcore_original_edges.py' --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Original-edge review stopped. Preserve all files and send the error.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-original-edge-review-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
