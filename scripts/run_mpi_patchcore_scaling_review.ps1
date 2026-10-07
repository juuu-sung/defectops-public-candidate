[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'Read-only: verify existing scores and copy selected NORMAL images and saved maps.'
Write-Host 'No GPU, new inference, training, threshold update or original-data changes.'
Write-Host 'Keep the finished scaling folders and earlier handoff ZIPs unchanged.'
& $defectPython 'scripts\export_mpi_patchcore_scaling_review.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Read-only export stopped. Preserve files and send the full error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-scaling-normal-review-v1.zip'
Write-Host "Send this ZIP privately to the current task: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
