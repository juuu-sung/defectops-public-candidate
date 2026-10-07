[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'Read-only: 14 high-score normals and defect0024, all12 regions. No GPU or training.'
Write-Host 'Keep original archive, evaluation caches and all previous ZIPs unchanged.'
& $defectPython 'scripts\review_mpi_patchcore_score_overlap.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Score overlap export stopped. Keep files and send the full error. Do not rerun training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-score-overlap-v1.zip'
Write-Host "Send this file: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
