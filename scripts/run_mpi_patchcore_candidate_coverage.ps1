[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep all original models, feature caches and handoff ZIPs.'
Write-Host 'Fixed20 tiles: representative bank versus all retained candidate features. No fitting or threshold change.'
Write-Host 'No elapsed-time cap. Ctrl+C can resume verified blocks; errors require review.'
& $defectPython 'scripts\diagnose_mpi_patchcore_candidate_coverage.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -eq 130) { Write-Host 'Interrupted safely. Rerun the same command to resume.'; exit 130 }
if ($LASTEXITCODE -ne 0) { throw 'Candidate coverage diagnosis stopped. Preserve outputs and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-candidate-coverage-v1.zip'
Write-Host "Send this file: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
