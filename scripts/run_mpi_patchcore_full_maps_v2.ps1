[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve v1 failure, parity audit, models, caches and ZIPs.'
Write-Host 'Map-only precision checks -> same20 full-candidate maps. No training or threshold changes.'
Write-Host 'No time cap. Ctrl+C preserves completed blocks; other failures require review.'
& $defectPython 'scripts\compare_mpi_patchcore_full_maps_v2.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -eq 130) { Write-Host 'Interrupted; same v2 command resumes verified phases.'; exit 130 }
if ($LASTEXITCODE -ne 0) { throw 'Stable full-map comparison stopped. Preserve outputs and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-full-map-comparison-v2.zip'
Write-Host "Send this file: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
