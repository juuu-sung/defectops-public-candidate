[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve all previous models, caches and ZIPs.'
Write-Host 'Fixed20 full maps: no fitting, new thresholds or photo PASS decisions. No time cap.'
Write-Host 'First completed block reports search cost; Ctrl+C resumes completed blocks.'
& $defectPython 'scripts\compare_mpi_patchcore_full_maps.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -eq 130) { Write-Host 'Interrupted. Same command resumes verified blocks.'; exit 130 }
if ($LASTEXITCODE -ne 0) { throw 'Full-map diagnosis stopped. Keep outputs and send the error; do not rerun training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-full-map-comparison-v1.zip'
Write-Host "Send this file: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
