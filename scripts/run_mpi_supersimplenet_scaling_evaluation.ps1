[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive, (Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-scaling-v1.zip'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Frozen A/C/D: all55 defects + normal54 + separate normal256. Prior12 does NOT stop full evaluation.'
Write-Host 'Then saved map attribution, warm cost and 600 normal-calibration draws per arm. No training or threshold changes.'
Write-Host 'No elapsed-time cap. Need 20 GiB disk, 6 GiB free GPU; RAM20 / GPU reserved6 GiB limits remain.'
Write-Host 'Close other GPU jobs normally. Preserve all fit folders and ZIPs. Do not git pull while running.'
Write-Host 'Ctrl+C once and wait. Recorded interruptions resume with this same command; other errors require review.'
& $defectPython 'scripts\run_mpi_supersimplenet_scaling_evaluation.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Scaling evaluation stopped. Keep all files and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-scaling-evaluation-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
