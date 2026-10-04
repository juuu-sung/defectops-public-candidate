[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Frozen SuperSimpleNet epoch300 / step28800. No training or threshold tuning.'
Write-Host 'Replay 8 normals first. Fixed 12-case screen; only 12/12 opens full observed comparison and warm timing.'
Write-Host 'Accuracy and speed must pass before 600 paired normal calibration sensitivity checks.'
Write-Host 'Close other GPU jobs normally. Need 12 GiB free disk and 6 GiB free GPU; cumulative evaluation cap 4 hours.'
Write-Host 'Keep existing fit, handoff ZIPs and approved recovery image unchanged. No package upgrades.'
& $defectPython 'scripts\run_mpi_supersimplenet_evaluation.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'SuperSimpleNet evaluation stopped. Keep all files and send the error. Do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-evaluation-v1.zip'
Write-Host "Send this JSON-only ZIP even when a performance gate failed: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
