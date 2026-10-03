[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Frozen 70000-update EfficientAD. No training or threshold tuning.'
Write-Host '12-case screen first; only 12/12 opens full observed comparison and warm timing.'
Write-Host 'If accuracy and speed pass: 600 normal resampling checks. 4-hour cumulative cap; 12 GiB disk and 6 GiB GPU free.'
Write-Host 'Close other GPU jobs normally. Keep the existing fit folder and handoff ZIP unchanged.'
Write-Host 'Evaluation v2 preserves failed v1 outputs and counts their recorded time. No retraining.'
& $defectPython 'scripts\run_mpi_efficientad_evaluation.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'EfficientAD evaluation stopped. Keep all files and send the error. Do not rerun training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-evaluation-v2.zip'
Write-Host "Send this JSON-only ZIP even when a performance gate failed: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
