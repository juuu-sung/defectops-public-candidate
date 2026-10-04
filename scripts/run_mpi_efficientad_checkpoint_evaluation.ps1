[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep all historical fit/checkpoint folders and handoff ZIPs.'
Write-Host 'No training. Fixed 10k and 30k only; separate normal calibration, reload, 12-case screen.'
Write-Host 'Full comparison only after 12/12. Stability only after accuracy and warm gates pass.'
Write-Host 'Need 30 GiB free disk and 6 GiB free GPU. Up to 4 hours per model, 8 hours total.'
& $defectPython 'scripts\run_mpi_efficientad_checkpoint_evaluation.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint evaluation stopped. Keep outputs and send the full error. Do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-checkpoint-evaluation-v1.zip'
Write-Host "Send this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
