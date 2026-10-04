[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
foreach ($name in @('mpi-pcb-normal-bank-preparation-v1.zip', 'mpi-pcb-score-probe-v1.zip', 'mpi-supersimplenet-probe-v1.zip')) {
    $path = Join-Path $projectPath "artifacts\handoff\$name"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing reviewed handoff: $path" }
}
Set-Location $projectPath
Write-Host 'SuperSimpleNet: fresh 300 epochs / batch8 / 28800 updates, then normal128/169 calibration and reload.'
Write-Host 'No defect evaluation or service changes. Existing official weights are reused; no package upgrades.'
Write-Host 'Close other GPU jobs normally. Need 20 GiB free disk and 6 GiB free GPU.'
Write-Host '4-hour cumulative compute cap. Checkpoints every10 epochs; final300 only, no best-checkpoint selection.'
Write-Host 'Ctrl+C can resume from a committed checkpoint within remaining budget. Never delete failed outputs.'
& $defectPython 'scripts\run_mpi_supersimplenet_fit.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'SuperSimpleNet fit stopped. Preserve all outputs and send the error. Do not run test.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-fit-v1.zip'
Write-Host "Send this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
