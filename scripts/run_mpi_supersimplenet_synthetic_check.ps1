[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep the existing final model and all outputs.'
Write-Host '48 fixed normal-only trials. No optimizer updates, test images or threshold changes.'
Write-Host 'Train-mode BatchNorm changes are temporary in memory and restored before inference.'
& $defectPython 'scripts\run_mpi_supersimplenet_synthetic_check.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Synthetic function check stopped. Preserve files and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-synthetic-check-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
