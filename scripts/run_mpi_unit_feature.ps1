[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
& $defectPython -c 'import psutil'
if ($LASTEXITCODE -ne 0) { throw 'Memory monitor psutil is missing. Preserve files and report this error.' }
Write-Host 'One fixed feature-normalization candidate. No training, no coreset rebuild, no service change.'
Write-Host 'Normal calibration -> reload297 -> previously observed comparison -> conditional stability check.'
Write-Host 'Close other GPU jobs normally. Cumulative cap 2 hours; reserved 6 GiB; RSS 20 GiB.'
& $defectPython 'scripts\run_mpi_unit_feature.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Unit-feature experiment stopped. Keep all outputs and send the error; do not change the recipe.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-unit-feature-v1.zip'
Write-Host "Share this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
