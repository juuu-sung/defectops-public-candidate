[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'EfficientAD-S: 70000 updates, then normal128/169 calibration and reload. No test evaluation.'
Write-Host 'Use the existing official assets. Need 30 GiB free disk and 6 GiB free GPU; close other GPU jobs normally.'
Write-Host '4-hour cumulative compute cap. Checkpoint every 1000 updates. Never delete failed outputs.'
& $defectPython 'scripts\run_mpi_efficientad_fit.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'EfficientAD fit stopped. Keep all outputs and send the error. Do not run test.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-fit-v1.zip'
Write-Host "Send this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
