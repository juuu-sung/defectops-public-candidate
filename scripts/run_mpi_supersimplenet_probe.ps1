[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
$preparation = Join-Path $projectPath 'artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip'
foreach ($path in @($defectPython, $Archive, $preparation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Phase 1: SuperSimpleNet normal-only probe, NOT full training or accuracy evaluation.'
Write-Host 'Close other GPU jobs normally. Input batch8, internal feature batch16, max100 updates.'
Write-Host 'GPU phase: 15 minutes / 6 GiB reserved / 20 GiB process RAM caps. No automatic shrinking.'
Write-Host 'First run downloads pinned official weights (~276 MB); existing environments are not upgraded.'
& $defectPython 'scripts\run_mpi_supersimplenet_probe.py' --project $projectPath --archive $Archive --preparation $preparation --allow-download
if ($LASTEXITCODE -ne 0) { throw 'SuperSimpleNet probe stopped. Keep all files and send the error. Do not retrain or open test.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-probe-v1.zip'
Write-Host "Send this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
