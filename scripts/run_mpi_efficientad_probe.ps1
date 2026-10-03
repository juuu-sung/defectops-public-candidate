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
Write-Host 'EfficientAD-S normal-only resource probe. No defect evaluation or full training.'
Write-Host 'First run downloads official Imagenette and teacher weights (allow several GB free disk).'
Write-Host 'Research use only. Original images and weights are not exported or pushed to Git.'
Write-Host 'Close other GPU jobs normally. 100 attempted updates max; GPU phase 15 minutes max.'
& $defectPython 'scripts\run_mpi_efficientad_probe.py' --project $projectPath --archive $Archive --preparation $preparation --allow-download
if ($LASTEXITCODE -ne 0) { throw 'Probe stopped. Preserve all files and send the error. Do not run full training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-probe-v1.zip'
Write-Host "Send this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
