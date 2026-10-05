[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve all fit folders and prior handoff ZIPs.'
Write-Host 'Four comparisons: saved step0/final x two fixed synthetic seeds.'
Write-Host 'Each path performs one disposable AdamW update. No updated model is saved or deployed.'
Write-Host 'No test images, threshold changes, automatic next phase or retraining.'
& $defectPython 'scripts\run_mpi_supersimplenet_parity.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Implementation audit stopped. Keep files and send the full error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-implementation-parity-v1.zip'
Write-Host "Send this result ZIP even if differences need review: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
