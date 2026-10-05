[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve all prior model folders and handoff ZIPs.'
Write-Host '24 read-only gradient measurements. No optimizer step, parameter reset, real test or threshold change.'
& $defectPython 'scripts\run_mpi_supersimplenet_gradients.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Gradient audit stopped. Preserve files and send the error; do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-gradient-audit-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
