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
Write-Host 'Raw-feature nearest3 screen only. Existing model/bank unchanged; no training or full evaluation.'
Write-Host '297 normal calibration -> 12 fixed prior detections -> conditional 20-photo reload.'
Write-Host 'Close other GPU jobs normally. Cumulative cap 1 hour; reserved 6 GiB; RSS 20 GiB.'
& $defectPython 'scripts\run_mpi_knn3_screen.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Screen stopped. Keep all outputs and send the error; do not change cases or threshold.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-knn3-screen-v1.zip'
Write-Host "Share this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
