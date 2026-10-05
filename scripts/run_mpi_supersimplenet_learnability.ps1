[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep the original fit and handoff ZIP unchanged.'
Write-Host 'New diagnostic copy: 8 fixed normal tiles, identical synthetic task, 500 updates.'
Write-Host 'No test access, threshold changes or automatic service model replacement.'
Write-Host 'A normal Ctrl+C can resume at a verified checkpoint; hard stops/errors require review.'
& $defectPython 'scripts\run_mpi_supersimplenet_learnability.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Limited learning stopped. Preserve outputs and send the error; do not extend training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-learnability-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
