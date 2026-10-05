[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve all prior fit/diagnostic folders and handoff ZIPs.'
Write-Host 'Frozen varying015500 replay -> original model copy with fresh segdec only -> 500 updates -> reload.'
Write-Host 'Same eight normal tiles, noise 0.015, learning rates and synthetic seeds. No test or threshold changes.'
& $defectPython 'scripts\run_mpi_supersimplenet_head_reset.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Head-reset comparison stopped. Preserve files and send the error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-head-reset-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
