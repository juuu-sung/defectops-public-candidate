[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep original fit, fixed500 folder and handoff ZIP unchanged.'
Write-Host 'Read-only fixed500 control -> original model copy + 500 varying-seed updates -> final reload.'
Write-Host 'Eight SAME normal tiles; four held-out synthetic seeds. No real test or service changes.'
& $defectPython 'scripts\run_mpi_supersimplenet_varying.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Varying synthetic experiment stopped. Preserve files and send the error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-varying-synthetic-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
