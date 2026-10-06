[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep the formal fit and prior parity ZIP unchanged.'
Write-Host 'Two fixed seeds x four paths, each from step zero with one disposable update.'
Write-Host 'Repeat control -> same loss control -> author loss comparison -> post-update maps/scores.'
Write-Host 'No elapsed-time cap. Memory safeguards remain. No test, retraining or threshold changes.'
& $defectPython 'scripts\run_mpi_supersimplenet_update_audit.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Update audit stopped. Preserve outputs and send the full error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-update-audit-v1.zip'
Write-Host "Send this ZIP even if comparisons differ: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
