[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Preserve formal step0/final models and all prior ZIPs.'
Write-Host 'Trained adaptor vs initial adaptor; identical preserved step0 head; fresh AdamW in both.'
Write-Host '500 updates per arm. Same eight normal tiles and synthetic schedule. No elapsed-time cap.'
Write-Host 'Normal controls, held-out synthetic seeds and reload verification. No test or threshold changes.'
& $defectPython 'scripts\run_mpi_supersimplenet_adaptor_start.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Adaptor comparison stopped. Preserve files and send the full error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-adaptor-start-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
