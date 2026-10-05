[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. Keep original, fixed500 and varying500 folders and handoff ZIPs unchanged.'
Write-Host 'Read-only baseline -> std0.03 x500 -> fresh original std0.06 x500 -> cross-strength reload checks.'
Write-Host 'Same eight normal tiles; no real test, threshold change or automatic model selection.'
& $defectPython 'scripts\run_mpi_supersimplenet_noise.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Noise-strength comparison stopped. Preserve all files and send the error; do not reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-noise-strength-v1.zip'
Write-Host "Send this result ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
