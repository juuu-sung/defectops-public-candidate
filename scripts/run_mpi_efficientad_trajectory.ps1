[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Close other GPU jobs normally. No new training, threshold changes or model selection.'
Write-Host 'Keep fit/evaluation/checkpoint folders and handoff ZIPs. Need 5 GiB disk and 6 GiB free GPU.'
Write-Host 'Input verification -> training64 atlas -> final replay -> 10k/30k raw maps -> private-review ZIP.'
& $defectPython 'scripts\run_mpi_efficientad_trajectory.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint comparison stopped. Keep files and send the full error. Do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-trajectory-v1.zip'
Write-Host "Send this ZIP (derived images, no model weights): $handoff"
Write-Host "Viewer: $projectPath\artifacts\analyses\mpi-efficientad-trajectory-v1\bundle\index.html"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
