[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'CPU-only saved-map review: 12 observed misses + 8 normal controls, no inference or training.'
Write-Host 'Keep original fit/evaluation/recovery files and ZIPs. Do not run another job writing those folders.'
Write-Host 'Verifying original inputs and saved maps can take several minutes. At least 1 GiB free disk required.'
& $defectPython 'scripts\review_mpi_efficientad_maps.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Saved-map review stopped. Preserve files and send the error. Do not rerun training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-efficientad-map-diagnostics-v1.zip'
Write-Host "Send this private-review ZIP (contains derived diagnostic images): $handoff"
Write-Host "Local viewer: $projectPath\artifacts\analyses\mpi-efficientad-map-diagnostics-v1\bundle\index.html"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
