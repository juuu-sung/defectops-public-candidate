[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))

$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($required in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}
Set-Location $projectPath
Write-Host 'Exporting 10 normal photos / 120 tiles. CPU only; no model or test decoding.'
& $defectPython 'scripts\export_mpi_normal_visual_review.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw 'Normal visual export stopped. Original data and prior outputs preserved. Send the error; do not delete results.'
}
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-visual-review-v1.zip'
Write-Host "Attach this file for private review: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
