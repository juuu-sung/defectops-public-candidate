[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
$required = @($defectPython, (Join-Path $projectPath 'artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt'))
foreach ($name in @('mpi-normal-visual-review-v1', 'mpi-spatial-score-v1', 'mpi-pcb-normal-bank-preparation-v1', 'mpi-pcb-normal-bank-fit-v1')) {
    $required += Join-Path $projectPath "artifacts\handoff\$name.zip"
}
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input missing: $path. Do not delete or rebuild models."
    }
}
Set-Location $projectPath
Write-Host 'Five calibration normals / 60 tiles. Existing model only; no training or threshold changes.'
Write-Host 'Close other GPU jobs normally. The same command resumes verified completed photos.'
& $defectPython 'scripts\review_mpi_normal_maps.py' --project $projectPath
if ($LASTEXITCODE -ne 0) { throw 'Normal map review stopped. Send the error; preserve all results.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-map-review-v1.zip'
Write-Host "Share this ZIP for private review: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
