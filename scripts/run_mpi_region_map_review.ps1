[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @(
    $defectPython, $Archive,
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-region-search-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-evaluation-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-score-probe-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-patch-trace-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-map-diagnostics-v1.zip"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Existing input missing: $path. Preserve the caches; no automatic inference or fitting."
    }
}
foreach ($path in @(
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-normal-bank-evaluation-v1\samples"),
    (Join-Path $projectPath "artifacts\analyses\mpi-pcb-region-search-v1\test")
)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Saved-map cache missing: $path. Do not retrain or regenerate maps; share this error."
    }
}
Set-Location $projectPath
Write-Host "CPU-only saved map review: all 23 decision changes + 7 fixed controls."
Write-Host "No model load, inference, fitting, threshold change, automatic PASS or ticket."
Write-Host "Calibration 0411: numeric comparison + historical shared-only reference. No candidate normal map exists."
& $defectPython "scripts\review_mpi_region_maps.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Saved-map review stopped. Original inputs and completed cards were preserved. Share the error."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-region-map-review-v1.zip"
$viewer = Join-Path $projectPath "artifacts\analyses\mpi-pcb-region-map-review-v1\result\index.html"
Write-Host "Attach this ZIP: $handoff"
Write-Host "Review viewer: $viewer"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
