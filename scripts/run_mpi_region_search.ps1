[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @(
    $defectPython, $Archive,
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-fit-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-evaluation-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-map-diagnostics-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-score-probe-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-patch-trace-v1.zip"),
    (Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Existing input missing: $path. Preserve old data/model. No automatic retraining."
    }
}
Set-Location $projectPath
Write-Host "Shared bank + A versus same-region search + normally recalibrated A."
Write-Host "No model fitting, coreset selection, new data, or service model change."
Write-Host "Close other GPU work normally first. Completed phases are verified before reuse."
& $defectPython "scripts\compare_mpi_regions.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "MPI region comparison stopped. Share the error; completed phases and original data/model are preserved."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-region-search-v1.zip"
Write-Host "Attach this ZIP: $handoff"
Write-Host "Optional viewer: artifacts\analyses\mpi-pcb-region-search-v1\result\index.html"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
