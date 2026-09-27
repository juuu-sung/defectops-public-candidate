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
    (Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Existing input missing: $path. Do not retrain or delete old results."
    }
}
Set-Location $projectPath
Write-Host "12 selected A cases; replay 64 original normal photos to trace patch origins."
Write-Host "No fitting, coreset selection, threshold change or service update."
Write-Host "Close other GPU work normally first. Unmatched source vectors stay unresolved."
& $defectPython "scripts\trace_mpi_patches.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "MPI patch trace stopped. Share the error; completed phases and old data/model are preserved."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-patch-trace-v1.zip"
Write-Host "Attach this ZIP: $handoff"
Write-Host "Optional viewer: artifacts\analyses\mpi-pcb-patch-trace-v1\result\index.html"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
