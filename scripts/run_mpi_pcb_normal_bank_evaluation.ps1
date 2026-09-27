[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @(
    $defectPython,
    $Archive,
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-fit-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-map-diagnostics-v1.zip"),
    (Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required existing input is missing: $path. Preserve existing results; do not retrain."
    }
}
Set-Location $projectPath
Write-Host "NEW Python process: load saved MPI bank, check fixed normal scores/speed, evaluate the SAME 109 photos."
Write-Host "No fitting, new downloads, threshold changes or automatic PASS. Stop other GPU jobs normally first."
Write-Host "Normal probe stops before test on score drift or severe slowdown. Completed test photos are reusable."
& $defectPython "scripts\evaluate_mpi_pcb_normal_bank.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "MPI target-bank evaluation stopped. Share the error and printed attempt path. Do not delete model, data or outputs."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-evaluation-v1.zip"
Write-Host "Attach this JSON-only evaluation handoff: $handoff"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
