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
    (Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required existing input missing: $path. Do not retrain or delete existing results."
    }
}
Set-Location $projectPath
Write-Host "No retraining. Compare region-normalized top1 and native PatchCore raw score separately."
Write-Host "128 separate normal development + 169 calibration + SAME 109 previously observed test photos."
Write-Host "Close other GPU jobs normally first. Old checkpoint, thresholds and service are unchanged."
& $defectPython "scripts\probe_mpi_scores.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "MPI score probe stopped. Share error/attempt path; completed samples and old results are preserved."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-score-probe-v1.zip"
Write-Host "Attach this report and attributed review-card ZIP: $handoff"
Write-Host "Optional viewer: artifacts\analyses\mpi-pcb-score-probe-v1\result\index.html"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
