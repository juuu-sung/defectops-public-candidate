[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @($defectPython, $Archive)
foreach ($name in @("mpi-spatial-score-v1", "mpi-score-validation-v1", "mpi-pcb-normal-bank-preparation-v1", "mpi-pcb-normal-bank-fit-v1", "mpi-pcb-score-probe-v1", "mpi-pcb-external-evaluation-v2")) {
    $required += Join-Path $projectPath "artifacts\handoff\$name.zip"
}
$required += Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input missing: $path. Preserve existing data and models."
    }
}
Set-Location $projectPath
Write-Host "Six fixed defects: 0019, 0024, 0021, 0027, 0028, 0047; all 72 tiles."
Write-Host "No training or threshold changes. GT is used only after score replay."
Write-Host "Close other GPU jobs normally. Same command resumes verified cases."
& $defectPython "scripts\review_mpi_spatial_misses.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Six-case review stopped. Share the error; do not delete existing results."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-spatial-miss-review-v1.zip"
Write-Host "Share this ZIP: $handoff"
Write-Host "Local viewer: artifacts\analyses\mpi-spatial-miss-review-v1\result\index.html"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
