[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))
$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @($defectPython, $Archive)
foreach ($name in @("mpi-score-validation-v1", "mpi-pcb-normal-bank-preparation-v1", "mpi-pcb-normal-bank-fit-v1", "mpi-pcb-score-probe-v1", "mpi-pcb-external-evaluation-v2")) {
    $required += Join-Path $projectPath "artifacts\handoff\$name.zip"
}
$required += Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"
$required += Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png"
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required existing input missing: $path. Do not regenerate or delete frozen evidence."
    }
}
Set-Location $projectPath
Write-Host "Frozen model, one 26x26 spatial score candidate. No training or mask-based scoring."
Write-Host "Normal development 128 -> calibration 169 -> observed test 109 -> observed normals 256."
Write-Host "All 662 photos are already observed; this is exploratory, not an independent validation."
Write-Host "Finish other GPU work first. The same command resumes verified completed photos."
& $defectPython "scripts\run_mpi_spatial_score.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Spatial comparison stopped. Share the error; preserve existing results."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-spatial-score-v1.zip"
Write-Host "Share this ZIP: $handoff"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
