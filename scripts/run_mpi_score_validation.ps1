[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$required = @(
    $defectPython, $Archive,
    (Join-Path $projectPath "configs\mpi-score-validation-v1.json"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-fit-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-score-probe-v1.zip"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"),
    (Join-Path $projectPath "artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt"),
    (Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2\recovery\grid1-3-defect-0054.png")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required existing input missing: $path. Preserve existing models and results."
    }
}
Set-Location $projectPath
Write-Host "Frozen control + Q90 + native MAD. Existing checkpoint; no training or recalibration."
Write-Host "Replay 406 observed photos first; only after success, audit 256 unused normal photos."
Write-Host "Close other GPU jobs normally first. Re-running this command resumes verified samples."
& $defectPython "scripts\validate_mpi_scores.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Score validation stopped. Share the error/attempt path. Do not delete or regenerate frozen evidence."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-score-validation-v1.zip"
Write-Host "Share this result ZIP: $handoff"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
