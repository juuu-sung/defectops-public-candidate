[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip"))

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$preparation = Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip"
foreach ($path in @($defectPython, $Archive, $preparation)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required existing input is missing: $path"
    }
}
Set-Location $projectPath
Write-Host "MPI target-normal-bank pilot: 64 original training photos, 169 normal calibration photos."
Write-Host "Coreset selection may take hours. Consider stopping other GPU jobs first."
Write-Host "No test images, test masks, threshold tuning on test, or service model replacement."
& $defectPython "scripts\fit_mpi_pcb_normal_bank.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "MPI normal-bank fit stopped. Completed phases and incomplete attempts were preserved. Share the error; do not delete results."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-fit-v1.zip"
Write-Host "Attach this JSON-only fit handoff: $handoff"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
