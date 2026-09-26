[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceZip = Join-Path $env:USERPROFILE "Downloads\extracted.zip"
$priorZip = Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v1.zip"
$handoffZip = Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"
$outputPath = Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2"
Set-Location $projectPath

foreach ($file in @($defectPython, $sourceZip, $priorZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
foreach ($path in @($outputPath, $handoffZip)) {
    if (Test-Path -LiteralPath $path) {
        throw "Existing v2 output preserved: $path. Share the result/error before retrying."
    }
}

Write-Host "Integrity-amended v2: no training or threshold changes. Preserve all v1 files."
Write-Host "Internet required: about 4.3 MB of official aligned source data, NOT the 5.45 GB ZIP."
Write-Host "Verify 11 intact tiles, reconstruct one tile, exclude one duplicate normal, retain mask-only case."
Write-Host "Close the GPU inspection server. Compare 54 normal / 55 altered photos, plus 108-photo sensitivity."
& $defectPython "scripts\evaluate_mpi_pcb_external_v2.py" --project $projectPath --archive $sourceZip
$evaluationExitCode = $LASTEXITCODE
if (Test-Path -LiteralPath $handoffZip -PathType Leaf) {
    Write-Host "Attach this JSON-only handoff ZIP: $handoffZip"
    Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
}
if ($evaluationExitCode -ne 0) {
    throw "MPI-PCB v2 stopped (exit $evaluationExitCode). Do not delete data or outputs; share the ZIP/error."
}
