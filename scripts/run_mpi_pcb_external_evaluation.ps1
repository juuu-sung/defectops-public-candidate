[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceZip = Join-Path $env:USERPROFILE "Downloads\extracted.zip"
$handoffZip = Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v1.zip"
$outputPath = Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v1"
Set-Location $projectPath

foreach ($file in @($defectPython, $sourceZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
foreach ($path in @($outputPath, $handoffZip)) {
    if (Test-Path -LiteralPath $path) {
        throw "Existing output was preserved: $path. Share the previous result/error before retrying."
    }
}

Write-Host "Frozen two-model external test: no training or threshold changes."
Write-Host "Normal replay first, then integrity audit, then 55 normal / 55 altered photos."
Write-Host "Close the GPU inspection server before running; do not open or tune on test images."
& $defectPython "scripts\evaluate_mpi_pcb_external.py" --project $projectPath --archive $sourceZip
$evaluationExitCode = $LASTEXITCODE
if (Test-Path -LiteralPath $handoffZip -PathType Leaf) {
    Write-Host "Attach this JSON-only handoff ZIP to the Codex task: $handoffZip"
    Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
}
if ($evaluationExitCode -ne 0) {
    throw "MPI-PCB evaluation stopped (exit $evaluationExitCode). Preserve all data and outputs; share the ZIP/error before retrying."
}
