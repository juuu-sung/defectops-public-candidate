[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceZip = Join-Path $env:USERPROFILE "Downloads\extracted.zip"
$evaluationPath = Join-Path $projectPath "artifacts\evaluations\mpi-pcb-external-evaluation-v2"
$evaluationZip = Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip"
$outputPath = Join-Path $projectPath "artifacts\analyses\mpi-pcb-map-diagnostics-v1"
$handoffZip = Join-Path $projectPath "artifacts\handoff\mpi-pcb-map-diagnostics-v1.zip"
Set-Location $projectPath

foreach ($file in @($defectPython, $sourceZip, $evaluationZip, (Join-Path $evaluationPath "recovery\grid1-3-defect-0054.png"))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file not found: $file. Preserve prior results."
    }
}
foreach ($model in @("realiad_pcb", "visa_pcb3_v2")) {
    $mapPath = Join-Path $evaluationPath "maps\$model"
    if (-not (Test-Path -LiteralPath $mapPath -PathType Container)) {
        throw "Saved maps not found: $mapPath. Do not rerun training or evaluation."
    }
}
foreach ($path in @($outputPath, $handoffZip)) {
    if (Test-Path -LiteralPath $path) { throw "Existing diagnostic output preserved: $path" }
}

Write-Host "Read-only: verify 218 saved maps; describe all 109 photos; render 61 review cases."
Write-Host "No GPU inference, training, threshold changes, downloads, or automatic PASS."
Write-Host "Native 256px evidence cannot prove pre-resize information loss."
& $defectPython "scripts\analyze_mpi_pcb_maps.py" --project $projectPath --archive $sourceZip
if ($LASTEXITCODE -ne 0) {
    throw "MPI-PCB diagnostics stopped. Original data, models, maps and evaluation were not changed. Share the error."
}
Write-Host "Attach this diagnostic ZIP: $handoffZip"
Write-Host "Optional local review: $outputPath\index.html"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
