[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceTar = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
$oldZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-evaluation-v2.zip"
$newZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-pcb3-evaluation-v1.zip"
$fitZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-pcb3-v2.zip"
$oldSamples = Join-Path $projectPath "artifacts\visa-pcb-external-evaluation-v2\samples"
$newSamples = Join-Path $projectPath "artifacts\visa-pcb-normal-bank-pcb3-evaluation-v1\samples"
$outputPath = Join-Path $projectPath "artifacts\analyses\visa-pcb-normal-bank-pcb3-errors-v1"
$handoffZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-pcb3-errors-v1.zip"

Set-Location $projectPath
foreach ($file in @($defectPython, $sourceTar, $oldZip, $newZip, $fitZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
foreach ($folder in @($oldSamples, $newSamples)) {
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) {
        throw "Saved evaluation maps were not found: $folder"
    }
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Existing visual diagnostic output was not overwritten: $outputPath"
}
if (Test-Path -LiteralPath $handoffZip) {
    throw "Existing visual diagnostic handoff was not overwritten: $handoffZip"
}

Write-Host "Preparing read-only review cards for all 9 PCB3 misses and 6 normal false positives."
Write-Host "No training, inference, threshold changes, automatic PASS or production decisions."
& $defectPython "scripts\analyze_visa_pcb_normal_bank_visual.py" `
    --project $projectPath `
    --archive $sourceTar
if ($LASTEXITCODE -ne 0) {
    throw "PCB3 visual review stopped. Original data, models, saved maps and thresholds were not changed."
}

Write-Host "Attach this visual handoff ZIP to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
