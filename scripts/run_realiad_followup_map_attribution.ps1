[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$finalZip = Join-Path $projectPath "artifacts\handoff\realiad-followup-final-v1.zip"
$priorDiagnostics = Join-Path $projectPath "artifacts\handoff\realiad-followup-final-errors-v1.zip"
$sourceZip = Join-Path $env:USERPROFILE "Downloads\pcb.zip"
$samplesRoot = Join-Path $projectPath "artifacts\realiad-followup-final-v1\evaluations"
$outputPath = Join-Path $projectPath "artifacts\analyses\realiad-followup-map-attribution-v1"
$handoffZip = Join-Path $projectPath "artifacts\handoff\realiad-followup-map-attribution-v1.zip"

Set-Location $projectPath
foreach ($file in @($defectPython, $finalZip, $priorDiagnostics, $sourceZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
if (-not (Test-Path -LiteralPath $samplesRoot -PathType Container)) {
    throw "Saved final maps were not found: $samplesRoot"
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Existing map attribution output was not overwritten: $outputPath"
}
if (Test-Path -LiteralPath $handoffZip) {
    throw "Existing map attribution handoff was not overwritten: $handoffZip"
}

& $defectPython "scripts\analyze_realiad_followup_map_attribution.py" `
    --handoff $finalZip `
    --prior-diagnostics $priorDiagnostics `
    --source-zip $sourceZip `
    --samples-root $samplesRoot `
    --output $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Read-only map attribution failed with exit code $LASTEXITCODE"
}

Compress-Archive -Path (Join-Path $outputPath "*") -DestinationPath $handoffZip
Write-Host "Descriptive saved-map attribution completed. No model or threshold changed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
