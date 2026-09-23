[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$finalZip = Join-Path $projectPath "artifacts\handoff\realiad-followup-final-v1.zip"
$samplesRoot = Join-Path $projectPath "artifacts\realiad-followup-final-v1\evaluations"
$sourceZip = Join-Path $env:USERPROFILE "Downloads\pcb.zip"
$outputPath = Join-Path $projectPath "artifacts\analyses\realiad-followup-final-errors-v1"
$handoffZip = Join-Path $projectPath "artifacts\handoff\realiad-followup-final-errors-v1.zip"

Set-Location $projectPath
foreach ($file in @($defectPython, $finalZip, $sourceZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
if (-not (Test-Path -LiteralPath $samplesRoot -PathType Container)) {
    throw "Saved final sample maps were not found: $samplesRoot"
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Existing diagnostic output was not overwritten: $outputPath"
}
if (Test-Path -LiteralPath $handoffZip) {
    throw "Existing diagnostic handoff was not overwritten: $handoffZip"
}

& $defectPython "scripts\analyze_realiad_followup_final.py" `
    --handoff $finalZip `
    --source-zip $sourceZip `
    --samples-root $samplesRoot `
    --output $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Read-only final diagnostics failed with exit code $LASTEXITCODE"
}

Compress-Archive -Path (Join-Path $outputPath "*") -DestinationPath $handoffZip
Write-Host "Post-final descriptive diagnostics completed. No model or threshold changed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
