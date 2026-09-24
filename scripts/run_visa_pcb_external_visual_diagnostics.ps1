[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$evaluationZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-evaluation-v2.zip"
$sourceTar = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
$samplesRoot = Join-Path $projectPath "artifacts\visa-pcb-external-evaluation-v2\samples"
$outputPath = Join-Path $projectPath "artifacts\analyses\visa-pcb-external-visual-diagnostics-v1"
$handoffZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-visual-diagnostics-v1.zip"
$planPath = Join-Path $projectPath "data\manifests\visa-pcb-external-plan-v1.json"
$configPath = Join-Path $projectPath "configs\visa-pcb-external-evaluation-v2.toml"

Set-Location $projectPath
foreach ($file in @($defectPython, $evaluationZip, $sourceTar, $planPath, $configPath)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
if (-not (Test-Path -LiteralPath $samplesRoot -PathType Container)) {
    throw "Saved evaluation maps were not found: $samplesRoot"
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Existing visual diagnostic output was not overwritten: $outputPath"
}
if (Test-Path -LiteralPath $handoffZip) {
    throw "Existing visual diagnostic handoff was not overwritten: $handoffZip"
}

& $defectPython "scripts\analyze_visa_pcb_external_visual.py" `
    --handoff $evaluationZip `
    --plan $planPath `
    --config $configPath `
    --archive $sourceTar `
    --samples $samplesRoot `
    --output $outputPath `
    --handoff-output $handoffZip
if ($LASTEXITCODE -ne 0) {
    throw "Read-only VisA visual diagnostics stopped. Source data, model and thresholds were not changed."
}

Write-Host "Post-test descriptive visual review is ready. No model inference or tuning was performed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
