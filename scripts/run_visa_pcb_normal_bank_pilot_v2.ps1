[CmdletBinding()]
param([ValidateSet("pcb1", "pcb2", "pcb3", "pcb4")][string]$Category = "pcb3")

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceTar = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
$previousEvaluation = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-evaluation-v2.zip"
$previousPreflight = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-preflight-v1.zip"

Set-Location $projectPath
foreach ($file in @($defectPython, $sourceTar, $previousEvaluation, $previousPreflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Preserve and resolve local changes before fitting."
}

Write-Host "Fitting $Category from all disjoint official normal train images with an evenly spaced 1/4 patch pool."
Write-Host "No official test image or mask is opened. This changed post-test development run is exploratory."
Write-Host "Runtime is not established; leave the computer available and do not treat the first progress-bar estimate as reliable."
& $defectPython "scripts\fit_visa_pcb_normal_bank.py" `
    --project $projectPath `
    --archive $sourceTar `
    --category $Category `
    --version 2
if ($LASTEXITCODE -ne 0) {
    throw "Memory-bounded normal-bank fit stopped. Original data and prior results were not deleted."
}

$handoffZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-$Category-v2.zip"
Write-Host "Attach this metadata-only handoff to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
