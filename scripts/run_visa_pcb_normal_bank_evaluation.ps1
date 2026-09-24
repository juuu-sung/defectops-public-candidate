[CmdletBinding()]
param([switch]$RunExploratory)

$ErrorActionPreference = "Stop"
if (-not $RunExploratory) {
    throw "VisA test was already observed. Pass -RunExploratory to run this descriptive comparison."
}
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$sourceTar = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
$fitZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-pcb3-v2.zip"
$fitCheckpoint = Join-Path $projectPath "artifacts\visa-pcb-normal-bank-development-v2\pcb3\model.pt"
$baselineZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-external-evaluation-v2.zip"

Set-Location $projectPath
foreach ($file in @($defectPython, $sourceTar, $fitZip, $fitCheckpoint, $baselineZip)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required existing file was not found: $file"
    }
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Preserve and resolve local changes before evaluation."
}

Write-Host "Comparing PCB3 v2 against the locked prior PCB3 result on the same 201 already-observed test images."
Write-Host "This is post-test exploratory analysis only; no automatic PASS or model selection."
& $defectPython "scripts\evaluate_visa_pcb_normal_bank.py" `
    --project $projectPath `
    --archive $sourceTar `
    --exploratory-reuse-observed-test
if ($LASTEXITCODE -ne 0) {
    throw "Exploratory PCB3 evaluation stopped. Fit, source data, and completed samples were not deleted."
}

$handoffZip = Join-Path $projectPath "artifacts\handoff\visa-pcb-normal-bank-pcb3-evaluation-v1.zip"
Write-Host "Attach this metadata-only handoff to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
