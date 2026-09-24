[CmdletBinding()]
param([switch]$RunCorrected)

$ErrorActionPreference = "Stop"
if (-not $RunCorrected) {
    throw "This is a separate exploratory run after the first VisA test mask was observed. Pass -RunCorrected explicitly."
}
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$archivePath = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Registered VisA archive was not found: $archivePath"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes first."
}
& $defectPython "scripts\run_visa_pcb_external_evaluation.py" `
    --project-root $projectPath `
    --archive $archivePath `
    --open-final `
    --mask-correction-v2
if ($LASTEXITCODE -ne 0) {
    throw "Corrected exploratory VisA evaluation stopped. The failed v1 output, source data and v2 sample cache were not deleted."
}
Get-FileHash -LiteralPath (Join-Path $projectPath "artifacts\handoff\visa-pcb-external-evaluation-v2.zip") -Algorithm SHA256
