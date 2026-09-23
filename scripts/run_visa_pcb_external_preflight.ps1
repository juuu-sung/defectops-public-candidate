[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
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
& $defectPython "scripts\run_visa_pcb_external_preflight.py" `
    --project-root $projectPath `
    --archive $archivePath
if ($LASTEXITCODE -ne 0) {
    throw "VisA calibration-only preflight failed. Existing data and outputs were not deleted."
}
Get-FileHash -LiteralPath (Join-Path $projectPath "artifacts\handoff\visa-pcb-external-preflight-v1.zip") -Algorithm SHA256
