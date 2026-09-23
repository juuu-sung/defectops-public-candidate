[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes first."
}
& $defectPython "scripts\run_realiad_development_score_probe.py" `
    --project-root $projectPath `
    --output (Join-Path $projectPath "artifacts\realiad-development-score-probe-v1") `
    --handoff (Join-Path $projectPath "artifacts\handoff\realiad-development-score-probe-v1.zip")
if ($LASTEXITCODE -ne 0) {
    throw "Exploratory development score probe failed; existing samples were not deleted."
}
Get-FileHash -LiteralPath (Join-Path $projectPath "artifacts\handoff\realiad-development-score-probe-v1.zip") -Algorithm SHA256
