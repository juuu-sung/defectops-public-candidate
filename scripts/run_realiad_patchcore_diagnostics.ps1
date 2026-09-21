[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$datasetRoot = Join-Path $projectPath "data"
$outputPath = Join-Path $projectPath "artifacts\analyses\realiad-patchcore-multiseed-v1"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $defectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before diagnostics."
}
Invoke-CheckedPython -Arguments @("scripts\register_realiad.py", "verify", "--root", $datasetRoot)

if (-not (Test-Path -LiteralPath $outputPath)) {
    Invoke-CheckedPython -Arguments @(
        "scripts\analyze_realiad_multiseed.py",
        "--root", $datasetRoot,
        "--evaluations-root", "artifacts\evaluations",
        "--output", $outputPath
    )
} else {
    $analysisPath = Join-Path $outputPath "analysis.json"
    if (-not (Test-Path -LiteralPath $analysisPath -PathType Leaf)) {
        throw "Existing Real-IAD diagnostics are partial: $outputPath"
    }
    $analysis = Get-Content -LiteralPath $analysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $analysis.status -ne "analyzed_multiseed_diagnostics" -or
        $analysis.model_changed -ne $false -or
        $analysis.thresholds_changed -ne $false
    ) {
        throw "Existing Real-IAD diagnostics are invalid: $outputPath"
    }
    Write-Host "Reusing completed Real-IAD multiseed diagnostics."
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffZip = Join-Path $handoffDirectory "realiad-patchcore-multiseed-diagnostics-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffZip)) {
    Compress-Archive -Path (Join-Path $outputPath "*") -DestinationPath $handoffZip
}

Write-Host "Real-IAD multiseed diagnostics completed without changing models or thresholds."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
