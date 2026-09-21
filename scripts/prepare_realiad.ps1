[CmdletBinding()]
param(
    [string]$DownloadDirectory = (Join-Path $env:USERPROFILE "Downloads")
)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$datasetRoot = Join-Path $projectPath "data"
$jsonsZip = Join-Path $DownloadDirectory "realiad_jsons.zip"
$transistorZip = Join-Path $DownloadDirectory "transistor1.zip"
$plasticNutZip = Join-Path $DownloadDirectory "plastic_nut.zip"

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
foreach ($path in @($jsonsZip, $transistorZip, $plasticNutZip)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Real-IAD archive was not found: $path"
    }
}

$registeredRoot = Join-Path $datasetRoot "real_iad"
if (-not (Test-Path -LiteralPath $registeredRoot)) {
    Invoke-CheckedPython -Arguments @(
        "scripts\register_realiad.py", "register",
        "--root", $datasetRoot,
        "--jsons-zip", $jsonsZip,
        "--transistor-zip", $transistorZip,
        "--plastic-nut-zip", $plasticNutZip,
        "--seed", "42",
        "--calibration-fraction", "0.20"
    )
}

Invoke-CheckedPython -Arguments @(
    "scripts\register_realiad.py", "verify",
    "--root", $datasetRoot
)

$auditJobs = @(
    @{ Name = "realiad-transistor1-v1"; Manifest = "data\manifests\realiad-transistor1-v1.json" },
    @{ Name = "realiad-plastic-nut-v1"; Manifest = "data\manifests\realiad-plastic-nut-v1.json" }
)
foreach ($job in $auditJobs) {
    $auditPath = Join-Path $projectPath "artifacts\data-audit\$($job.Name)"
    if (-not (Test-Path -LiteralPath $auditPath)) {
        Invoke-CheckedPython -Arguments @(
            "scripts\audit_data.py",
            "--root", $datasetRoot,
            "--manifest", $job.Manifest,
            "--output", $auditPath
        )
    } else {
        $auditJson = Join-Path $auditPath "audit.json"
        if (-not (Test-Path -LiteralPath $auditJson -PathType Leaf)) {
            throw "Existing Real-IAD audit is partial: $auditPath"
        }
        $audit = Get-Content -LiteralPath $auditJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $manifest = Get-Content -LiteralPath $job.Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($audit.status -ne "passed" -or $audit.manifest_sha256 -ne $manifest.manifest_sha256) {
            throw "Existing Real-IAD audit is invalid: $auditPath"
        }
    }
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "realiad-data-registration-v1"
$handoffPath = Join-Path $handoffDirectory "realiad-data-registration-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffPath)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists: $handoffStage"
    }
    $manifestStage = Join-Path $handoffStage "manifests"
    $auditStage = Join-Path $handoffStage "audits"
    New-Item -ItemType Directory -Path $manifestStage -Force | Out-Null
    New-Item -ItemType Directory -Path $auditStage -Force | Out-Null
    foreach ($job in $auditJobs) {
        Copy-Item -LiteralPath (Join-Path $projectPath $job.Manifest) -Destination $manifestStage
        $destination = Join-Path $auditStage $job.Name
        New-Item -ItemType Directory -Path $destination | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectPath "artifacts\data-audit\$($job.Name)\audit.json") `
            -Destination $destination
        Copy-Item -LiteralPath (Join-Path $projectPath "artifacts\data-audit\$($job.Name)\summary.md") `
            -Destination $destination
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffPath
}

Write-Host "Real-IAD single-view registration completed and verified."
Write-Host "The official test files were decoded only for integrity; they were not previewed or scored."
Write-Host "Attach this file to the Codex task: $handoffPath"
Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256
