[CmdletBinding()]
param(
    [string]$DownloadDirectory = (Join-Path $env:USERPROFILE "Downloads")
)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$datasetRoot = Join-Path $projectPath "data"
$jsonsZip = Join-Path $DownloadDirectory "realiad_jsons.zip"
$pcbZip = Join-Path $DownloadDirectory "pcb.zip"
$terminalblockZip = Join-Path $DownloadDirectory "terminalblock.zip"
$planPath = Join-Path $projectPath "data\manifests\realiad-followup-split-plan-v1.json"

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
foreach ($path in @($jsonsZip, $pcbZip, $terminalblockZip, $planPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Real-IAD follow-up input was not found: $path"
    }
}

Invoke-CheckedPython -Arguments @(
    "scripts\plan_realiad_followup.py",
    "--jsons-zip", $jsonsZip,
    "--config", "configs\realiad-followup-preregistration-v1.toml",
    "--output", $planPath
)

$registeredRoot = Join-Path $datasetRoot "real_iad_followup"
if (-not (Test-Path -LiteralPath $registeredRoot)) {
    Invoke-CheckedPython -Arguments @(
        "scripts\register_realiad_followup.py", "register",
        "--root", $datasetRoot,
        "--plan", $planPath,
        "--pcb-zip", $pcbZip,
        "--terminalblock-zip", $terminalblockZip
    )
}

Invoke-CheckedPython -Arguments @(
    "scripts\register_realiad_followup.py", "verify",
    "--root", $datasetRoot,
    "--plan", $planPath
)

$auditJobs = @(
    @{ Name = "realiad-pcb-development-v1"; Manifest = "data\manifests\realiad-pcb-development-v1.json" },
    @{ Name = "realiad-terminalblock-development-v1"; Manifest = "data\manifests\realiad-terminalblock-development-v1.json" }
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
            throw "Existing follow-up audit is partial: $auditPath"
        }
        $audit = Get-Content -LiteralPath $auditJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $manifest = Get-Content -LiteralPath $job.Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($audit.status -ne "passed" -or $audit.manifest_sha256 -ne $manifest.manifest_sha256) {
            throw "Existing follow-up audit is invalid: $auditPath"
        }
    }
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "realiad-followup-registration-v1"
$handoffPath = Join-Path $handoffDirectory "realiad-followup-registration-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffPath)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists: $handoffStage"
    }
    $manifestStage = Join-Path $handoffStage "manifests"
    $auditStage = Join-Path $handoffStage "audits"
    New-Item -ItemType Directory -Path $manifestStage -Force | Out-Null
    New-Item -ItemType Directory -Path $auditStage -Force | Out-Null
    $manifestNames = @(
        "realiad-followup-split-plan-v1.json",
        "realiad-pcb-development-v1.json",
        "realiad-terminalblock-development-v1.json",
        "realiad-pcb-final-sealed-v1.json",
        "realiad-terminalblock-final-sealed-v1.json"
    )
    foreach ($name in $manifestNames) {
        Copy-Item -LiteralPath (Join-Path $datasetRoot "manifests\$name") -Destination $manifestStage
    }
    foreach ($job in $auditJobs) {
        $destination = Join-Path $auditStage $job.Name
        New-Item -ItemType Directory -Path $destination | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectPath "artifacts\data-audit\$($job.Name)\audit.json") `
            -Destination $destination
        Copy-Item -LiteralPath (Join-Path $projectPath "artifacts\data-audit\$($job.Name)\summary.md") `
            -Destination $destination
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffPath
}

Write-Host "Real-IAD follow-up development registration completed and verified."
Write-Host "Final holdout members were checksummed but were not extracted, decoded, previewed, or scored."
Write-Host "Attach this file to the Codex task: $handoffPath"
Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256

