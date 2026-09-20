[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$datasetRoot = Join-Path $projectPath "data"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $defectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Assert-CategoryLayout {
    param([Parameter(Mandatory = $true)][string]$Category)
    foreach ($relativePath in @("train\good", "test\good", "ground_truth")) {
        $path = Join-Path $datasetRoot "$Category\$relativePath"
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Missing extracted MVTec directory: $path"
        }
    }
}

function Assert-ArchiveChecksumIfPresent {
    param(
        [Parameter(Mandatory = $true)][string]$ArchiveName,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    $archivePath = Join-Path $datasetRoot $ArchiveName
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        Write-Host "Archive not retained; extracted files will still be checksummed: $archivePath"
        return
    }
    $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256) {
        throw "Archive SHA-256 mismatch for $ArchiveName`: $actual"
    }
    Write-Host "Archive checksum verified: $ArchiveName"
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}

Assert-CategoryLayout "cable"
Assert-CategoryLayout "metal_nut"
Assert-ArchiveChecksumIfPresent `
    "cable.tar.xz" `
    "1e3d59b80a6a8da5adf32fb0cf311c7b964cde146d6920aef9ca66380999a9cd"
Assert-ArchiveChecksumIfPresent `
    "metal_nut.tar.xz" `
    "86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a"

$jobs = @(
    @{
        Category = "cable"
        Config = "configs\cable-padim-v1.toml"
        Manifest = "data\manifests\cable-v1.json"
        Audit = "artifacts\data-audit\cable-v1"
    },
    @{
        Category = "metal_nut"
        Config = "configs\metal-nut-padim-v1.toml"
        Manifest = "data\manifests\metal-nut-v1.json"
        Audit = "artifacts\data-audit\metal-nut-v1"
    }
)

foreach ($job in $jobs) {
    $manifestPath = Join-Path $projectPath $job.Manifest
    $auditPath = Join-Path $projectPath $job.Audit
    if (Test-Path -LiteralPath $manifestPath) {
        throw "Manifest already exists and was not overwritten: $manifestPath"
    }
    if (Test-Path -LiteralPath $auditPath) {
        throw "Audit output already exists and was not overwritten: $auditPath"
    }
    Invoke-CheckedPython -Arguments @(
        "scripts\prepare_data.py", "create",
        "--root", $datasetRoot,
        "--config", $job.Config,
        "--output", $job.Manifest
    )
    Invoke-CheckedPython -Arguments @(
        "scripts\prepare_data.py", "verify",
        "--root", $datasetRoot,
        "--manifest", $job.Manifest
    )
    Invoke-CheckedPython -Arguments @(
        "scripts\audit_data.py",
        "--root", $datasetRoot,
        "--manifest", $job.Manifest,
        "--output", $job.Audit
    )
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
$handoffPath = Join-Path $handoffDirectory "additional-category-data-registration-v1.zip"
if (Test-Path -LiteralPath $handoffPath) {
    throw "Handoff archive already exists and was not overwritten: $handoffPath"
}
Compress-Archive -LiteralPath @(
    (Join-Path $projectPath "data\manifests\cable-v1.json"),
    (Join-Path $projectPath "data\manifests\metal-nut-v1.json"),
    (Join-Path $projectPath "artifacts\data-audit\cable-v1\summary.md"),
    (Join-Path $projectPath "artifacts\data-audit\metal-nut-v1\summary.md")
) -DestinationPath $handoffPath

Write-Host "Additional-category data registration completed."
Write-Host "Do not open or tune on the test images."
Write-Host "Attach this file to the Codex task: $handoffPath"
Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256
