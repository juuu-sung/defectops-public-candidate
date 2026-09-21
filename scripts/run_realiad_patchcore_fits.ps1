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

function Get-FitState {
    param(
        [string]$Path,
        [int]$Seed,
        [string]$ManifestSha256
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return "missing" }
    $metadataPath = Join-Path $Path "run.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) { return "invalid" }
    try {
        $record = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch { return "invalid" }
    if (
        $record.status -eq "fit_calibrated_preregistered" -and
        $record.test_evaluated -eq $false -and
        $record.config.model -eq "patchcore" -and
        $record.config.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq $ManifestSha256 -and
        (Test-Path -LiteralPath (Join-Path $Path $record.artifacts.checkpoint) -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path $record.artifacts.calibration_scores) -PathType Leaf)
    ) { return "complete" }
    return "invalid"
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before training."
}

Invoke-CheckedPython -Arguments @("scripts\register_realiad.py", "verify", "--root", $datasetRoot)

$jobs = @(
    @{
        Category = "realiad-transistor1"
        Manifest = "data\manifests\realiad-transistor1-v1.json"
        Config = "configs\realiad-transistor1-patchcore-v1.toml"
    },
    @{
        Category = "realiad-plastic-nut"
        Manifest = "data\manifests\realiad-plastic-nut-v1.json"
        Config = "configs\realiad-plastic-nut-patchcore-v1.toml"
    }
)

foreach ($job in $jobs) {
    $manifest = Get-Content -LiteralPath $job.Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    $job["ManifestSha256"] = $manifest.manifest_sha256
}

# All six fits finish before the separate evaluation script may open test images.
foreach ($job in $jobs) {
    foreach ($seed in @(42, 43, 44)) {
        $runName = "patchcore-$($job.Category)-seed$seed"
        $runPath = Join-Path $projectPath "artifacts\runs\$runName"
        $state = Get-FitState $runPath $seed $job.ManifestSha256
        if ($state -eq "missing") {
            Invoke-CheckedPython -Arguments @(
                "scripts\train_patchcore.py",
                "--root", $datasetRoot,
                "--manifest", $job.Manifest,
                "--config", $job.Config,
                "--seed", "$seed",
                "--output", $runPath
            )
        } elseif ($state -eq "complete") {
            Write-Host "Reusing completed fit: $runName"
        } else {
            throw "Existing fit is partial or invalid and was not changed: $runPath"
        }
    }
}

Write-Host "Real-IAD PatchCore fit/calibration phase completed."
Write-Host "Completed six fits: two categories x three seeds. No test image was evaluated."
