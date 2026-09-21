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
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][string]$ManifestSha256
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return "missing"
    }
    $metadataPath = Join-Path $Path "run.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        return "invalid"
    }
    try {
        $record = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "invalid"
    }
    $expectedStatus = if ($Model -eq "padim") {
        "fit_calibrated"
    } else {
        "fit_calibrated_preregistered"
    }
    $checkpointName = $record.artifacts.checkpoint
    $calibrationName = $record.artifacts.calibration_scores
    if ($checkpointName -isnot [string] -or $calibrationName -isnot [string]) {
        return "invalid"
    }
    if (
        $record.status -eq $expectedStatus -and
        $record.test_evaluated -eq $false -and
        $record.config.model -eq $Model -and
        $record.config.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq $ManifestSha256 -and
        (Test-Path -LiteralPath (Join-Path $Path $checkpointName) -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path $calibrationName) -PathType Leaf)
    ) {
        return "complete"
    }
    return "invalid"
}

function Get-EvaluationState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][string]$ManifestSha256,
        [Parameter(Mandatory = $true)][int]$TestCount,
        [Parameter(Mandatory = $true)][int]$NormalCount,
        [Parameter(Mandatory = $true)][int]$AnomalyCount
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    $evaluationPath = Join-Path $Path "evaluation.json"
    $metricsPath = Join-Path $Path "metrics.json"
    $fitPath = Join-Path $Path "fit-run.json"
    if (
        -not (Test-Path -LiteralPath $evaluationPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $metricsPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $fitPath -PathType Leaf)
    ) {
        return "invalid"
    }
    try {
        $record = Get-Content -LiteralPath $evaluationPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $fit = Get-Content -LiteralPath $fitPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "invalid"
    }
    $expectedStatus = if ($Model -eq "padim") { "evaluated" } else { "evaluated_preregistered" }
    $actualMetricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $record.status -eq $expectedStatus -and
        $record.test_evaluated -eq $true -and
        $record.fit.model_seed -eq $Seed -and
        $fit.config.model -eq $Model -and
        $fit.config.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq $ManifestSha256 -and
        $record.counts.test -eq $TestCount -and
        $record.counts.normal -eq $NormalCount -and
        $record.counts.anomaly -eq $AnomalyCount -and
        $record.counts.failed -eq 0 -and
        $record.artifacts.metrics_sha256 -eq $actualMetricsSha256
    ) {
        return "complete"
    }
    return "invalid"
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before evaluation."
}

$jobs = @(
    @{
        Category = "cable"
        Manifest = "data\manifests\cable-v1.json"
        ManifestSha256 = "f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70"
        TestCount = 150
        NormalCount = 58
        AnomalyCount = 92
    },
    @{
        Category = "metal-nut"
        Manifest = "data\manifests\metal-nut-v1.json"
        ManifestSha256 = "6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5"
        TestCount = 115
        NormalCount = 22
        AnomalyCount = 93
    }
)

Invoke-CheckedPython -Arguments @(
    "scripts\check_ml_environment.py",
    "--output", "artifacts\environment-lab-gpu-three-category-evaluation-v1.json"
)

foreach ($job in $jobs) {
    Invoke-CheckedPython -Arguments @(
        "scripts\prepare_data.py", "verify",
        "--root", $datasetRoot,
        "--manifest", $job.Manifest
    )
    foreach ($model in @("padim", "patchcore")) {
        foreach ($seed in @(42, 43, 44)) {
            $runPath = Join-Path $projectPath "artifacts\runs\$model-$($job.Category)-seed$seed"
            if ((Get-FitState $runPath $model $seed $job.ManifestSha256) -ne "complete") {
                throw "Required completed fit is missing or invalid: $runPath"
            }
        }
    }
}

# The model, preprocessing, thresholds, metrics, and selection rule were frozen before this loop.
foreach ($job in $jobs) {
    foreach ($model in @("padim", "patchcore")) {
        $evaluator = if ($model -eq "padim") {
            "scripts\evaluate_padim.py"
        } else {
            "scripts\evaluate_patchcore.py"
        }
        $evaluationConfig = if ($model -eq "padim") {
            "configs\additional-padim-evaluation-v1.toml"
        } else {
            "configs\additional-patchcore-evaluation-v1.toml"
        }
        foreach ($seed in @(42, 43, 44)) {
            $runName = "$model-$($job.Category)-seed$seed"
            $runPath = Join-Path $projectPath "artifacts\runs\$runName"
            $evaluationName = "$runName-eval-v1"
            $evaluationPath = Join-Path $projectPath "artifacts\evaluations\$evaluationName"
            $state = Get-EvaluationState `
                $evaluationPath $model $seed $job.ManifestSha256 `
                $job.TestCount $job.NormalCount $job.AnomalyCount
            if ($state -eq "missing") {
                Invoke-CheckedPython -Arguments @(
                    $evaluator,
                    "--root", $datasetRoot,
                    "--fit-run", $runPath,
                    "--manifest", $job.Manifest,
                    "--evaluation-config", $evaluationConfig,
                    "--output", $evaluationPath
                )
            } elseif ($state -eq "complete") {
                Write-Host "Reusing completed evaluation: $evaluationName"
            } else {
                throw (
                    "Existing evaluation is partial or invalid and was not changed: $evaluationPath`n" +
                    "Keep it for diagnosis, rename it, then run this script again."
                )
            }
        }
    }
}

$summaryPath = Join-Path $projectPath "artifacts\summaries\three-category-padim-patchcore-v1"
$summaryJson = Join-Path $summaryPath "summary.json"
if (-not (Test-Path -LiteralPath $summaryPath)) {
    $summaryArguments = @("scripts\summarize_three_category.py")
    foreach ($category in @("transistor", "cable", "metal-nut")) {
        foreach ($model in @("padim", "patchcore")) {
            foreach ($seed in @(42, 43, 44)) {
                $summaryArguments += @(
                    "--evaluation",
                    "artifacts\evaluations\$model-$category-seed$seed-eval-v1"
                )
            }
        }
    }
    $summaryArguments += @("--output", $summaryPath)
    Invoke-CheckedPython -Arguments $summaryArguments
} elseif (Test-Path -LiteralPath $summaryJson -PathType Leaf) {
    try {
        $summary = Get-Content -LiteralPath $summaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "Existing summary JSON is invalid and was not changed: $summaryJson"
    }
    if ($summary.status -ne "summarized_preregistered_three_category_mixed_evidence") {
        throw "Existing summary is incomplete and was not changed: $summaryJson"
    }
    Write-Host "Reusing completed three-category summary: $summaryPath"
} else {
    throw "Existing summary output is partial or invalid and was not changed: $summaryPath"
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "three-category-evaluation-v1"
$handoffZip = Join-Path $handoffDirectory "three-category-evaluation-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffZip)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists and was not changed: $handoffStage"
    }
    New-Item -ItemType Directory -Path $handoffStage | Out-Null
    Copy-Item -LiteralPath $summaryPath -Destination (Join-Path $handoffStage "summary") -Recurse
    Copy-Item -LiteralPath (
        Join-Path $projectPath "artifacts\environment-lab-gpu-three-category-evaluation-v1.json"
    ) -Destination $handoffStage
    foreach ($job in $jobs) {
        foreach ($model in @("padim", "patchcore")) {
            foreach ($seed in @(42, 43, 44)) {
                $evaluationName = "$model-$($job.Category)-seed$seed-eval-v1"
                $source = Join-Path $projectPath "artifacts\evaluations\$evaluationName"
                $destination = Join-Path $handoffStage $evaluationName
                New-Item -ItemType Directory -Path $destination | Out-Null
                foreach ($name in @(
                    "evaluation.json", "metrics.json", "fit-run.json", "calibration-contract.json"
                )) {
                    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $destination
                }
            }
        }
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffZip
}

Write-Host "Frozen additional-category evaluation phase completed."
Write-Host "Completed 12 evaluations and the 18-run three-category summary."
Write-Host "No post-test model, threshold, ROI, or metric tuning was performed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
