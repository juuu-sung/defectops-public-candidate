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

function Get-EvaluationState {
    param(
        [string]$Path,
        [int]$Seed,
        [string]$ManifestSha256,
        [int]$TestCount,
        [int]$NormalCount,
        [int]$AnomalyCount
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return "missing" }
    $evaluationPath = Join-Path $Path "evaluation.json"
    $metricsPath = Join-Path $Path "metrics.json"
    $fitPath = Join-Path $Path "fit-run.json"
    foreach ($required in @($evaluationPath, $metricsPath, $fitPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { return "invalid" }
    }
    try {
        $record = Get-Content -LiteralPath $evaluationPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $fit = Get-Content -LiteralPath $fitPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch { return "invalid" }
    $metricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $record.status -eq "evaluated_preregistered" -and
        $record.test_evaluated -eq $true -and
        $record.fit.model_seed -eq $Seed -and
        $fit.config.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq $ManifestSha256 -and
        $record.counts.test -eq $TestCount -and
        $record.counts.normal -eq $NormalCount -and
        $record.counts.anomaly -eq $AnomalyCount -and
        $record.counts.failed -eq 0 -and
        $record.artifacts.metrics_sha256 -eq $metricsSha256
    ) { return "complete" }
    return "invalid"
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before evaluation."
}
Invoke-CheckedPython -Arguments @("scripts\register_realiad.py", "verify", "--root", $datasetRoot)

$jobs = @(
    @{ Category = "realiad-transistor1"; Manifest = "data\manifests\realiad-transistor1-v1.json" },
    @{ Category = "realiad-plastic-nut"; Manifest = "data\manifests\realiad-plastic-nut-v1.json" }
)

foreach ($job in $jobs) {
    $manifest = Get-Content -LiteralPath $job.Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    $job["ManifestSha256"] = $manifest.manifest_sha256
    $job["TestCount"] = @($manifest.records | Where-Object { $_.split -eq "test" }).Count
    $job["NormalCount"] = @(
        $manifest.records | Where-Object { $_.split -eq "test" -and $_.label -eq 0 }
    ).Count
    $job["AnomalyCount"] = @(
        $manifest.records | Where-Object { $_.split -eq "test" -and $_.label -eq 1 }
    ).Count
    foreach ($seed in @(42, 43, 44)) {
        $runName = "patchcore-$($job.Category)-seed$seed"
        $runPath = Join-Path $projectPath "artifacts\runs\$runName"
        $runJson = Join-Path $runPath "run.json"
        if (-not (Test-Path -LiteralPath $runJson -PathType Leaf)) {
            throw "Required fit is missing: $runPath"
        }
        $run = Get-Content -LiteralPath $runJson -Raw -Encoding UTF8 | ConvertFrom-Json
        if (
            $run.status -ne "fit_calibrated_preregistered" -or
            $run.test_evaluated -ne $false -or
            $run.config.model_seed -ne $seed -or
            $run.manifest_sha256 -ne $manifest.manifest_sha256 -or
            -not (Test-Path -LiteralPath (Join-Path $runPath $run.artifacts.checkpoint) -PathType Leaf) -or
            -not (Test-Path -LiteralPath (Join-Path $runPath $run.artifacts.calibration_scores) -PathType Leaf)
        ) {
            throw "Required fit is incomplete or invalid: $runPath"
        }
    }
}

foreach ($job in $jobs) {
    foreach ($seed in @(42, 43, 44)) {
        $runName = "patchcore-$($job.Category)-seed$seed"
        $evaluationName = "$runName-eval-v1"
        $evaluationPath = Join-Path $projectPath "artifacts\evaluations\$evaluationName"
        $state = Get-EvaluationState `
            $evaluationPath $seed $job.ManifestSha256 `
            $job.TestCount $job.NormalCount $job.AnomalyCount
        if ($state -eq "missing") {
            Invoke-CheckedPython -Arguments @(
                "scripts\evaluate_patchcore.py",
                "--root", $datasetRoot,
                "--fit-run", "artifacts\runs\$runName",
                "--manifest", $job.Manifest,
                "--evaluation-config", "configs\realiad-patchcore-evaluation-v1.toml",
                "--output", $evaluationPath
            )
        } elseif ($state -eq "complete") {
            Write-Host "Reusing existing evaluation: $evaluationName"
        } else {
            throw "Existing evaluation is partial or invalid and was not changed: $evaluationPath"
        }
    }
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "realiad-patchcore-evaluations-v1"
$handoffZip = Join-Path $handoffDirectory "realiad-patchcore-evaluations-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffZip)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists: $handoffStage"
    }
    New-Item -ItemType Directory -Path $handoffStage | Out-Null
    foreach ($job in $jobs) {
        foreach ($seed in @(42, 43, 44)) {
            $name = "patchcore-$($job.Category)-seed$seed-eval-v1"
            $destination = Join-Path $handoffStage $name
            New-Item -ItemType Directory -Path $destination | Out-Null
            foreach ($file in @("evaluation.json", "metrics.json", "fit-run.json", "calibration-contract.json")) {
                Copy-Item -LiteralPath (Join-Path $projectPath "artifacts\evaluations\$name\$file") -Destination $destination
            }
        }
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffZip
}

Write-Host "Real-IAD frozen PatchCore evaluation completed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
