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

function Get-AnalysisState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$EvaluationId
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    $metadataPath = Join-Path $Path "analysis.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        return "invalid"
    }
    try {
        $record = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "invalid"
    }
    if (
        $record.status -eq "analyzed_exploratory" -and
        $record.model -eq "patchcore" -and
        $record.model_changed -eq $false -and
        $record.threshold_changed -eq $false -and
        $record.trace.evaluation_id -eq $EvaluationId
    ) {
        return "complete"
    }
    return "invalid"
}

function Get-BenchmarkState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$EvaluationId
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    $metadataPath = Join-Path $Path "benchmark.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        return "invalid"
    }
    try {
        $record = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "invalid"
    }
    if (
        $record.status -eq "benchmarked_warm" -and
        $record.model -eq "patchcore" -and
        $record.warmup_invocations -eq 5 -and
        $record.repeats -eq 1 -and
        $record.unique_image_count -eq 100 -and
        $record.measured_invocations -eq 100 -and
        $record.trace.evaluation_id -eq $EvaluationId -and
        $record.trace.thresholds_changed -eq $false -and
        $record.trace.labels_used_for_timing -eq $false
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
    throw "Git working tree is not clean. Pull and resolve local changes before diagnostics."
}

$jobs = @(
    @{ Category = "cable" },
    @{ Category = "metal-nut" }
)

foreach ($job in $jobs) {
    $runName = "patchcore-$($job.Category)-seed42"
    $evaluationName = "$runName-eval-v1"
    $fitPath = Join-Path $projectPath "artifacts\runs\$runName"
    $evaluationPath = Join-Path $projectPath "artifacts\evaluations\$evaluationName"
    if (-not (Test-Path -LiteralPath (Join-Path $fitPath "run.json") -PathType Leaf)) {
        throw "Required seed-42 fit is missing: $fitPath"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $evaluationPath "evaluation.json") -PathType Leaf)) {
        throw "Required seed-42 evaluation is missing: $evaluationPath"
    }

    $analysisName = "$runName-errors-v1"
    $analysisPath = Join-Path $projectPath "artifacts\analyses\$analysisName"
    $analysisState = Get-AnalysisState $analysisPath $evaluationName
    if ($analysisState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\analyze_patchcore_errors.py",
            "--root", $datasetRoot,
            "--fit-run", $fitPath,
            "--evaluation", $evaluationPath,
            "--output", $analysisPath
        )
    } elseif ($analysisState -eq "complete") {
        Write-Host "Reusing completed error analysis: $analysisName"
    } else {
        throw "Existing error analysis is partial or invalid and was not changed: $analysisPath"
    }

    $benchmarkName = "$runName-warm-v1"
    $benchmarkPath = Join-Path $projectPath "artifacts\benchmarks\$benchmarkName"
    $benchmarkState = Get-BenchmarkState $benchmarkPath $evaluationName
    if ($benchmarkState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\benchmark_patchcore_warm.py",
            "--root", $datasetRoot,
            "--fit-run", $fitPath,
            "--evaluation", $evaluationPath,
            "--warmup", "5",
            "--repeats", "1",
            "--count", "100",
            "--output", $benchmarkPath
        )
    } elseif ($benchmarkState -eq "complete") {
        Write-Host "Reusing completed warm benchmark: $benchmarkName"
    } else {
        throw "Existing benchmark is partial or invalid and was not changed: $benchmarkPath"
    }
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "patchcore-additional-diagnostics-v1"
$handoffZip = Join-Path $handoffDirectory "patchcore-additional-diagnostics-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffZip)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists and was not changed: $handoffStage"
    }
    New-Item -ItemType Directory -Path $handoffStage | Out-Null
    foreach ($job in $jobs) {
        $runName = "patchcore-$($job.Category)-seed42"
        Copy-Item -LiteralPath (
            Join-Path $projectPath "artifacts\analyses\$runName-errors-v1"
        ) -Destination (Join-Path $handoffStage "$runName-errors-v1") -Recurse
        Copy-Item -LiteralPath (
            Join-Path $projectPath "artifacts\benchmarks\$runName-warm-v1"
        ) -Destination (Join-Path $handoffStage "$runName-warm-v1") -Recurse
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffZip
}

Write-Host "PatchCore additional-category diagnostics completed."
Write-Host "No model, threshold, ROI, or postprocessing setting was changed."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
