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

function Assert-ExistingInput {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required existing seed-42 result was not found: $Path"
    }
}

function Get-RunState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    if (-not (Test-Path -LiteralPath $Path)) {
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
    $expectedStatus = if ($Model -eq "padim") { "fit_calibrated" } else { "fit_calibrated_exploratory" }
    $checkpointName = $record.artifacts.checkpoint
    $calibrationName = $record.artifacts.calibration_scores
    if ($checkpointName -isnot [string] -or $calibrationName -isnot [string]) {
        return "invalid"
    }
    $checkpointPath = Join-Path $Path $checkpointName
    $calibrationPath = Join-Path $Path $calibrationName
    if (
        $record.status -eq $expectedStatus -and
        $record.test_evaluated -eq $false -and
        $record.config.model -eq $Model -and
        $record.config.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq "9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585" -and
        (Test-Path -LiteralPath $checkpointPath -PathType Leaf) -and
        (Test-Path -LiteralPath $calibrationPath -PathType Leaf)
    ) {
        return "complete"
    }
    return "invalid"
}

function Get-EvaluationState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    $metadataPath = Join-Path $Path "evaluation.json"
    $metricsPath = Join-Path $Path "metrics.json"
    if (
        -not (Test-Path -LiteralPath $metadataPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $metricsPath -PathType Leaf)
    ) {
        return "invalid"
    }
    try {
        $record = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "invalid"
    }
    $expectedStatus = if ($Model -eq "padim") { "evaluated" } else { "evaluated_exploratory" }
    if (
        $record.status -eq $expectedStatus -and
        $record.test_evaluated -eq $true -and
        $record.fit.model_seed -eq $Seed -and
        $record.manifest_sha256 -eq "9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585" -and
        $record.counts.test -eq 100 -and
        $record.counts.failed -eq 0
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
    throw "Git working tree is not clean. Commit, stash, or remove local changes before training."
}

Assert-ExistingInput (Join-Path $projectPath "artifacts\evaluations\padim-transistor-seed42-eval-v1")
Assert-ExistingInput (Join-Path $projectPath "artifacts\evaluations\patchcore-transistor-seed42-eval-v1")

Invoke-CheckedPython -Arguments @(
    "scripts\check_ml_environment.py",
    "--output", "artifacts\environment-lab-gpu-multiseed.json"
)
Invoke-CheckedPython -Arguments @(
    "scripts\prepare_data.py", "verify",
    "--root", $datasetRoot,
    "--manifest", "data\manifests\transistor-v1.json"
)

foreach ($seed in @(43, 44)) {
    $padimRun = Join-Path $projectPath "artifacts\runs\padim-transistor-seed$seed"
    $padimEval = Join-Path $projectPath "artifacts\evaluations\padim-transistor-seed$seed-eval-v1"
    $runState = Get-RunState $padimRun "padim" $seed
    if ($runState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\train_padim.py",
            "--root", $datasetRoot,
            "--manifest", "data\manifests\transistor-v1.json",
            "--config", "configs\pilot.toml",
            "--seed", "$seed",
            "--output", $padimRun
        )
    } elseif ($runState -eq "complete") {
        Write-Host "Reusing completed PaDiM seed $seed fit: $padimRun"
    } else {
        throw "Existing PaDiM seed $seed fit is partial or invalid; it was not changed: $padimRun"
    }
    $evaluationState = Get-EvaluationState $padimEval "padim" $seed
    if ($evaluationState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\evaluate_padim.py",
            "--root", $datasetRoot,
            "--fit-run", $padimRun,
            "--manifest", "data\manifests\transistor-v1.json",
            "--evaluation-config", "configs\padim-evaluation-repeat-v1.toml",
            "--output", $padimEval
        )
    } elseif ($evaluationState -eq "complete") {
        Write-Host "Reusing completed PaDiM seed $seed evaluation: $padimEval"
    } else {
        throw "Existing PaDiM seed $seed evaluation is partial or invalid; it was not changed: $padimEval"
    }
}

foreach ($seed in @(43, 44)) {
    $patchcoreRun = Join-Path $projectPath "artifacts\runs\patchcore-transistor-seed$seed"
    $patchcoreEval = Join-Path $projectPath "artifacts\evaluations\patchcore-transistor-seed$seed-eval-v1"
    $runState = Get-RunState $patchcoreRun "patchcore" $seed
    if ($runState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\train_patchcore.py",
            "--root", $datasetRoot,
            "--manifest", "data\manifests\transistor-v1.json",
            "--config", "configs\patchcore-repeat-v1.toml",
            "--seed", "$seed",
            "--output", $patchcoreRun
        )
    } elseif ($runState -eq "complete") {
        Write-Host "Reusing completed PatchCore seed $seed fit: $patchcoreRun"
    } else {
        throw "Existing PatchCore seed $seed fit is partial or invalid; it was not changed: $patchcoreRun"
    }
    $evaluationState = Get-EvaluationState $patchcoreEval "patchcore" $seed
    if ($evaluationState -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\evaluate_patchcore.py",
            "--root", $datasetRoot,
            "--fit-run", $patchcoreRun,
            "--manifest", "data\manifests\transistor-v1.json",
            "--evaluation-config", "configs\patchcore-evaluation-repeat-v1.toml",
            "--output", $patchcoreEval
        )
    } elseif ($evaluationState -eq "complete") {
        Write-Host "Reusing completed PatchCore seed $seed evaluation: $patchcoreEval"
    } else {
        throw "Existing PatchCore seed $seed evaluation is partial or invalid; it was not changed: $patchcoreEval"
    }
}

$summaryOutput = Join-Path $projectPath "artifacts\summaries\transistor-padim-patchcore-3seed-v1"
$summaryJson = Join-Path $summaryOutput "summary.json"
if (-not (Test-Path -LiteralPath $summaryOutput)) {
    Invoke-CheckedPython -Arguments @(
        "scripts\summarize_multiseed.py",
        "--evaluation", "artifacts\evaluations\padim-transistor-seed42-eval-v1",
        "--evaluation", "artifacts\evaluations\padim-transistor-seed43-eval-v1",
        "--evaluation", "artifacts\evaluations\padim-transistor-seed44-eval-v1",
        "--evaluation", "artifacts\evaluations\patchcore-transistor-seed42-eval-v1",
        "--evaluation", "artifacts\evaluations\patchcore-transistor-seed43-eval-v1",
        "--evaluation", "artifacts\evaluations\patchcore-transistor-seed44-eval-v1",
        "--output", $summaryOutput
    )
} elseif (Test-Path -LiteralPath $summaryJson -PathType Leaf) {
    try {
        $summaryRecord = Get-Content -LiteralPath $summaryJson -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "Existing summary JSON is invalid and was not changed: $summaryJson"
    }
    if ($summaryRecord.status -ne "summarized_exploratory_multiseed") {
        throw "Existing summary is incomplete and was not changed: $summaryJson"
    }
    Write-Host "Reusing existing multi-seed summary: $summaryOutput"
} else {
    throw "Existing summary output is partial or invalid; it was not changed: $summaryOutput"
}

Write-Host "Multi-seed exploratory run completed."
Write-Host "Summary: $summaryOutput\summary.md"
