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

function Assert-MissingOutput {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        throw "Output already exists and will not be overwritten: $Path"
    }
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
    Assert-MissingOutput $padimRun
    Assert-MissingOutput $padimEval
    Invoke-CheckedPython -Arguments @(
        "scripts\train_padim.py",
        "--root", $datasetRoot,
        "--manifest", "data\manifests\transistor-v1.json",
        "--config", "configs\pilot.toml",
        "--seed", "$seed",
        "--output", $padimRun
    )
    Invoke-CheckedPython -Arguments @(
        "scripts\evaluate_padim.py",
        "--root", $datasetRoot,
        "--fit-run", $padimRun,
        "--manifest", "data\manifests\transistor-v1.json",
        "--evaluation-config", "configs\padim-evaluation-repeat-v1.toml",
        "--output", $padimEval
    )
}

foreach ($seed in @(43, 44)) {
    $patchcoreRun = Join-Path $projectPath "artifacts\runs\patchcore-transistor-seed$seed"
    $patchcoreEval = Join-Path $projectPath "artifacts\evaluations\patchcore-transistor-seed$seed-eval-v1"
    Assert-MissingOutput $patchcoreRun
    Assert-MissingOutput $patchcoreEval
    Invoke-CheckedPython -Arguments @(
        "scripts\train_patchcore.py",
        "--root", $datasetRoot,
        "--manifest", "data\manifests\transistor-v1.json",
        "--config", "configs\patchcore-repeat-v1.toml",
        "--seed", "$seed",
        "--output", $patchcoreRun
    )
    Invoke-CheckedPython -Arguments @(
        "scripts\evaluate_patchcore.py",
        "--root", $datasetRoot,
        "--fit-run", $patchcoreRun,
        "--manifest", "data\manifests\transistor-v1.json",
        "--evaluation-config", "configs\patchcore-evaluation-repeat-v1.toml",
        "--output", $patchcoreEval
    )
}

$summaryOutput = Join-Path $projectPath "artifacts\summaries\transistor-padim-patchcore-3seed-v1"
Assert-MissingOutput $summaryOutput
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

Write-Host "Multi-seed exploratory run completed."
Write-Host "Summary: $summaryOutput\summary.md"
