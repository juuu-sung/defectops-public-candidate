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

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before training."
}

$jobs = @(
    @{
        Category = "cable"
        Manifest = "data\manifests\cable-v1.json"
        ManifestSha256 = "f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70"
        PadimConfig = "configs\cable-padim-v1.toml"
        PatchcoreConfig = "configs\cable-patchcore-v1.toml"
    },
    @{
        Category = "metal-nut"
        Manifest = "data\manifests\metal-nut-v1.json"
        ManifestSha256 = "6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5"
        PadimConfig = "configs\metal-nut-padim-v1.toml"
        PatchcoreConfig = "configs\metal-nut-patchcore-v1.toml"
    }
)

Invoke-CheckedPython -Arguments @(
    "scripts\check_ml_environment.py",
    "--output", "artifacts\environment-lab-gpu-additional-v1.json"
)

foreach ($job in $jobs) {
    Invoke-CheckedPython -Arguments @(
        "scripts\prepare_data.py", "verify",
        "--root", $datasetRoot,
        "--manifest", $job.Manifest
    )
}

# All 12 fits finish before any command is allowed to open an official test image.
foreach ($job in $jobs) {
    foreach ($model in @("padim", "patchcore")) {
        $config = if ($model -eq "padim") { $job.PadimConfig } else { $job.PatchcoreConfig }
        $trainer = if ($model -eq "padim") {
            "scripts\train_padim.py"
        } else {
            "scripts\train_patchcore.py"
        }
        foreach ($seed in @(42, 43, 44)) {
            $runName = "$model-$($job.Category)-seed$seed"
            $runPath = Join-Path $projectPath "artifacts\runs\$runName"
            $state = Get-FitState $runPath $model $seed $job.ManifestSha256
            if ($state -eq "missing") {
                Invoke-CheckedPython -Arguments @(
                    $trainer,
                    "--root", $datasetRoot,
                    "--manifest", $job.Manifest,
                    "--config", $config,
                    "--seed", "$seed",
                    "--output", $runPath
                )
            } elseif ($state -eq "complete") {
                Write-Host "Reusing completed fit: $runName"
            } else {
                throw (
                    "Existing fit is partial or invalid and was not changed: $runPath`n" +
                    "Keep it for diagnosis, rename it, then run this script again."
                )
            }
        }
    }
}

Write-Host "Additional-category fit/calibration phase completed."
Write-Host "Completed 12 fits: 2 categories x 2 models x 3 seeds."
Write-Host "No official cable or metal_nut test image was evaluated by this script."
Write-Host "Next: return the final console output before running the frozen evaluation phase."
