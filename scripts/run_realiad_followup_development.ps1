[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$datasetRoot = Join-Path $projectPath "data"
$artifactRoot = Join-Path $projectPath "artifacts\realiad-followup-development-v1"
$selectionPath = Join-Path $artifactRoot "selection"
$planSha256 = "c68562288b378d44d969c3eda68a69a00c6c31e8dad0f4ff21e5c05073ac8f20"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $defectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Test-FileHash {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected
}

function Get-CategoryState {
    param([string]$Path, [string]$Category, [string]$ManifestSha256, [string]$GitCommit)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return "missing" }
    $resultPath = Join-Path $Path "category-result.json"
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) { return "invalid" }
    try {
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch { return "invalid" }
    if (
        $result.status -ne "evaluated_preregistered_development" -or
        $result.category -ne $Category -or
        $result.development_evaluated -ne $true -or
        $result.final_holdout_evaluated -ne $false -or
        $result.plan_sha256 -ne $planSha256 -or
        $result.git.commit -ne $GitCommit -or
        $result.development_manifest_sha256 -ne $ManifestSha256 -or
        @($result.candidates.PSObject.Properties).Count -ne 4
    ) { return "invalid" }
    foreach ($layer in @($result.layer_groups.PSObject.Properties.Value)) {
        if (-not (Test-FileHash (Join-Path $Path $layer.checkpoint) $layer.checkpoint_sha256)) {
            return "invalid"
        }
    }
    foreach ($candidate in @($result.candidates.PSObject.Properties.Value)) {
        if (
            -not (Test-FileHash (Join-Path $Path $candidate.artifacts.calibration_scores) $candidate.artifacts.calibration_scores_sha256) -or
            -not (Test-FileHash (Join-Path $Path $candidate.artifacts.predictions) $candidate.artifacts.predictions_sha256)
        ) { return "invalid" }
    }
    return "complete"
}

function Get-SelectionState {
    param([string]$Path, [string]$GitCommit)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return "missing" }
    $reportPath = Join-Path $Path "selection-report.json"
    $summaryPath = Join-Path $Path "summary.md"
    if (
        -not (Test-Path -LiteralPath $reportPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $summaryPath -PathType Leaf)
    ) { return "invalid" }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch { return "invalid" }
    if (
        $report.status -notin @("candidate_selected_final_still_sealed", "no_eligible_candidate") -or
        $report.git_commit -ne $GitCommit -or
        $report.plan_sha256 -ne $planSha256 -or
        $report.final_holdout_evaluated -ne $false -or
        @($report.candidate_results).Count -ne 4
    ) { return "invalid" }
    return "complete"
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes before development scoring."
}
$gitCommit = git rev-parse HEAD

Invoke-CheckedPython -Arguments @(
    "scripts\register_realiad_followup.py", "verify",
    "--root", $datasetRoot,
    "--plan", "data\manifests\realiad-followup-split-plan-v1.json"
)

$jobs = @(
    @{
        Category = "realiad_pcb"
        Name = "pcb"
        Manifest = "data\manifests\realiad-pcb-development-v1.json"
        FinalManifest = "data\manifests\realiad-pcb-final-sealed-v1.json"
        ManifestSha256 = "2d16e21b2e665defbf54266c1d5e8b08f41838dfae679fff8f65dff615b49ed4"
    },
    @{
        Category = "realiad_terminalblock"
        Name = "terminalblock"
        Manifest = "data\manifests\realiad-terminalblock-development-v1.json"
        FinalManifest = "data\manifests\realiad-terminalblock-final-sealed-v1.json"
        ManifestSha256 = "a5f6e45935e2a16f64a23f4757fa383f569efec84b25e7ed3ddf197456c2b338"
    }
)

foreach ($job in $jobs) {
    $outputPath = Join-Path $artifactRoot $job.Name
    $state = Get-CategoryState $outputPath $job.Category $job.ManifestSha256 $gitCommit
    if ($state -eq "missing") {
        Invoke-CheckedPython -Arguments @(
            "scripts\run_realiad_followup_development.py", "category",
            "--root", $datasetRoot,
            "--manifest", $job.Manifest,
            "--final-manifest", $job.FinalManifest,
            "--category", $job.Category,
            "--output", $outputPath
        )
    } elseif ($state -eq "complete") {
        Write-Host "Reusing completed development result: $($job.Category)"
    } else {
        throw "Existing development result is partial or invalid and was not changed: $outputPath"
    }
}

$selectionState = Get-SelectionState $selectionPath $gitCommit
if ($selectionState -eq "missing") {
    Invoke-CheckedPython -Arguments @(
        "scripts\run_realiad_followup_development.py", "select",
        "--pcb-result", (Join-Path $artifactRoot "pcb\category-result.json"),
        "--terminalblock-result", (Join-Path $artifactRoot "terminalblock\category-result.json"),
        "--output", $selectionPath
    )
} elseif ($selectionState -eq "complete") {
    Write-Host "Reusing completed candidate selection."
} else {
    throw "Existing selection result is partial or invalid and was not changed: $selectionPath"
}

$handoffDirectory = Join-Path $projectPath "artifacts\handoff"
$handoffStage = Join-Path $handoffDirectory "realiad-followup-development-v1"
$handoffZip = Join-Path $handoffDirectory "realiad-followup-development-v1.zip"
New-Item -ItemType Directory -Path $handoffDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $handoffZip)) {
    if (Test-Path -LiteralPath $handoffStage) {
        throw "Partial handoff staging directory exists: $handoffStage"
    }
    New-Item -ItemType Directory -Path $handoffStage | Out-Null
    foreach ($job in $jobs) {
        $source = Join-Path $artifactRoot $job.Name
        $destination = Join-Path $handoffStage $job.Name
        New-Item -ItemType Directory -Path $destination | Out-Null
        Copy-Item -LiteralPath (Join-Path $source "category-result.json") -Destination $destination
        Copy-Item -Path (Join-Path $source "calibration-*.jsonl") -Destination $destination
        Copy-Item -Path (Join-Path $source "predictions-*.jsonl") -Destination $destination
    }
    $selectionDestination = Join-Path $handoffStage "selection"
    New-Item -ItemType Directory -Path $selectionDestination | Out-Null
    Copy-Item -LiteralPath (Join-Path $selectionPath "selection-report.json") -Destination $selectionDestination
    Copy-Item -LiteralPath (Join-Path $selectionPath "summary.md") -Destination $selectionDestination
    $contractDestination = Join-Path $handoffStage "contracts"
    New-Item -ItemType Directory -Path $contractDestination | Out-Null
    Copy-Item -LiteralPath "configs\realiad-followup-preregistration-v1.toml" -Destination $contractDestination
    Copy-Item -LiteralPath "configs\realiad-followup-development-execution-v1.toml" -Destination $contractDestination
    Copy-Item -LiteralPath "data\manifests\realiad-followup-split-plan-v1.json" -Destination $contractDestination
    foreach ($job in $jobs) {
        Copy-Item -LiteralPath $job.Manifest -Destination $contractDestination
        Copy-Item -LiteralPath $job.FinalManifest -Destination $contractDestination
    }
    Compress-Archive -Path (Join-Path $handoffStage "*") -DestinationPath $handoffZip
}

$selection = Get-Content -LiteralPath (Join-Path $selectionPath "selection-report.json") -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "Real-IAD follow-up development comparison completed."
Write-Host "Selected candidate: $($selection.selected_candidate_id)"
Write-Host "Final holdout remains sealed and was not scored."
Write-Host "Attach this file to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
