[CmdletBinding()]
param([switch]$OpenFinalHoldout)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$handoffRoot = Join-Path $projectPath "artifacts\handoff"
$artifactRoot = Join-Path $projectPath "artifacts\realiad-followup-final-v1"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $defectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Final workflow stopped with Python exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Resolve local changes before the locked final run."
}

$jobs = @(
    @{ Category = "realiad_pcb"; Short = "pcb"; Seed = 42 },
    @{ Category = "realiad_pcb"; Short = "pcb"; Seed = 43 },
    @{ Category = "realiad_pcb"; Short = "pcb"; Seed = 44 },
    @{ Category = "realiad_terminalblock"; Short = "terminalblock"; Seed = 42 },
    @{ Category = "realiad_terminalblock"; Short = "terminalblock"; Seed = 43 },
    @{ Category = "realiad_terminalblock"; Short = "terminalblock"; Seed = 44 }
)

foreach ($job in $jobs) {
    Invoke-CheckedPython -Arguments @(
        "scripts\run_realiad_followup_final.py", "verify",
        "--category", $job.Category, "--seed", [string]$job.Seed
    )
}
foreach ($job in $jobs) {
    Invoke-CheckedPython -Arguments @(
        "scripts\run_realiad_followup_final.py", "fit",
        "--category", $job.Category, "--seed", [string]$job.Seed
    )
}

if (-not $OpenFinalHoldout) {
    Write-Host "All six locked fits/calibrations are ready. Final holdout remains sealed."
    Write-Host "To open and score it once, rerun this script with -OpenFinalHoldout."
    return
}

foreach ($job in $jobs) {
    Invoke-CheckedPython -Arguments @(
        "scripts\run_realiad_followup_final.py", "evaluate",
        "--category", $job.Category, "--seed", [string]$job.Seed
    )
}
Invoke-CheckedPython -Arguments @("scripts\run_realiad_followup_final.py", "summarize")

New-Item -ItemType Directory -Path $handoffRoot -Force | Out-Null
$stage = Join-Path $handoffRoot "realiad-followup-final-v1"
$handoffZip = Join-Path $handoffRoot "realiad-followup-final-v1.zip"
if (Test-Path -LiteralPath $handoffZip -PathType Leaf) {
    Write-Host "Existing handoff ZIP was not overwritten: $handoffZip"
} else {
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $artifactRoot "summary.json") -Destination $stage -Force
    $contracts = Join-Path $stage "contracts"
    New-Item -ItemType Directory -Path $contracts -Force | Out-Null
    Copy-Item -LiteralPath "configs\realiad-followup-final-lock-v1.toml" -Destination $contracts -Force
    Copy-Item -LiteralPath "configs\realiad-followup-preregistration-v1.toml" -Destination $contracts -Force
    Copy-Item -LiteralPath "data\manifests\realiad-pcb-final-sealed-v1.json" -Destination $contracts -Force
    Copy-Item -LiteralPath "data\manifests\realiad-terminalblock-final-sealed-v1.json" -Destination $contracts -Force
    Copy-Item -LiteralPath "artifacts\realiad-followup-development-v1\selection\selection-report.json" -Destination $contracts -Force
    foreach ($job in $jobs) {
        $name = "$($job.Short)-seed$($job.Seed)"
        $destination = Join-Path $stage $name
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $artifactRoot "fits\$name\fit.json") -Destination $destination -Force
        Copy-Item -LiteralPath (Join-Path $artifactRoot "evaluations\$name\evaluation.json") -Destination $destination -Force
        Copy-Item -LiteralPath (Join-Path $artifactRoot "evaluations\$name\predictions.jsonl") -Destination $destination -Force
    }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $handoffZip
}
Write-Host "Locked final evaluation completed; automatic PASS remains disabled."
Write-Host "Attach this ZIP to the Codex task: $handoffZip"
Get-FileHash -LiteralPath $handoffZip -Algorithm SHA256
