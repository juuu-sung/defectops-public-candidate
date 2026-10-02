[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive, (Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-fit-v1.zip'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required input missing: $path" }
}
Set-Location $projectPath
Write-Host 'D-stage: observed test109 + normal256, separate fresh-process warm benchmarks, 3000 paired normal bootstraps.'
Write-Host 'No training, threshold tuning, automatic PASS or service promotion. C-stage resource violation remains recorded.'
Write-Host 'Close other GPU work normally. Cumulative cap: 2 hours, GPU reserved 6 GiB, process RSS 20 GiB.'
& $defectPython 'scripts\evaluate_mpi_normal_diversity.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw 'D-stage stopped. Preserve all files and send the error. Do not retrain or reroll the experiment.'
}
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-evaluation-v1.zip'
Write-Host "Share this JSON-only ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
