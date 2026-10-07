[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, (Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-scaling-v1.zip'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Synthetic CUDA probe only. NO product images, model fit, test, threshold or service changes.'
Write-Host 'Checks official small-case parity then 3,145,728 candidates with 100 measured selection iterations.'
Write-Host 'Need 6 GiB GPU free and 2 GiB disk; RAM20 / GPU reserved6 GiB limits; no elapsed-time cap.'
Write-Host 'Close other GPU jobs normally. Do not git pull while running. Preserve every previous output.'
Write-Host 'Ctrl+C once and wait. Completed stages resume; incomplete synthetic stage replays the same fixed seed.'
& $defectPython 'scripts\run_mpi_patchcore_scaling_probe.py' --project $projectPath
if ($LASTEXITCODE -ne 0) { throw 'PatchCore synthetic probe stopped. Preserve files and send the error. Do not start training.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-scaling-probe-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
