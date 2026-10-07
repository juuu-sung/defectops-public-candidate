[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive,
    (Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-scaling-probe-v1.zip'),
    (Join-Path $projectPath 'artifacts\handoff\mpi-supersimplenet-scaling-v1.zip'),
    (Join-Path $projectPath 'artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Fixed PatchCore normal64 -> 256; ALL 3,145,728 candidates; SAME 78,643 bank size.'
Write-Host 'Extract to disk -> chunked coreset -> normal128/169 calibration/reload -> full observed comparison.'
Write-Host '32 GiB free disk at start, 6 GiB free GPU. RAM20 / GPU reserved6 GiB limits. NO elapsed-time cap.'
Write-Host 'Close other GPU jobs normally. Do not git pull mid-run. Preserve all older results.'
Write-Host 'Ctrl+C once and wait for cleanup. Same command resumes completed photo/chunk/1000-selection checkpoints.'
Write-Host 'Failed/resource-limited/unaccounted attempts require review; never delete, shrink or reroll.'
& $defectPython 'scripts\run_mpi_patchcore_scaling.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'PatchCore scaling stopped. Preserve every file and send full error. No service change.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-scaling-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
