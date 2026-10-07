[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive,
    (Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-scaling-v1.zip'),
    (Join-Path $projectPath 'artifacts\handoff\mpi-normal-diversity-preparation-v1.zip'),
    (Join-Path $projectPath 'artifacts\mpi-patchcore-scaling-v1\model\model.pt'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Same frozen256-photo PatchCore bank. Development normals128 ->256. No training.'
Write-Host 'Select128 disjoint normals -> replay -> normal-only inference -> new contract -> full cached comparison.'
Write-Host 'No test-image reads. Original model, contracts, service and files stay unchanged.'
Write-Host 'Need5 GiB disk and6 GiB free GPU. RAM20 / GPU reserved6 GiB limits; NO time cap.'
Write-Host 'Close other GPU jobs normally. Ctrl+C once and wait; same command resumes photos.'
Write-Host 'Do not git pull mid-run. Failed or unaccounted attempts require review; keep every file.'
& $defectPython 'scripts\run_mpi_patchcore_development_scaling.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Development scaling stopped. Preserve files and send full error. Do not retrain.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-development-scaling-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
