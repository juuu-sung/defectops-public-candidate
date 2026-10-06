[CmdletBinding()]
param(
    [string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'),
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
foreach ($name in @('mpi-supersimplenet-fit-v1.zip', 'mpi-normal-diversity-preparation-v1.zip')) {
    $path = Join-Path $projectPath "artifacts\handoff\$name"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing reviewed handoff: $path" }
}
Set-Location $projectPath
Write-Host 'Freeze original64 + deterministic192 normals. Development128/calibration169/validation256 remain excluded.'
Write-Host 'Fresh A64x300 -> fresh C256x75 -> fresh D256x300. Same official loss, batch8, seed42; no defect gate.'
Write-Host 'No elapsed-time cap. Close other GPU jobs normally. RAM20 GiB / GPU reserved6 GiB limits remain.'
Write-Host 'Need 100 GiB free before training. All checkpoints kept. No data/model/package changes to prior runs.'
Write-Host 'Ctrl+C once: wait for the worker to stop. Same command resumes committed checkpoints. Do not git pull mid-run.'
$arguments = @('scripts\run_mpi_supersimplenet_scaling.py', '--project', $projectPath, '--archive', $Archive)
if ($PrepareOnly) { $arguments += '--prepare-only' }
& $defectPython @arguments
if ($LASTEXITCODE -ne 0) { throw 'Scaling stopped. Keep every output and send the error. No test evaluation or reroll.' }
$name = if ($PrepareOnly) { 'mpi-supersimplenet-scaling-v1-preparation.zip' } else { 'mpi-supersimplenet-scaling-v1.zip' }
$handoff = Join-Path $projectPath "artifacts\handoff\$name"
Write-Host "Send this JSON-only evidence ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
