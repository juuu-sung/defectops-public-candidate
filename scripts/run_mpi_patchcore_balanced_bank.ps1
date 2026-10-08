[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Same256 training photos, SAME78643 bank. Change representative selection ONLY.'
Write-Host '12 balanced region quotas; inference still searches the entire bank, not same-region only.'
Write-Host 'Keep existing raw/projected caches, fit folders and handoff ZIPs unchanged.'
Write-Host 'Close other GPU jobs normally. Need10 GiB free disk and6 GiB free GPU. NO elapsed-time cap.'
Write-Host 'Select -> normal256/169 calibration -> reload -> all55 defects and normal54/256 -> maps/cost/600 draws.'
Write-Host 'Ctrl+C once, wait for cleanup. Same command resumes sealed checkpoints. Do not git pull mid-run.'
& $defectPython 'scripts\run_mpi_patchcore_balanced_bank.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Balanced bank stopped. Preserve every file; send error. No deletion or reroll.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-balanced-bank-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
