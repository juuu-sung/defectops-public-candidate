[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
foreach ($path in @($defectPython, $Archive)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" }
}
Set-Location $projectPath
Write-Host 'Same256 training photos and78643 representatives. Global greedy in native768D, no quotas or projection.'
Write-Host 'WARNING:9GiB CPU feature corpus; about691TiB cumulative CPU-to-GPU traffic. This can take days.'
Write-Host 'First25 actual iterations print a timing estimate, then continue automatically. NO elapsed-time cap.'
Write-Host 'Close other GPU jobs normally. Keep ALL prior caches/models/ZIPs. Need10GiB free disk and6GiB free GPU.'
Write-Host 'Parity -> selection -> normal256/169 calibration -> reload -> all55/54/256 -> maps/cost/600 draws.'
Write-Host 'Ctrl+C once, wait for cleanup; resume from last1000-selection checkpoint. Do not update source mid-run.'
& $defectPython 'scripts\run_mpi_patchcore_raw_bank.py' --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Native768 bank stopped. Preserve every output; send error. No reroll or deletion.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-patchcore-raw-bank-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
