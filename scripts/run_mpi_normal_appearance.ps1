[CmdletBinding()]
param([string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip'))
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'CPU census: fixed256 training and256 validation normal photos,6144 PNG tiles.'
Write-Host 'No GPU, model loading, inference, test images, masks or threshold changes.'
& $defectPython 'scripts\review_mpi_normal_appearance.py' --archive $Archive
if ($LASTEXITCODE -ne 0) { throw 'Normal appearance census stopped. Preserve all outputs and send the error.' }
$handoff = Join-Path $projectPath 'artifacts\handoff\mpi-normal-appearance-census-v1.zip'
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
