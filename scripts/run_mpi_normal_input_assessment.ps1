[CmdletBinding()]
param(
    [switch]$VerifyArchive,
    [string]$Archive = (Join-Path $env:USERPROFILE 'Downloads\extracted.zip')
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defectPython = Join-Path $projectPath '.venv\Scripts\python.exe'
Set-Location $projectPath
Write-Host 'CPU-only input assessment. Existing model alerts and all false positives remain unchanged.'
$arguments = @('scripts\assess_mpi_normal_inputs.py')
$mode = 'records'
if ($VerifyArchive) { $arguments += @('--archive', $Archive); $mode = 'archive' }
& $defectPython @arguments
if ($LASTEXITCODE -ne 0) { throw 'Input assessment stopped. Preserve outputs and send the error.' }
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-normal-input-assessment-v1-$mode.zip"
Write-Host "Send this ZIP: $handoff"
Get-FileHash -LiteralPath $handoff -Algorithm SHA256
