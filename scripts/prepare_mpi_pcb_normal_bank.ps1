[CmdletBinding()]
param(
    [string]$Archive = (Join-Path $env:USERPROFILE "Downloads\extracted.zip")
)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
Set-Location $projectPath

$required = @(
    $defectPython,
    $Archive,
    (Join-Path $projectPath "artifacts\data-audit\mpi-pcb-normal-pool-integrity-v1.json"),
    (Join-Path $projectPath "artifacts\data-audit\mpi-pcb-normal-preflight-v4.json"),
    (Join-Path $projectPath "artifacts\handoff\mpi-pcb-external-evaluation-v2.zip")
)
foreach ($inputPath in $required) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Required existing input is missing: $inputPath"
    }
}

Write-Host "Preparing 64 original normal photos x 12 regions; keeping 169 calibration photos unchanged."
Write-Host "Normal-only fingerprint audit. No training, GPU inference, test decoding or threshold changes."
& $defectPython "scripts\prepare_mpi_pcb_normal_bank.py" --project $projectPath --archive $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Preparation needs review. Original data and existing results were not changed. Share the error or generated ZIP."
}
$handoff = Join-Path $projectPath "artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip"
Write-Host "Attach this preparation ZIP for review before training: $handoff"
Get-FileHash -Algorithm SHA256 -LiteralPath $handoff
