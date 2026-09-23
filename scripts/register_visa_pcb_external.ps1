[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defectPython = Join-Path $projectPath ".venv\Scripts\python.exe"
$archivePath = Join-Path $env:USERPROFILE "Downloads\VisA_20220922.tar"
$archiveUrl = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
$expectedArchiveBytes = [int64]1929840640
Set-Location $projectPath
if (-not (Test-Path -LiteralPath $defectPython -PathType Leaf)) {
    throw "Project Python was not found: $defectPython"
}
if (git status --porcelain) {
    throw "Git working tree is not clean. Pull and resolve local changes first."
}
$currentBytes = if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
    (Get-Item -LiteralPath $archivePath).Length
} else { [int64]0 }
if ($currentBytes -gt $expectedArchiveBytes) {
    throw "Existing VisA archive is larger than the official source and was not changed: $archivePath"
}
if ($currentBytes -lt $expectedArchiveBytes) {
    Write-Host "Downloading/resuming the official 1.93 GB VisA archive to $archivePath"
    & curl.exe --location --fail --continue-at - --retry 3 --output $archivePath $archiveUrl
    if ($LASTEXITCODE -ne 0) {
        throw "VisA download failed; the partial file was kept so the same command can resume it."
    }
}
if ((Get-Item -LiteralPath $archivePath).Length -ne $expectedArchiveBytes) {
    throw "VisA archive size differs from the preregistered source; the file was not deleted."
}
& $defectPython "scripts\register_visa_pcb_external.py" --archive $archivePath
if ($LASTEXITCODE -ne 0) {
    throw "VisA metadata registration failed; no test image was decoded by the registration script."
}
Get-FileHash -LiteralPath (Join-Path $projectPath "artifacts\handoff\visa-pcb-external-registration-v1.zip") -Algorithm SHA256
