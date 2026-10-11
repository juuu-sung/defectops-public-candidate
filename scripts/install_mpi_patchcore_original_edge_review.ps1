[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ProjectPath)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $ProjectPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.venv\Scripts\python.exe') -PathType Leaf)) {
    throw 'Expected existing DefectOps project and Python environment.'
}
function Get-NormalizedHash([string]$Path) {
    $text = [System.IO.File]::ReadAllText($Path).Replace("`r`n", "`n")
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runner-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$pending = @()
foreach ($entry in $manifest.PSObject.Properties) {
    $relative = $entry.Name.Replace('/', '\')
    if ([System.IO.Path]::IsPathRooted($relative) -or $relative.Contains('..') -or
        $relative.Contains(':') -or $relative -notmatch '^(scripts|tests|docs|src)\\') {
        throw "Unsafe package path: $relative"
    }
    $source = Join-Path (Join-Path $PSScriptRoot 'files') $relative
    $target = Join-Path $projectRoot $relative
    if ((Get-NormalizedHash $source) -ne $entry.Value) { throw "Package checksum differs: $relative" }
    if (Test-Path -LiteralPath $target) {
        if ((Get-NormalizedHash $target) -ne $entry.Value) {
            throw "Existing file differs; nothing has been copied. Preserve and report: $target"
        }
    } else { $pending += [pscustomobject]@{Source=$source; Target=$target} }
}
# Validate every existing file before copying anything. Never overwrite local changes.
foreach ($entry in $pending) {
    $parent = Split-Path -Parent $entry.Target
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    [System.IO.File]::Copy($entry.Source, $entry.Target, $false)
}
Write-Host "Ready. Added $($pending.Count) missing source files; existing matching files unchanged. No models or results touched."
