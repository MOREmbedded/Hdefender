# Keep every Cargo download, cache and temp file on F: (nothing on C:).
# Run once per PowerShell window, from the repository root:
#   . .\dev-env.ps1
$root = $PSScriptRoot
$env:CARGO_HOME = Join-Path $root ".cache\cargo"
$env:TEMP = Join-Path $root ".cache\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:CARGO_HOME, $env:TEMP | Out-Null
Write-Host "CARGO_HOME = $env:CARGO_HOME"
Write-Host "TEMP/TMP   = $env:TEMP"
