$ErrorActionPreference = "SilentlyContinue"

$root = $env:CLAUDE_PROJECT_DIR
if (-not $root) { $root = (Get-Location).Path }

$logDir = Join-Path $root ".claude\logs"
$logFile = Join-Path $logDir "postcompact.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$inputJson = [Console]::In.ReadToEnd()
if (-not $inputJson) { exit 0 }

try {
    $obj = $inputJson | ConvertFrom-Json
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logFile -Value "===== $timestamp ====="
    Add-Content -Path $logFile -Value ("trigger: " + $obj.trigger)
    Add-Content -Path $logFile -Value "summary:"
    Add-Content -Path $logFile -Value $obj.compact_summary
    Add-Content -Path $logFile -Value ""
} catch {
    exit 0
}
