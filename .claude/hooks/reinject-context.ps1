$ErrorActionPreference = "SilentlyContinue"

$root = $env:CLAUDE_PROJECT_DIR
if (-not $root) { $root = (Get-Location).Path }

# STATUS.md / TASKS.md / DECISIONS.md may live at root or in output/
# Check both locations so the hook works regardless of where they were placed.
function Find-MemoryFile {
    param([string]$filename)
    $candidates = @(
        (Join-Path $root $filename),
        (Join-Path $root "output\$filename")
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$files = @("STATUS.md", "TASKS.md", "DECISIONS.md")

Write-Output "========================================================"
Write-Output "COMPACTION REMINDER — Fresh Start NBA"
Write-Output "========================================================"
Write-Output "Treat repo files as canonical project memory."
Write-Output "Read STATUS.md, TASKS.md, and DECISIONS.md before continuing non-trivial work."
Write-Output "Do not rely on older chat messages when those files disagree."
Write-Output ""
Write-Output "Key rules:"
Write-Output "  - Never break run_morning.bat (runs at ~9am with real money bets)"
Write-Output "  - Read actual files before editing — never assume column names"
Write-Output "  - Patch surgically — change only what needs to change"
Write-Output "  - line_source column in picks_history.csv tracks PrizePicks vs Odds API"
Write-Output "  - pick_source='volume_fill' = relaxed-threshold picks, capped at 63% confidence"
Write-Output ""
Write-Output "CURRENT SNAPSHOT:"

foreach ($f in $files) {
    $path = Find-MemoryFile $f
    if ($path) {
        Write-Output ""
        Write-Output "===== $f ====="
        Get-Content $path -TotalCount 120
    } else {
        Write-Output ""
        Write-Output "===== $f — NOT FOUND (check root and output/) ====="
    }
}
