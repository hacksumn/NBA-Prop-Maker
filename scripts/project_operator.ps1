[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'RefreshLayers', 'CapturePlan', 'StartCapture', 'DailySafe', 'ReleaseManifest')]
    [string]$Action = 'Preflight',
    [string]$SlateDate = (Get-Date -Format 'yyyy-MM-dd'),
    [switch]$PairSportsbookT60
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$ProductionPatterns = @(
    'xgb_*_advanced.pkl',
    'clf_*_advanced.pkl',
    'edge_xgb_*_advanced.pkl',
    'meta_clf_*_advanced.pkl',
    'prob_sidecar_*_advanced.pkl',
    'probability_calibrator_advanced.pkl',
    'selected_features_advanced.json',
    'probability_diagnostics_advanced.json'
)

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$PythonArgs
    )
    Write-Host "`n[$Label]"
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Get-ProductionHashes {
    $files = foreach ($pattern in $ProductionPatterns) {
        Get-ChildItem -LiteralPath '.\models' -File -Filter $pattern -ErrorAction SilentlyContinue
    }
    $result = [ordered]@{}
    foreach ($file in ($files | Sort-Object FullName -Unique)) {
        $relative = [System.IO.Path]::GetRelativePath($ProjectRoot, $file.FullName)
        $result[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}

function Assert-ProductionUnchanged {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )
    $beforeJson = $Before | ConvertTo-Json -Compress
    $afterJson = $After | ConvertTo-Json -Compress
    if ($beforeJson -ne $afterJson) {
        throw 'Protected production-model hashes changed during a non-promotion operation.'
    }
    Write-Host "Protected production bundle unchanged ($($After.Count) hashed artifacts)." -ForegroundColor Green
}

function Assert-RequiredFiles {
    $required = @(
        '.\run_daily.py',
        '.\nba_props.py',
        '.\data\nba_data.csv',
        '.\data\historical_lines.csv',
        '.\data\model_data_defenserating_advanced.csv',
        '.\data\pointintime_team_defense_ratings.csv',
        '.\models\selected_features_advanced.json',
        '.\models\probability_calibrator_advanced.pkl',
        '.\scripts\capture_prizepicks_closing_snapshots.py'
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        throw "Required project files are missing: $($missing -join ', ')"
    }
}

function Show-Preflight {
    Assert-RequiredFiles
    Invoke-PythonStep -Label 'Python syntax/import smoke' -PythonArgs @(
        '-m', 'py_compile',
        '.\run_daily.py',
        '.\nba_props.py',
        '.\prizepicks_scraper.py',
        '.\scripts\capture_prizepicks_closing_snapshots.py',
        '.\scripts\build_release_manifest.py'
    )
    $hashes = Get-ProductionHashes
    if ($hashes.Count -lt 56) {
        throw "Expected at least 56 protected production artifacts; found $($hashes.Count)."
    }
    Write-Host "Required files present; $($hashes.Count) protected production artifacts hashed." -ForegroundColor Green
    Write-Host 'Odds API spend remains disabled unless DailySafe is called with -PairSportsbookT60 and the Python config also permits it.'
}

switch ($Action) {
    'Preflight' {
        Show-Preflight
    }
    'RefreshLayers' {
        Show-Preflight
        $before = Get-ProductionHashes
        Invoke-PythonStep -Label 'Refresh recovered modeling layers only' -PythonArgs @(
            '.\run_daily.py',
            '--layers-only',
            '--skip-picks',
            '--skip-pp-closing-capture',
            '--skip-weekly-retrain'
        )
        $after = Get-ProductionHashes
        Assert-ProductionUnchanged -Before $before -After $after
    }
    'CapturePlan' {
        Show-Preflight
        Invoke-PythonStep -Label "PrizePicks T-60/T-5 schedule for $SlateDate" -PythonArgs @(
            '.\scripts\capture_prizepicks_closing_snapshots.py',
            '--plan',
            '--date', $SlateDate
        )
    }
    'StartCapture' {
        if ($PairSportsbookT60) {
            throw 'StartCapture is intentionally PrizePicks-only. Use DailySafe -PairSportsbookT60 so both paid-spend gates are enforced.'
        }
        Show-Preflight
        $logPath = Join-Path $ProjectRoot 'logs\prizepicks_snapshot_watcher.log'
        New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null
        $arguments = @(
            '.\scripts\capture_prizepicks_closing_snapshots.py',
            '--watch',
            '--date', $SlateDate
        )
        $process = Start-Process -FilePath 'python' -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $logPath -PassThru
        Write-Host "PrizePicks-only watcher started (PID $($process.Id)). Log: $logPath" -ForegroundColor Green
    }
    'DailySafe' {
        Show-Preflight
        $before = Get-ProductionHashes
        $arguments = @('.\run_daily.py', '--skip-weekly-retrain')
        if ($PairSportsbookT60) {
            $arguments += '--pair-sportsbook-t60'
        }
        Invoke-PythonStep -Label 'Safe daily pipeline' -PythonArgs $arguments
        $after = Get-ProductionHashes
        Assert-ProductionUnchanged -Before $before -After $after
    }
    'ReleaseManifest' {
        Show-Preflight
        Invoke-PythonStep -Label 'Build release manifest' -PythonArgs @(
            '.\scripts\build_release_manifest.py'
        )
    }
}

