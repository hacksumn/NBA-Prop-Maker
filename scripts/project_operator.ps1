[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'ConfigureOddsKey', 'RefreshLayers', 'CapturePlan', 'StartCapture', 'FanDuelCapturePlan', 'StartFanDuelCapture', 'RefreshTrainingData', 'RetrainCandidate', 'InstallAutomation', 'AutomationStatus', 'PredictCached', 'PicksNow', 'DailySafe', 'ReleaseManifest')]
    [string]$Action = 'Preflight',
    [string]$SlateDate = (Get-Date -Format 'yyyy-MM-dd'),
    [switch]$PairSportsbookT60,
    [int]$FanDuelDailyCreditCap = 200,
    [int]$FanDuelMinimumRemainingCredits = 100
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# An already-open terminal can retain a canceled process-level key after the
# validated replacement has been saved at user scope. Always synchronize the
# persisted key before Python is launched so child processes cannot inherit the
# stale value. ConfigureOddsKey replaces both scopes after validation.
$PersistedOddsApiKey = [Environment]::GetEnvironmentVariable('ODDS_API_KEY', 'User')
if (-not [string]::IsNullOrWhiteSpace($PersistedOddsApiKey)) {
    $env:ODDS_API_KEY = $PersistedOddsApiKey
}
$PersistedOddsApiKey = $null

$ProductionPatterns = @(
    'xgb_*_advanced.pkl',
    'clf_*_advanced.pkl',
    'edge_xgb_*_advanced.pkl',
    'meta_clf_*_advanced.pkl',
    'prob_sidecar_*_advanced.pkl',
    'probability_calibrator_advanced.pkl',
    'bias_advanced.json',
    'defense_rating_feature_contract.json',
    'feature_importance_advanced.json',
    'market_feature_sets_advanced.json',
    'probability_oof_advanced.csv',
    'training_edge_analysis.json',
    'training_probability_analysis.json',
    'training_results.json',
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
        $relative = Join-Path 'models' $file.Name
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

function Invoke-ProtectedPythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$PythonArgs
    )
    $before = Get-ProductionHashes
    try {
        Invoke-PythonStep -Label $Label -PythonArgs $PythonArgs
    }
    finally {
        $after = Get-ProductionHashes
        Assert-ProductionUnchanged -Before $before -After $after
    }
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
        '.\scripts\capture_prizepicks_closing_snapshots.py',
        '.\scripts\capture_fanduel_closing_snapshots.py',
        '.\scripts\run_training_maintenance.py',
        '.\scripts\capture_candidate_future_holdout.py'
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
        '.\scripts\capture_fanduel_closing_snapshots.py',
        '.\scripts\run_training_maintenance.py',
        '.\scripts\capture_candidate_future_holdout.py',
        '.\scripts\build_release_manifest.py'
    )
    Invoke-PythonStep -Label 'Runtime dependency smoke' -PythonArgs @(
        '-c',
        'import pandas, numpy, sklearn, requests, nba_api'
    )
    $hashes = Get-ProductionHashes
    if ($hashes.Count -lt 56) {
        throw "Expected at least 56 protected production artifacts; found $($hashes.Count)."
    }
    Write-Host "Required files present; $($hashes.Count) protected production artifacts hashed." -ForegroundColor Green
    Write-Host 'Live cards use exact FanDuel lines from the Odds API; PrizePicks is not part of DailySafe.'
}

switch ($Action) {
    'Preflight' {
        Show-Preflight
    }
    'ConfigureOddsKey' {
        $secureKey = Read-Host 'Paste the active Odds API key (input is hidden)' -AsSecureString
        $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        try {
            $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer).Trim()
            if ($plainKey.Length -lt 16) {
                throw 'The supplied Odds API key is empty or unexpectedly short.'
            }
            $env:ODDS_API_KEY = $plainKey
            Invoke-PythonStep -Label 'Validate Odds API key without purchasing prop snapshots' -PythonArgs @(
                '-c',
                "import os,requests; r=requests.get('https://api.the-odds-api.com/v4/sports/basketball_nba/events',params={'apiKey':os.environ['ODDS_API_KEY']},timeout=20); p=r.json() if 'json' in r.headers.get('content-type','') else {}; print('HTTP',r.status_code,'remaining=',r.headers.get('x-requests-remaining','unknown'),'used=',r.headers.get('x-requests-used','unknown')); print((p.get('error_code','')+': '+p.get('message','')).strip(': ')) if isinstance(p,dict) and r.status_code != 200 else None; raise SystemExit(0 if r.status_code == 200 else 1)"
            )
            [Environment]::SetEnvironmentVariable('ODDS_API_KEY', $plainKey, 'User')
            Write-Host 'The validated key is active for this PowerShell session and future sessions.' -ForegroundColor Green
            Write-Host 'Run: .\scripts\project_operator.ps1 -Action DailySafe'
        }
        finally {
            if ($keyPointer -ne [IntPtr]::Zero) {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
            }
            $plainKey = $null
        }
    }
    'RefreshLayers' {
        Show-Preflight
        Invoke-ProtectedPythonStep -Label 'Refresh recovered modeling layers only' -PythonArgs @(
            '.\run_daily.py',
            '--layers-only',
            '--skip-picks',
            '--skip-pp-closing-capture',
            '--skip-weekly-retrain'
        )
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
        $errorLogPath = Join-Path $ProjectRoot 'logs\prizepicks_snapshot_watcher.error.log'
        New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null
        $arguments = @(
            '.\scripts\capture_prizepicks_closing_snapshots.py',
            '--watch',
            '--date', $SlateDate
        )
        $process = Start-Process -FilePath 'python' -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorLogPath -PassThru
        Write-Host "PrizePicks-only watcher started (PID $($process.Id)). Logs: $logPath and $errorLogPath" -ForegroundColor Green
    }
    'FanDuelCapturePlan' {
        Show-Preflight
        Invoke-PythonStep -Label "FanDuel T-60/T-5 schedule for $SlateDate" -PythonArgs @(
            '.\scripts\capture_fanduel_closing_snapshots.py',
            '--plan',
            '--date', $SlateDate,
            '--max-daily-credits', $FanDuelDailyCreditCap.ToString(),
            '--min-remaining-credits', $FanDuelMinimumRemainingCredits.ToString()
        )
    }
    'StartFanDuelCapture' {
        Show-Preflight
        $logPath = Join-Path $ProjectRoot 'logs\fanduel_closing_watcher.log'
        $errorLogPath = Join-Path $ProjectRoot 'logs\fanduel_closing_watcher.error.log'
        New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null
        $arguments = @(
            '.\scripts\capture_fanduel_closing_snapshots.py',
            '--watch',
            '--date', $SlateDate,
            '--max-daily-credits', $FanDuelDailyCreditCap.ToString(),
            '--min-remaining-credits', $FanDuelMinimumRemainingCredits.ToString()
        )
        $process = Start-Process -FilePath 'python' -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorLogPath -PassThru
        Write-Host "FanDuel T-60/T-5 watcher started (PID $($process.Id)). Logs: $logPath and $errorLogPath" -ForegroundColor Green
    }
    'RefreshTrainingData' {
        Show-Preflight
        Invoke-ProtectedPythonStep -Label 'Rebuild unified availability + defense training table' -PythonArgs @(
            '.\scripts\run_training_maintenance.py', '--mode', 'refresh'
        )
    }
    'RetrainCandidate' {
        Show-Preflight
        Invoke-ProtectedPythonStep -Label 'Rebuild data and train isolated full-layer candidate' -PythonArgs @(
            '.\scripts\run_training_maintenance.py', '--mode', 'all'
        )
    }
    'InstallAutomation' {
        & '.\scripts\install_project_automation.ps1'
    }
    'AutomationStatus' {
        Get-ScheduledTask | Where-Object { $_.TaskName -like 'FreshStartNBA-*' } |
            Select-Object TaskName, State, @{N='Execute';E={$_.Actions.Execute}}, @{N='Arguments';E={$_.Actions.Arguments}} |
            Format-Table -AutoSize
    }
    'DailySafe' {
        Show-Preflight
        $arguments = @('.\run_daily.py', '--skip-weekly-retrain')
        if ($PairSportsbookT60) {
            $arguments += '--pair-sportsbook-t60'
        }
        Invoke-ProtectedPythonStep -Label 'Safe daily pipeline' -PythonArgs $arguments
        Invoke-ProtectedPythonStep -Label 'Grade and capture frozen candidate future holdout' -PythonArgs @(
            '.\scripts\capture_candidate_future_holdout.py'
        )
    }
    'PicksNow' {
        Show-Preflight
        Invoke-ProtectedPythonStep -Label 'Refresh FanDuel lines and generate picks' -PythonArgs @(
            '.\run_daily.py',
            '--skip-layers',
            '--skip-weekly-retrain'
        )
    }
    'PredictCached' {
        Show-Preflight
        Invoke-ProtectedPythonStep -Label 'Generate picks from exact cached FanDuel lines' -PythonArgs @(
            '.\nba_props.py', 'predict'
        )
    }
    'ReleaseManifest' {
        Show-Preflight
        Invoke-PythonStep -Label 'Build release manifest' -PythonArgs @(
            '.\scripts\build_release_manifest.py'
        )
    }
}

