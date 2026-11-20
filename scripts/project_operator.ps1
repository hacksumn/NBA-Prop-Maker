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
    # Windows PowerShell 5.1 turns native stderr into an ErrorRecord, and with
    # $ErrorActionPreference='Stop' the first stderr line (for example a Python
    # logging.INFO line) aborts the step even when the exit code is 0 (DEC-072).
    # Run the native command with 'Continue', render stderr as plain text, and
    # decide success from the exit code only.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($PythonArgs[0] -match "\.py$") {
            $PythonArgs = @(".\artifact_retention.py", "--run") + $PythonArgs
        }
        & python @PythonArgs 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
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

# Rebuild the read-only Command Center page (output\dashboard_pro.html) after the daily run (DEC-088).
# Informational only, so by design it:
#   - never throws and never changes the result of the action it runs after;
#   - is NOT part of Preflight / Assert-RequiredFiles (a dashboard problem must never be able to block picks);
#   - is NOT routed through Invoke-PythonStep, which wraps .py files in artifact_retention.py --run and would
#     keep every replaced copy of this regenerable page forever;
#   - is time-limited, and says plainly in the log when the page was NOT refreshed (the page also shows its own
#     age and an "Out of date" banner after 26 hours).
function Invoke-DashboardRebuild {
    param([int]$TimeoutSeconds = 180)
    $label = '[Dashboard rebuild]'
    $stdoutPath = $null
    $stderrPath = $null
    try {
        $builder = Join-Path $ProjectRoot 'scripts\build_dashboard.py'
        if (-not (Test-Path -LiteralPath $builder -PathType Leaf)) {
            Write-Host "$label SKIPPED: $builder not found; output\dashboard_pro.html was NOT refreshed." -ForegroundColor Yellow
            return
        }
        $python = (Get-Command python -ErrorAction Stop).Source
        $stdoutPath = [IO.Path]::GetTempFileName()
        $stderrPath = [IO.Path]::GetTempFileName()
        $process = Start-Process -FilePath $python -ArgumentList @("`"$builder`"") -WorkingDirectory $ProjectRoot `
            -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $null = $process.Handle   # keep the handle so ExitCode is readable after WaitForExit
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { Stop-Process -Id $process.Id -Force -ErrorAction Stop } catch { }
            Write-Host "$label FAILED: no result after $TimeoutSeconds s, so it was stopped; output\dashboard_pro.html was NOT refreshed." -ForegroundColor Yellow
            return
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        foreach ($path in @($stdoutPath, $stderrPath)) {
            foreach ($line in @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)) {
                if ($line) { Write-Host "$label $line" }
            }
        }
        if ($exitCode -eq 0) {
            Write-Host "$label OK" -ForegroundColor Green
        }
        else {
            Write-Host "$label FAILED with exit code $exitCode; output\dashboard_pro.html may be out of date." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "$label FAILED: $($_.Exception.Message); output\dashboard_pro.html may be out of date." -ForegroundColor Yellow
    }
    finally {
        foreach ($path in @($stdoutPath, $stderrPath)) {
            if ($path) { Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue }
        }
    }
}

# Upload the page rebuilt above to Cloudflare Pages so the shared link stays current (DEC-100).
# Opt-in (scripts\dashboard_publish.local.json, "enabled": true) and informational only, so by design it:
#   - never throws and never changes the result of the action it runs after;
#   - is NOT part of Preflight / Assert-RequiredFiles and is NOT routed through Invoke-PythonStep;
#   - uploads only a throwaway copy of output\dashboard_pro.html (plus noindex headers), never the project folder;
#   - refuses to upload a page that was not rebuilt in this run, so a failed rebuild cannot re-publish old data
#     as if it were new (the hosted page also shows its own age and an "Out of date" banner after 26 hours);
#   - is time-limited and says plainly in the log when the hosted page was NOT updated.
# Credentials are Cloudflare's own: the `wrangler login` session, or CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID
# at user scope. They are never read into the log.
function Invoke-DashboardPublish {
    param([int]$TimeoutSeconds = 180, [int]$MaxPageAgeMinutes = 30)
    $label = '[Dashboard publish]'
    $stdoutPath = $null
    $stderrPath = $null
    $stageDir = $null
    $savedEnv = @{}
    try {
        $configPath = Join-Path $ProjectRoot 'scripts\dashboard_publish.local.json'
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            Write-Host "$label SKIPPED: $configPath not found; the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        $config = Get-Content -LiteralPath $configPath -Raw -ErrorAction Stop | ConvertFrom-Json
        # Strict: PowerShell's -ne would treat the strings "true"/"yes" and the number 1 as enabled.
        if (($config.enabled -isnot [bool]) -or ($config.enabled -ne $true)) {
            Write-Host "$label SKIPPED: not enabled in scripts\dashboard_publish.local.json; the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        $project = [string]$config.project
        $branch = [string]$config.branch
        if ($project -notmatch '^[a-z0-9][a-z0-9-]{1,56}[a-z0-9]$' -or $branch -notmatch '^[A-Za-z0-9._-]{1,64}$') {
            Write-Host "$label FAILED: scripts\dashboard_publish.local.json has an invalid project or branch name; the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        # Where wrangler keeps its Cloudflare login. The Claude desktop app is a packaged app: it redirects everything it
        # writes under AppData into a private folder that Task Scheduler cannot see, so a login made from inside it is
        # invisible to the 07:30 run (DEC-102). "wranglerConfigHome" in the settings file names a folder outside AppData
        # that every program, including the scheduled task, sees; wrangler is pointed at it through XDG_CONFIG_HOME.
        $configHome = [string]$config.wranglerConfigHome
        if ($configHome) {
            if (-not [IO.Path]::IsPathRooted($configHome) -or -not (Test-Path -LiteralPath $configHome -PathType Container)) {
                Write-Host "$label FAILED: wranglerConfigHome in scripts\dashboard_publish.local.json is not an existing absolute folder; the hosted dashboard was NOT updated." -ForegroundColor Yellow
                return
            }
            $credentialFile = Join-Path $configHome '.wrangler\config\default.toml'
        }
        elseif ($env:XDG_CONFIG_HOME) {
            $credentialFile = Join-Path $env:XDG_CONFIG_HOME '.wrangler\config\default.toml'
        }
        else {
            $credentialFile = Join-Path $env:APPDATA 'xdg.config\.wrangler\config\default.toml'
        }
        $page = Join-Path $ProjectRoot 'output\dashboard_pro.html'
        if (-not (Test-Path -LiteralPath $page -PathType Leaf)) {
            Write-Host "$label SKIPPED: $page not found; the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        $ageMinutes = ((Get-Date) - (Get-Item -LiteralPath $page).LastWriteTime).TotalMinutes
        if ($ageMinutes -gt $MaxPageAgeMinutes) {
            Write-Host ("$label SKIPPED: output\dashboard_pro.html is {0:N0} minutes old, so it was not rebuilt in this run; the hosted dashboard was NOT updated." -f $ageMinutes) -ForegroundColor Yellow
            return
        }
        $npx = (Get-Command npx.cmd -ErrorAction Stop).Source
        # Same pattern as ODDS_API_KEY: a scheduled task may not carry the persisted user-level values.
        foreach ($name in @('CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID')) {
            $persisted = [Environment]::GetEnvironmentVariable($name, 'User')
            if (-not [string]::IsNullOrWhiteSpace($persisted) -and -not (Test-Path -LiteralPath "Env:$name")) {
                $savedEnv[$name] = $null
                Set-Item -LiteralPath "Env:$name" -Value $persisted
            }
        }
        $envPairs = @(@('CI', 'true'), @('WRANGLER_SEND_METRICS', 'false'))
        if ($configHome) { $envPairs += , @('XDG_CONFIG_HOME', $configHome) }
        foreach ($pair in $envPairs) {
            $savedEnv[$pair[0]] = [Environment]::GetEnvironmentVariable($pair[0], 'Process')
            Set-Item -LiteralPath ("Env:" + $pair[0]) -Value $pair[1]
        }
        # Say plainly when there is no login to use, instead of wrangler's generic "non-interactive environment" error.
        if ([string]::IsNullOrWhiteSpace($env:CLOUDFLARE_API_TOKEN) -and -not (Test-Path -LiteralPath $credentialFile -PathType Leaf)) {
            Write-Host "$label FAILED: no Cloudflare login found at $credentialFile and no CLOUDFLARE_API_TOKEN is set. Run scripts\cloudflare_login.bat once from a normal window (not from inside another app); the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        $stageDir = Join-Path ([IO.Path]::GetTempPath()) ('dash_site_' + [guid]::NewGuid().ToString('N'))
        $null = New-Item -ItemType Directory -Path $stageDir -ErrorAction Stop
        Copy-Item -LiteralPath $page -Destination (Join-Path $stageDir 'index.html') -ErrorAction Stop
        Set-Content -LiteralPath (Join-Path $stageDir '_headers') -Encoding ASCII -Value @(
            '/*', '  X-Robots-Tag: noindex, nofollow, noarchive', '  Referrer-Policy: no-referrer', '  X-Content-Type-Options: nosniff')
        Set-Content -LiteralPath (Join-Path $stageDir 'robots.txt') -Encoding ASCII -Value @('User-agent: *', 'Disallow: /')
        $stdoutPath = [IO.Path]::GetTempFileName()
        $stderrPath = [IO.Path]::GetTempFileName()
        $message = 'dashboard refresh ' + (Get-Date -Format 'yyyy-MM-dd HH:mm')
        # Run from the throwaway folder so wrangler never reads this project's git history or config.
        $process = Start-Process -FilePath $npx -WorkingDirectory $stageDir -NoNewWindow -PassThru `
            -ArgumentList @('--yes', 'wrangler@4.164.0', 'pages', 'deploy', '.', '--project-name', $project, '--branch', $branch,
                            '--commit-message', "`"$message`"", '--commit-dirty=true') `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $null = $process.Handle   # keep the handle so ExitCode is readable after WaitForExit
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { & taskkill.exe /PID $process.Id /T /F *> $null } catch { }
            Write-Host "$label FAILED: no result after $TimeoutSeconds s, so it was stopped; the hosted dashboard was NOT updated." -ForegroundColor Yellow
            return
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        foreach ($path in @($stdoutPath, $stderrPath)) {
            foreach ($line in @(Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)) {
                # Drop colour codes and box-drawing characters so the log stays readable.
                $clean = ($line -replace '\x1b\[[0-9;]*m', '') -replace '[^\x20-\x7E]', ''
                if ($clean.Trim()) { Write-Host "$label $clean" }
            }
        }
        if ($exitCode -eq 0) {
            Write-Host "$label OK: hosted dashboard updated ($project, branch $branch)." -ForegroundColor Green
        }
        else {
            Write-Host "$label FAILED with exit code $exitCode; the hosted dashboard was NOT updated and will show its age. If wrangler reports an expired or missing login, run scripts\cloudflare_login.bat once from a normal window." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "$label FAILED: $($_.Exception.Message); the hosted dashboard was NOT updated." -ForegroundColor Yellow
    }
    finally {
        foreach ($name in $savedEnv.Keys) {
            if ($null -eq $savedEnv[$name]) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
            else { Set-Item -LiteralPath "Env:$name" -Value $savedEnv[$name] }
        }
        foreach ($path in @($stdoutPath, $stderrPath)) {
            if ($path) { Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue }
        }
        if ($stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force -ErrorAction SilentlyContinue }
    }
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
        Invoke-PythonStep -Label 'Retain previous watcher logs' -PythonArgs @('.\artifact_retention.py', '--retain', $logPath, $errorLogPath)
        $arguments = @('.\artifact_retention.py', '--run') + $arguments
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
        Invoke-PythonStep -Label 'Retain previous watcher logs' -PythonArgs @('.\artifact_retention.py', '--retain', $logPath, $errorLogPath)
        $arguments = @('.\artifact_retention.py', '--run') + $arguments
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
        try {
            Invoke-ProtectedPythonStep -Label 'Safe daily pipeline' -PythonArgs $arguments
            Invoke-ProtectedPythonStep -Label 'Grade and capture frozen candidate future holdout' -PythonArgs @(
                '.\scripts\capture_candidate_future_holdout.py'
            )
        }
        finally {
            # Runs even when a step above failed, so the page's Health view shows the failure. It cannot
            # throw, so it can never mask the real error or turn a failed run into a passing one (DEC-088).
            Invoke-DashboardRebuild
            # Opt-in upload of the page just rebuilt; same guarantees: it cannot throw (DEC-100).
            Invoke-DashboardPublish
        }
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

