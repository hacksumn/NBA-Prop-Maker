[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot 'run_scheduled_action.ps1'
$PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
$UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Scheduled action runner is missing: $Runner"
}

$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12)

$Definitions = @(
    @{
        Name = 'FreshStartNBA-DailySafe'
        Action = 'DailySafe'
        Trigger = New-ScheduledTaskTrigger -Daily -At '8:30 AM'
        Description = 'Refresh completed NBA data, live layers, FanDuel lines, picks, and frozen shadow grading.'
    },
    @{
        Name = 'FreshStartNBA-TrainingRefresh'
        Action = 'RefreshTrainingData'
        Trigger = New-ScheduledTaskTrigger -Daily -At '10:30 AM'
        Description = 'Rebuild leakage-safe advanced, availability/minutes, and point-in-time defense training tables.'
    },
    @{
        Name = 'FreshStartNBA-FanDuelClosing'
        Action = 'StartFanDuelCapture'
        Trigger = New-ScheduledTaskTrigger -Daily -At '12:00 PM'
        Description = 'Start the guarded FanDuel T-60/T-5 per-event snapshot watcher.'
    },
    @{
        Name = 'FreshStartNBA-WeeklyCandidate'
        Action = 'RetrainCandidate'
        # Keep the expensive refresh/retrain safely behind Sunday's ordinary
        # 10:30 AM training-table refresh, including the DST boundary.
        Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '3:00 PM'
        Description = 'Train and audit an isolated full-layer candidate; production promotion is prohibited.'
    }
)

foreach ($Definition in $Definitions) {
    $Arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Action {1}' -f $Runner, $Definition.Action
    $Action = New-ScheduledTaskAction `
        -Execute $PowerShell `
        -Argument $Arguments `
        -WorkingDirectory $ProjectRoot
    Register-ScheduledTask `
        -TaskName $Definition.Name `
        -Action $Action `
        -Trigger $Definition.Trigger `
        -Principal $Principal `
        -Settings $Settings `
        -Description $Definition.Description `
        -Force | Out-Null
}

Get-ScheduledTask | Where-Object { $_.TaskName -like 'FreshStartNBA-*' } |
    Select-Object TaskName, State, @{N='Execute';E={$_.Actions.Execute}}, @{N='Arguments';E={$_.Actions.Arguments}} |
    Sort-Object TaskName |
    Format-Table -AutoSize
