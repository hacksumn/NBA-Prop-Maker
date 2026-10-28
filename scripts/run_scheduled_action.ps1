[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('DailySafe', 'RefreshTrainingData', 'StartFanDuelCapture', 'RetrainCandidate')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDirectory = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("scheduled_{0}.log" -f $Action.ToLowerInvariant())
$Started = Get-Date

try {
    "[$($Started.ToString('s'))] START $Action" | Add-Content -LiteralPath $LogPath
    & (Join-Path $PSScriptRoot 'project_operator.ps1') -Action $Action *>&1 |
        Tee-Object -FilePath $LogPath -Append
    "[$((Get-Date).ToString('s'))] COMPLETE $Action" | Add-Content -LiteralPath $LogPath
    exit 0
}
catch {
    "[$((Get-Date).ToString('s'))] FAILED $Action :: $($_.Exception.Message)" |
        Add-Content -LiteralPath $LogPath
    throw
}
