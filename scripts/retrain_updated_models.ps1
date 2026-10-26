[CmdletBinding()]
param(
    [string]$CandidateDir = "",
    [switch]$Promote
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$RunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if ([string]::IsNullOrWhiteSpace($CandidateDir)) {
    $CandidateDir = Join-Path 'models' "manual_retrain_$RunStamp"
}

$CandidateOof = Join-Path $CandidateDir 'probability_oof_advanced.csv'
$CandidateFeatures = Join-Path $CandidateDir 'selected_features_advanced.json'
$DetailOutput = Join-Path 'output' "manual_retrain_${RunStamp}_same_row_oof.csv"
$SummaryOutput = Join-Path 'output' "manual_retrain_${RunStamp}_same_row_oof_summary.json"

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

Invoke-PythonStep -Label 'Training/live selected-feature contract' -PythonArgs @(
    '.\scripts\audit_training_live_feature_parity.py'
)

Invoke-PythonStep -Label 'Isolated nine-prop retrain' -PythonArgs @(
    '.\train_advanced_models.py',
    '--input', '.\data\model_data_defenserating_advanced.csv',
    '--model-dir', $CandidateDir,
    '--require-defense-rating-features',
    '--no-auto-build-features'
)

Invoke-PythonStep -Label 'Same-row OOF comparison' -PythonArgs @(
    '.\scripts\evaluate_defense_rating_retrain.py',
    '--production', '.\models\probability_oof_advanced.csv',
    '--candidate', $CandidateOof,
    '--production-features', '.\models\selected_features_advanced.json',
    '--candidate-features', $CandidateFeatures,
    '--detail', $DetailOutput,
    '--summary', $SummaryOutput
)

if ($Promote) {
    $BackupDir = Join-Path 'models\backups' "manual_pre_activation_$RunStamp"
    $Manifest = Join-Path 'output' "manual_retrain_${RunStamp}_activation_manifest.json"
    Invoke-PythonStep -Label 'Atomic production activation' -PythonArgs @(
        '.\scripts\promote_defense_rating_models.py',
        '--source', $CandidateDir,
        '--backup', $BackupDir,
        '--manifest', $Manifest
    )
    Write-Host "Production backup: $BackupDir"
    Write-Host "Activation manifest: $Manifest"
}

Write-Host "`nCandidate models: $CandidateDir"
Write-Host "OOF comparison: $SummaryOutput"
if (-not $Promote) {
    Write-Host 'Production was not modified. Re-run with -CandidateDir and -Promote to activate this exact candidate.'
}
