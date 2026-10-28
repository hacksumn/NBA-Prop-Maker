[CmdletBinding()]
param(
    [string]$BranchName = 'release/dec059-operator-safe',
    [string]$CommitMessage = 'DEC-059: guarded operations and audited release boundary'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$GitDirectory = (Resolve-Path -LiteralPath '.git').Path
$ReleaseIndex = Join-Path $GitDirectory ("index.dec059_release_" + [guid]::NewGuid().ToString('N'))
$LegacyReleaseIndex = Join-Path $GitDirectory 'index.dec059_release_20261025'
$MainIndex = Join-Path $GitDirectory 'index'
if (-not $ReleaseIndex.StartsWith($GitDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Temporary release index escaped the repository .git directory.'
}
if (Test-Path -LiteralPath $ReleaseIndex) {
    throw "Temporary release index already exists: $ReleaseIndex"
}

$HeadBefore = (& git rev-parse HEAD).Trim()
$MainIndexHashBefore = (Get-FileHash -LiteralPath $MainIndex -Algorithm SHA256).Hash
$ReleaseFiles = @(
    '.gitignore',
    'README.md',
    'ARCHITECTURE.md',
    'CHANGELOG.md',
    'CLAUDE.md',
    'CONTEXT.md',
    'PROJECT_STATE.md',
    'feature_cols.json',
    'feature_cols_advanced.json',
    'run_morning.bat',
    'run_gui.bat',
    'output/AGENTS.md',
    'output/ARCHITECTURE.md',
    'output/STATUS.md',
    'output/TASKS.md',
    'output/DECISIONS.md'
)
$ReleaseFiles += Get-ChildItem -LiteralPath $ProjectRoot -File -Filter '*.py' |
    ForEach-Object { $_.Name }
$ReleaseFiles += Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'scripts') -File |
    Where-Object { $_.Extension -in '.py', '.ps1' } |
    ForEach-Object { 'scripts/' + $_.Name }
$ReleaseFiles += Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'tests') -File -Filter '*.py' |
    ForEach-Object { 'tests/' + $_.Name }
$ReleaseFiles += Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'models') -File -Filter '*.json' |
    ForEach-Object { 'models/' + $_.Name }
$ReleaseFiles = $ReleaseFiles | Sort-Object -Unique

$env:GIT_INDEX_FILE = $ReleaseIndex
try {
    & git read-tree HEAD
    if ($LASTEXITCODE -ne 0) { throw 'git read-tree failed' }

    & git add -f -- $ReleaseFiles
    if ($LASTEXITCODE -ne 0) { throw 'git add to the isolated release index failed' }

    $Tree = (& git write-tree).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'git write-tree failed' }

    $Commit = ($CommitMessage | & git commit-tree $Tree -p $HeadBefore).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Commit)) {
        throw 'git commit-tree failed'
    }

    & git branch -f $BranchName $Commit
    if ($LASTEXITCODE -ne 0) { throw 'release branch creation failed' }
}
finally {
    Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $ReleaseIndex) {
        Remove-Item -LiteralPath $ReleaseIndex -Force
    }
    if (Test-Path -LiteralPath $LegacyReleaseIndex) {
        Remove-Item -LiteralPath $LegacyReleaseIndex -Force
    }
}

$HeadAfter = (& git rev-parse HEAD).Trim()
$MainIndexHashAfter = (Get-FileHash -LiteralPath $MainIndex -Algorithm SHA256).Hash
if ($HeadAfter -ne $HeadBefore) {
    throw 'The active branch HEAD changed unexpectedly.'
}
if ($MainIndexHashAfter -ne $MainIndexHashBefore) {
    throw 'The user Git index changed unexpectedly.'
}

[ordered]@{
    release_branch = $BranchName
    release_commit = $Commit
    active_head_unchanged = $HeadAfter
    user_index_unchanged = $true
    release_file_count = $ReleaseFiles.Count
    temporary_index_cleaned = -not (Test-Path -LiteralPath $ReleaseIndex)
} | ConvertTo-Json
