# Pipeline data retention

Daily runs retain cumulative pick, candidate, slip, tracking and line histories. Working datasets and named latest outputs receive version retention whenever protected Python processes write, replace or remove them.

## Version archive

`.retained_versions/` holds compressed, SHA-256-addressed file contents. Identical bytes share one object. Path records associate each version with its original filename, byte length and preservation time. Checkpoint manifests record path-to-version mappings. Retention has no automatic expiry or pruning.

The protected paths are `data/`, `output/`, `models/`, `logs/`, `feature_cols.json` and `feature_cols_advanced.json` within this workspace. The archive includes previous grades, candidate promotions, rebuilt feature tables, injury/status snapshots, projections, reports, prices and model versions. It preserves the bytes available when enabled and subsequent revisions made through protected processes.

The baseline captured 13,141 existing files. Verification decompressed and checked the hashes of 12,569 unique objects. The compressed archive initially occupied approximately 1.3 GB. Future storage use depends on changes to the saved datasets.

## Write protection

`artifact_retention.py` installs a project-scoped Python audit hook before pipeline writes. It saves existing file bytes before write access and deletion. Before atomic replacement, it retains both the old destination and the staged replacement. If preservation raises an error, that mutation is blocked. Newly written files are also checkpointed on normal process exit.

The daily runner, prediction CLI, feature builder, source scrapers, model-layer modules and training entry points enable retention. Scheduled and manual operator wrappers launch Python scripts through `artifact_retention.py --run`. Training-maintenance and daily child processes use the same wrapper. Watcher logs are retained before PowerShell opens redirected output files.

Cumulative-history append operations use a cross-process lock around reading, merging and saving. Existing additive merge behavior retains earlier rows and columns. Run snapshot timestamps now include microseconds. Conflicting legacy archive filenames get a unique destination.

## Inspect and restore

List saved versions of a file from the project directory.

```powershell
python artifact_retention.py --list-versions output/picks_history.csv
```

Restore a listed SHA-256 version to a new file.

```powershell
python artifact_retention.py --restore HASH_FROM_LIST --output recovered_picks.csv
```

Verify compressed objects or capture a new complete checkpoint.

```powershell
python artifact_retention.py --verify
python artifact_retention.py --snapshot
```

For another standalone Python writer, use the protected launcher.

```powershell
python artifact_retention.py --run scripts/example_writer.py
```

External applications and separately running processes that started before installation require their own protected launch. The store is local to this workspace. Retention does not reconstruct files already absent when it was enabled.

## Verification

The full suite passed 256 tests and 10 subtests. Added regressions cover direct overwrite, atomic replacement, append, deletion, duplicate-content reuse, retention failure blocking a write, and overlapping cumulative-history appends. A saved pick-history version was restored into a temporary directory and matched the current source byte for byte. Both modified PowerShell launchers parsed successfully. The protected launcher was smoke-tested without provider requests or model training.
