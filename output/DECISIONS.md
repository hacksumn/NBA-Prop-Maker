# DECISIONS.md

# Architecture / Modeling Decision Log

---

## [DEC-001] Use chronological splits instead of random splits
- Date: (pre-April 2026)
- Status: Accepted
- Decision owner: Jake

### Context
Random splits inflate performance for time-series sports prediction tasks because future information can bleed into training distribution.

### Decision
Use chronological train/validation/test separation for all production-facing experiments. Most recent season is held out as test.

### Why
Better matches real deployment conditions and reduces optimistic bias.

### Consequences
- Metrics may look worse initially.
- Results are more trustworthy and forward-valid.

### Alternatives Considered
- Random split
- K-fold CV without time awareness

---

## [DEC-002] Shift all rolling features by one prediction step
- Date: (pre-April 2026)
- Status: Accepted
- Decision owner: Jake

### Context
Rolling features can accidentally incorporate same-game or future information if computed naïvely.

### Decision
All rolling player/team stats used for prediction must be shifted so they only reflect information available before the prediction timestamp (game tip-off).

### Why
Prevents leakage and preserves causal ordering.

### Consequences
- Feature generation is slightly more complex.
- Forward validity confidence increases.

### Alternatives Considered
- Keep unshifted rolling features
- Shift only selected features

---

## [DEC-003] Record dataset snapshots for every training run
- Date: (pre-April 2026)
- Status: Accepted
- Decision owner: Jake

### Context
Without dataset versioning, results are hard to reproduce and compare.

### Decision
Every training/evaluation run stores metrics in `models/training_results.json` and `models/training_edge_analysis.json`. These files drive the live quality policy.

### Why
Reproducibility and debugging. Quality policy is dynamic — it reads from these files at prediction time.

### Alternatives Considered
- Ad hoc notes
- Static quality config file

---

## [DEC-004] Layer 4 luck scores always rebuilt on every run
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`step7_luck_model()` had a 24-hour staleness check: if `player_luck_scores.csv` was less than 24 hours old, it skipped the rebuild and logged "Luck scores are fresh (16.3h old) — skipping rebuild." This meant luck scores from yesterday morning were applied to today's projections — stale during end-of-season/playoff race when team pace, rest patterns, and game scripts change dramatically night-to-night.

### Decision
Remove the staleness check. Always rebuild luck scores every run. The 15-second API cooldown sleep (after Layer 3's 30+ API calls) is retained.

### Why
Luck scores are derived from the current season's aggregate stats via 3 NBA API calls (`LeagueDashPlayerStats` Base, Advanced, and `LeagueDashPlayerShotLocations`). After Step 1 fetches new box scores, luck scores computed before Step 1 are stale by definition. The computational cost is 3 API calls — acceptable for a once-daily pipeline.

### Consequences
- Each run makes 3 additional NBA API calls for luck stats. Acceptable.
- The 15s cooldown prevents rate-limit errors from back-to-back Layer 3 + Layer 4 API use.
- Morning run will take ~30s longer for luck score computation.

### Alternatives Considered
- Tighten threshold to 6 hours (rejected — arbitrary; same problem on any daily pipeline)
- Cache luck scores in `nba_data.csv` directly (rejected — different granularity and update cadence)

---

## [DEC-005] Lower blowout baseline GP threshold from 15 to 5
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`step3_blowout_index()` in `run_daily.py` filtered the active player list to `GP >= 15` before building blowout-corrected baselines. This excluded 42% of players in the game log — specifically injured returners, callups, and rotation players with 5–14 games. Those are the players most likely to have blowout-distorted numbers in their limited sample. Log showed 467/801 players covered.

### Decision
Lower the minimum games played threshold to 5 in both `run_daily.py` (daily pipeline) and `blowout_discount_model.py` (standalone `--build` mode). The constant is named `MIN_GP_FOR_BASELINE = 5` with a comment explaining it and pointing back to the other file.

### Why
A player with 5 games has enough sample for the blowout tag to be meaningful. The 15 GP cutoff was protecting against a noise concern that doesn't apply here — blowout correction is applied as a weighted average, so thin samples get mild correction rather than noise amplification. The real cost of no correction is garbage-time inflation baked into projections.

### Consequences
- More players qualify (~150–200 additional). Each requires one NBA API call during baseline build. At 1.5s/call, this adds ~4 minutes to Step 3 runtime.
- The 12-hour staleness check on `CLEAN_BASELINES` limits this cost to once per day.
- Coverage log now shows `N/M qualifying players (X% coverage)` after each build.

### Alternatives Considered
- `GP >= 10` (rejected — still excludes callups who played 2-3 weeks of games)
- `GP >= 1` (rejected — single-game appearances are pure noise)
- Build baselines only for players with active props today (rejected — creates chicken-and-egg dependency with Step 9)

---

## [DEC-006] Add name-based fallback to _merge_layer_features()
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`_merge_layer_features()` joins the enriched feature CSV onto the game log using `(player_id, season)`. 17.6% of players (122 of 691) failed to match, silently receiving NaN for all 34 enriched columns. The log printed a single summary count with no player names. Root causes: new callups/rookies not yet in enriched CSV, player_id format drift, diacritic name encoding differences.

### Decision
After the primary player_id join, add a name-based second pass for still-unmatched rows:
- Normalize player names (NFKD → ASCII → lowercase → strip) on both sides
- Join unmatched rows on `(_pnorm, season)` against the same enriched data
- Fill recovered values back into `merged`
- Log exactly which players remain unmatched after both passes

### Why
A player with no enriched features gets predictions based on rolling averages only — weaker model. If that player has active props today, the prediction quality is degraded without any signal to the operator. The name fallback recovers diacritic mismatches (e.g. "Luka Dončić" vs "Luka Doncic") with zero risk of cross-player contamination.

### Consequences
- Some of the 122 unmatched players will be recovered via name matching.
- Remaining unmatched players are now visible by name in the log.
- Players genuinely absent from the enriched CSV (callups, G-League) will remain unmatched — the fix doesn't solve the data gap, only makes it visible and recovers what it can.

### Alternatives Considered
- Fuzzy name matching (rejected — risk of cross-player contamination, e.g. "Marcus Morris" vs "Markieff Morris")
- Rebuild enriched CSV daily (rejected — takes hours; right solution is incremental update, not in scope now)

---

## [DEC-007] Add volume-fill second pass to filter_best_picks()
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
Primary pick filter uses 6+ stacked gates (min_edge, meta_prob, regime checks, market edge, etc.). With live quality policy allowing only TRB/AST/BLK UNDER, primary pass produces ~3 picks on most days — well below the 5-pick minimum and insufficient for 3-pick betslip construction.

### Decision
After the primary pass, add a volume-fill second pass that relaxes thresholds to 65% of `min_edge_under`, requires `dir_prob ≥ 0.53`, skips meta gate and regime gate, and caps confidence at 63%. Fill picks are tagged `pick_source='volume_fill'`. Target pool size is `max(10, max_picks)`.

### Why
Without volume fill, many mornings produce no betslips at all. Volume-fill picks have lower confidence but still have directional signal (53%+ probability). Capping confidence at 63% ensures they are never mistaken for high-quality picks.

### Consequences
- More picks available for betslip construction.
- Volume-fill picks must be excluded from betslips (rule: `pick_source != 'volume_fill'` for betslip candidates). This filter is not yet implemented — it's a follow-up task.
- `pick_source` field must be present in all pick dicts.

### Alternatives Considered
- Loosen primary filter gates globally (rejected — degrades primary pick quality)
- Accept 0-pick mornings (rejected — defeats system purpose)
- Raise `max_picks` limit (rejected — doesn't address missing picks at low-confidence range)

---

## [DEC-008] Add line_source field to picks and picks_history.csv
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`fetch_vegas_lines()` tries Odds API first (when ≥50 credits remain), then falls back to PrizePicks. PrizePicks lines and Odds API lines can differ for the same player/prop. If a pick is graded against PrizePicks results but was generated against an Odds API line, the pick direction may be correct against the wrong number.

### Decision
- `fetch_vegas_lines()` now returns `(DataFrame, source_label)` tuple where `source_label` is `'odds_api'` or `'prizepicks'`.
- `save_lines_snapshot()` accepts `source` param and writes it to `historical_lines.csv`.
- `main()` in `nba_props.py` tracks `_line_source` and stamps all picks with `line_source` field.
- A `LINE SOURCE WARNING` is printed to log whenever `line_source != 'prizepicks'`.

### Why
Historical performance analysis is only meaningful if line source is known. Picks generated against Odds API lines cannot be fairly compared to PrizePicks graded results without this flag.

### Consequences
- `picks_history.csv` gains a `line_source` column. Existing rows will have null/missing — acceptable.
- Operator must check log for LINE SOURCE WARNING before placing bets.

### Alternatives Considered
- Always use PrizePicks lines (rejected — Odds API is the reliable fallback when PP is 403'd)
- Silently annotate without warning (rejected — too easy to miss)

---

## [DEC-009] Wire Playwright as 3rd fallback in prizepicks_scraper.py
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
Strategy A (Playwright with real Chrome profile) was implemented in `prizepicks_scraper.py` but never called. The fallback chain was: Strategy 0 (direct API) → Strategy B (cookie). Strategy A existed as dead code.

### Decision
Add Strategy A as the third fallback: direct API → cookie → Playwright. Playwright uses the real Chrome profile path from `PRIZEPICKS_CHROME_PROFILE` env var and may open a browser window (~30s).

### Why
Without Playwright, any day PrizePicks blocks the direct API and the cookie is stale results in Odds API lines (or no lines). Playwright has the best chance of bypassing Cloudflare by using a real authenticated session.

### Consequences
- Morning run may pause ~30s waiting for Playwright. Acceptable.
- If `PRIZEPICKS_CHROME_PROFILE` is not set or the profile is locked by another Chrome instance, Playwright will still fail silently (falls through to Odds API).
- Long-term: need automated cookie refresh or persistent authenticated session.

### Alternatives Considered
- Keep Playwright as manual-only (rejected — too easy to forget)
- Use puppeteer/selenium instead (rejected — already have Playwright installed)

---

## [DEC-010] Step 8 absence suppression — filter 0/5 from top-10 display, tag 1-2/5
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`step8_merge_projections()` in `run_daily.py` displays the top-10 projected scorers to help the operator assess today's slate. Before this fix, absent/injured players (0 recent games out of 5) dominated the list because projections are based on L20 rolling averages that don't decay for absences.

### Decision
After the main merge, read `data/detected_absences.csv` (output of Step 6 `detect_recent_absences()`). Merge `recent_gp` onto `proj` as `absence_recent_gp`. In the top-10 display loop:
- Skip any player with `absence_recent_gp == 0` (fully absent last 5 games).
- Tag any player with `absence_recent_gp <= 2` with `[ABSENCE RISK: N/5 games]`.
- Log suppressed players in a separate line below the top-10 table.

### Why
Showing Luka Dončić (out 5/5 games) as the #1 projected scorer is misleading and wastes operator attention. The display should reflect active players.

### Consequences
- Top-10 list is now "active players only" — more actionable.
- Suppression is display-only; the projection itself still exists in `proj` and is used downstream by `nba_props.py`.
- Join is on `player_id`; if `detected_absences.csv` uses different ID format, absence data silently doesn't attach (handled gracefully with `absence_recent_gp = NaN`).

### Alternatives Considered
- Filter absent players from the full `proj` DataFrame (rejected — they may still have valid lines for pick generation)
- Use a different absence threshold (considered 1/5 as the filter cutoff — rejected, 1/5 is "possible return" not "fully absent")

---

## [DEC-011] Step 8 dedup strategy — drop_duplicates on bl before merge, proj after all merges
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`step8_merge_projections()` merges four layers (PPP, usage, luck, blowout) onto a per-player L20 rolling average DataFrame (`proj`). The blowout baseline CSV (`bl`) had duplicate player entries, causing a many-to-one merge fan-out (one player row became multiple rows). Luka Dončić appeared twice in the top-10 as a result.

### Decision
Two-point fix:
1. `bl = bl.drop_duplicates(subset=["player"], keep="first")` — dedup blowout baselines before merge.
2. After all four layer merges: `proj = proj.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)` — global dedup catch-all.

### Why
The merge-time fix is the right place to handle this because the root cause (duplicate rows in source CSVs) has not been audited upstream. The global dedup after all merges is a safety net in case other source CSVs have the same issue.

### Consequences
- Downstream count log now correctly reports unique player count after dedup.
- `keep="first"` preserves the first occurrence — acceptable because duplicates have identical player stats (same player, same rolling average).
- Root cause (duplicate rows in blowout baseline) is still not fixed at source — flagged as a follow-up task.

### Alternatives Considered
- Fix at source in `blowout_discount_model.py` (preferred long-term, but more invasive)
- Use `groupby().first()` instead of `drop_duplicates` (rejected — same effect, more verbose)

---

## [DEC-012] Suppress luck-driven projection fields for absence-risk players in Step 8
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`step8_merge_projections()` merges Layer 4 luck outputs (`luck_label`, `total_luck_score`, `pts_luck_adj`, related luck metrics) before it merges `detected_absences.csv`. That allowed players with recent-absence signals to retain "Lucky" / "VERY LUCKY" tags and luck-adjusted point projections in `data/player_projections_today.csv`. This was especially misleading for players already flagged by the local absence scan (0–2 games played in the last 5).

### Decision
After `detected_absences.csv` is merged in Step 8, clear all luck-derived projection fields for any player with `absence_recent_gp <= 2` before computing `proj_pts_luck_adj` / `proj_pts_final` and before saving `data/player_projections_today.csv`.

### Why
The standalone luck model is historical and availability-agnostic by design. The live projection artifact is different: once the pipeline already has an absence-risk signal, presenting a strong regression-to-mean luck tag as if the player were a normal active projection is misleading. Clearing these fields makes the output availability-aware without altering the Layer 4 model itself.

### Consequences
- `data/player_projections_today.csv` now drops luck labels and luck-based point adjustments for all players already flagged by the absence scan.
- Absence-risk players fall back to the non-luck projection path for `proj_pts_final`.
- `data/player_luck_scores.csv` remains unchanged; this is a projection-assembly rule, not a model change.
- Players completely missed by Step 6 still retain luck fields, so this is not a substitute for a real live injury feed.

### Alternatives Considered
- Suppress only `recent_gp == 0` players (rejected — Luka-type `2/5` cases were still misleading, and the pipeline already treats `<=2/5` as absence risk)
- Change `luck_regression_model.py` itself (rejected — the historical artifact should remain availability-agnostic)
- Remove absent players from the full projection table (rejected — pick generation and diagnostics may still need the row even when luck fields are suppressed)

---

## [DEC-013] Replace the hard-coded live first-principles blend with adaptive per-player weights
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
Live inference in `nba_props.py` blended `player_projections_today.csv` into model predictions with a fixed fallback of `35% FP / 65% XGB` whenever `models/fp_blend_weights_advanced.json` was absent. That meant every player received the same FP/model mix regardless of sample depth, rookie vs veteran history, role stability, or whether the FP row had clean-baseline support.

### Decision
Keep the existing global stat priors when available, but convert the live blend itself to a per-player adaptive rule. The applied FP weight now moves using only leak-safe current-row signals already present in the live feature frame:
- total sample depth (`games_played`)
- current-season share of the player history (`season_games_played / games_played`)
- role stability (`mp_cv`, `mp_trend_pct`)
- current form drift vs season baseline (`{stat}_form_vs_season`, `{stat}_best_estimate`)
- whether the FP row has clean baseline coverage (`clean_pts`, `clean_reb`, `clean_ast`)
- recent absence risk from the Step 8 projection artifact (`absence_recent_gp`)

### Why
The fixed `35/65` rule was clearly wrong for players at opposite ends of the information spectrum. A player with deep multi-season history and a stable role should not get the same FP weight as a current-season-only player whose role is still evolving. The adaptive rule keeps the fix local to live inference, uses only already-available signals, and avoids waiting for a new training artifact.

### Consequences
- The live log no longer reports a universal `35/65` blend.
- FP weights now vary by player and by stat, even when no learned weight JSON exists.
- Example verification from the live feature frame: LeBron points blend ≈ `72/28 model/FP`, Jeremiah Fears points blend ≈ `59/41 model/FP`.
- This is still a heuristic gating rule, not a fully learned per-player meta-model. It should eventually be replaced or calibrated with offline evidence if the training pipeline starts emitting player-level blend policies.

### Alternatives Considered
- Leave the fixed fallback in place until a learned JSON exists (rejected — current live behavior was already misleading)
- Train a new per-player blend model immediately (rejected for this bugfix — larger scope, new artifacts, and more verification burden)
- Remove the live FP blend entirely (rejected — FP still carries orthogonal signal and is useful when weighted sanely)

---

## [DEC-014] Exclude volume_fill picks from betslip construction; restrict UNDER-Only pool to AST/TRB
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`log_betslips()` sorted picks by confidence and took the top 4 for slip construction without checking `pick_source`. On a thin-pick day, a volume_fill pick (confidence capped at 63%, relaxed thresholds) could enter a real-money parlay. ARCHITECTURE.md documented this as a required rule but the code did not enforce it.

The UNDER-Only betslip pool was set to `{'AST', 'TRB', 'STL', 'BLK'}`. STL is disabled entirely by the quality policy (not in picks). BLK has Meta AUC 0.846 but only 33% raw O/U accuracy — unreliable for parlay legs even though individual BLK UNDER picks are enabled.

### Decision
1. Before deduplication in `log_betslips()`, filter `eligible_df = picks_df[pick_source != 'volume_fill']`. All slip construction uses `eligible_df` instead of `picks_df`.
2. Change `_under_props` to `{'AST', 'TRB'}`: remove STL (dead code) and BLK (33% raw O/U makes it unsuitable for parlay legs).

### Why
volume_fill picks are a fill mechanism for informational pick volume. Allowing them into betslips would bet real money on picks generated with relaxed thresholds. The 63% confidence cap is explicitly below the intended betslip quality bar.

BLK's 33% raw O/U means it loses money more often than random in straight-up bets. Its market signal (Meta AUC) does not translate to parlay reliability.

### Consequences
- On thin-pick days (few market_model picks), betslips may have fewer than 2 eligible picks and won't be built. This is correct behavior — better no betslip than a bad one.
- STL/BLK picks remain visible in picks_latest.csv for informational review.
- `eligible_df` is reused for both the Power Play track and the UNDER-Only track.

### Alternatives Considered
- Keep volume_fill in betslips but add a warning label (rejected — warning labels don't prevent real-money exposure)
- Lower confidence threshold instead of pick_source filter (rejected — pick_source carries more information about the quality of the selection process)

---

## [DEC-015] Fix market_edge_pred confidence contribution and add extreme dampening gate
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`filter_best_picks()` had two bugs related to `market_edge_pred` (edge model output, trained on `y_edge = actual - line`):

1. **Always-positive confidence contribution**: `base_conf += min(0.08, abs(market_edge_pred) / rmse * 0.08)` added confidence regardless of whether the edge model was amplifying or dampening the raw signal. When edge model sees less downside than raw model (dampening), confidence should not increase.

2. **No dampening gate**: The existing sign-disagreement gate (`np.sign(model_edge) != np.sign(market_edge_pred)`) only rejected picks when signals pointed in opposite directions. No gate existed for same-direction dampening — e.g., model_edge=-1.7, market_edge_pred=-0.63 (edge model only 37% as bearish as raw model). Such picks pass the gate but the edge model is providing no independent market-context conviction.

Today's picks showed this concretely: Devin Carter (model=-1.7, market=-0.63) passed all gates despite the edge model being 63% less bearish than the raw signal.

### Decision
1. **Confidence adjustment**: Change the market_edge_pred contribution to be direction-aware:
   - Signs disagree: subtract confidence (`base_conf -= min(0.03, ...)`)
   - Same direction AND magnitude >= 90% of model_edge: add bonus (original behavior)
   - Same direction but dampening: no contribution (was incorrectly adding)

2. **Extreme dampening gate**: Reject pick when `abs(market_edge_pred) < abs(model_edge) * 0.40` (>60% dampening) AND `abs(combined_edge) < min_edge * 1.15` (edge barely above threshold).

### Why
`market_edge_pred` is trained on market-context features and is meant to provide independent conviction about predicted deviation from line. When it produces a number far weaker than `model_edge` in the same direction, the market-context signal is either noise or actively warning against the pick. Treating dampening the same as amplification inflates confidence.

### Consequences
- Picks where the edge model is significantly less bearish than the raw model get lower or no confidence boost
- Extreme dampening cases near the minimum edge threshold are now filtered
- Expected effect: Carter-type picks (strong raw model, very weak edge model) may not appear in future runs
- Fears/Garland-type picks (edge model amplifies or matches raw) are unaffected

### Alternatives Considered
- Hard gate on absolute minimum |market_edge_pred| threshold (e.g., >= 0.50): rejected as too blunt — would filter any pick where edge model is only mildly less bearish
- Subtract confidence on all dampening cases (not just zero out): rejected as too aggressive for mild dampening; reserved for sign-disagreement case

---

## [DEC-016] Filter Step 6 absence candidates by latest local team context
- Date: 2026-04-10
- Status: Accepted
- Decision owner: Jake

### Context
`detect_recent_absences()` in `usage_injury_model.py` built each team's candidate roster directly from `player_profiles.csv` using `team_abbr`. That profile file can carry stale roster assignments after trades or data drift. The function already merged `latest_played_team` from local `data/nba_data.csv`, but only used it for logging.

That created ghost absences. On the 2026-04-10 audit, Washington's scan flagged Anthony Davis and D'Angelo Russell as WAS absences even though their last local games were for DAL on 2026-01-08 and 2026-01-10. Those false positives could trigger incorrect teammate injury adjustments downstream.

### Decision
Keep `player_profiles.csv` as the official season-profile artifact, but treat `latest_played_team` from local game logs as a hard exclusion rule for Step 6 absence inference. If a player's profile team is `WAS` but their latest local game was for `DAL`, exclude them from WAS's absence candidates and log them as a skipped stale-roster player.

After the code fix, regenerate `data/detected_absences.csv` from the current `player_profiles.csv` snapshot so the live artifact matches the new rule.

### Why
Step 6 is an injury/availability proxy for current team rotation effects. It should only reason over players who have actual local evidence of belonging to that team's recent playing population. Logging the mismatch without filtering it was not enough because the false positive still propagated into `detected_absences.csv`.

This keeps the fix narrow and leak-safe:
- it uses only historical local game-log context already on disk
- it does not overwrite the official season profile team globally
- it prevents stale roster contamination without inventing new roster data

### Consequences
- `data/detected_absences.csv` no longer contains rows where `team_abbr != latest_played_team`.
- The specific WAS ghost cases for Anthony Davis and D'Angelo Russell are removed.
- Newly traded players who have not yet appeared in a local game for their new team will not be treated as injury absences for that team. That is correct for this proxy layer.
- The broader lack of a live injury feed remains unresolved.

### Alternatives Considered
- Keep the mismatch as a log-only warning: rejected because the false positive still contaminates absence outputs.
- Overwrite `team_abbr` in `player_profiles.csv` globally with `latest_played_team`: rejected because the official profile artifact is used elsewhere and local logs are context, not universal truth.
- Add a date-threshold heuristic on mismatches: rejected because any active mismatch is already sufficient to make the absence inference unsafe for the current team.

---

## [DEC-017] Use local `nba_data.csv` as the primary Step 3 player-log source
- Date: 2026-04-11
- Status: Accepted
- Decision owner: Jake

### Context
Step 3 clean-baseline rebuilds were slow because `build_clean_baselines()` called `PlayerGameLog` once per player and slept between requests. With `NBA_PLAYER_LOG_DELAY=1.5` and 500+ qualifying players, the delay budget alone was on the order of 12-14 minutes, which dominated the morning pipeline wall clock.

At the same time, Step 1 already refreshes `data/nba_data.csv` before Step 3 runs. That local box-score table contains the same per-player per-game fields that the blowout cleaner needs: `GAME_ID`, `game_date`, `matchup`, result, minutes, raw box score stats, and `plus_minus`.

### Decision
Refactor `blowout_discount_model.py` so Step 3 uses local `data/nba_data.csv` as the primary per-player game-log source for clean baseline builds. Keep the existing `PlayerGameLog` API path as a fallback only for players missing from the local cache.

Also normalize `game_id` formatting before blowout-tier lookup so local CSV game IDs (no leading zero padding) and API game IDs resolve to the same blowout index rows.

### Why
This removes the dominant runtime cost without changing the output schema or introducing a new data dependency. It is also safer than aggressive parallel API fan-out because:
- Step 1 already guarantees the local file is populated before Step 3
- local reads are deterministic and effectively free
- the API fallback remains available for any missing-player edge case

The isolated post-change benchmark on 2026-04-11 built `544` clean baselines in `3.7s`, with `544` local cache hits and `0` API fallbacks.

### Consequences
- Step 3 no longer incurs the per-player throttle cost when local data is present.
- `parse_minutes()` now accepts numeric minute values as well as `MM:SS` strings, which is required for local `mp` values.
- Blowout-tier lookup is now robust to game-ID formatting differences between CSVs and NBA API payloads.
- Full end-to-end pipeline wall clock was reverified on 2026-04-11 at `478.1s`, so the old 18.5-minute figure is no longer the active runtime assumption.

### Alternatives Considered
- Lower the API sleep without changing the data source: rejected because it still scales linearly with player count and risks rate-limit instability.
- Parallelize `PlayerGameLog` requests: rejected because it increases API burst risk and adds more operational fragility than a local-cache path.
- Derive the entire game blowout index from local data too: rejected for now because the single league-game-log API call was not the bottleneck.

---

## [DEC-018] Normalize first-principles blend inputs in the shared prediction helper
- Date: 2026-04-11
- Status: Accepted
- Decision owner: Jake

### Context
`logs/run_20260411.log` showed a hard Step 9 failure inside `nba_props.py:_predict()` during live pick generation. The primitive ridge-blend path expected `fp_input` to behave like a pandas object and called `.fillna()` directly:

```python
pd.to_numeric(fp_input, errors='coerce').fillna(0.0)
```

That assumption was false in the live one-row prediction path. `generate_predictions()` passed `fp_input=np.array([_row_fp_projection(...)])`, so `_predict()` received a NumPy array and crashed with:

```text
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'
```

This was not a model-quality problem. It was a shared prediction-helper contract problem between the vectorized batch path and the one-row live path.

### Decision
Keep the fix in the shared `_predict()` helper rather than changing each caller. `_predict()` now:
- accepts `fp_input` as a pandas Series, Python list, or NumPy array
- coerces it to a numeric NumPy array
- zero-fills NaN and non-finite values
- broadcasts a singleton FP value to `len(X)` when the feature frame has one or more rows
- raises an explicit `ValueError` only when a real length mismatch exists

### Why
The bug came from a type assumption in shared inference code, so the safest fix is to harden that shared boundary. Caller-specific patches would leave the same latent crash risk in other paths such as backtests or future probability sidecars.

This also preserves model logic. The blend weights, base-model predictions, and FP projections are unchanged. Only the input coercion layer changed.

### Consequences
- `python nba_props.py predict` now completes successfully on the current 2026-04-11 slate.
- `python run_daily.py` now completes end-to-end; `logs/run_20260411.log` ends with `[OK] Today's pick generation complete`, `Pipeline complete in 478.1s`, and `All steps completed successfully`.
- True FP/input shape mismatches will now fail with a clear error message instead of an ambiguous attribute error.
- The fix is scoped to inference and does not require retraining.

### Alternatives Considered
- Convert `fp_input` to a pandas Series only at the live caller: rejected because the batch prediction path also uses `_predict()` and should share one contract.
- Revert the primitive ridge-blend integration: rejected because that would throw away a correct feature path to avoid a narrow input-type bug.
- Silence the exception and skip FP input on error: rejected because that would create silent prediction drift and make debugging harder.

---

## [DEC-019] Hard-gate betslips at 70 confidence and restore STL/BLK under-only legs
- Date: 2026-04-11
- Status: Accepted
- Decision owner: Jake

### Context
Two open policy items were still unresolved in the live betslip builder:

1. `log_betslips()` was excluding `volume_fill`, but it was still willing to use any remaining pick regardless of confidence. That left open the possibility of future low-conviction market-model picks entering real-money slips.
2. The dedicated under-only pool was still restricted to `AST` and `TRB`, even though the retrained quality policy now enables `STL UNDER` and `BLK UNDER`.

Current retrained evidence:
- `AST UNDER`: `65.4%` hit, `n=2172`
- `TRB UNDER`: `63.1%` hit, `n=2278`
- `STL UNDER`: `74.0%` hit, `n=2034`, `clv_corr=0.324`
- `BLK UNDER`: `79.2%` hit, `n=1756`, `clv_corr=0.256`

The old exclusion of STL/BLK was based on stale or blended O/U reasoning that no longer matched the post-retrain under-only evidence.

### Decision
1. Add a hard `confidence >= 70` eligibility gate inside `log_betslips()`.
2. Keep excluding `volume_fill` picks.
3. Expand the dedicated under-only betslip pool from `{'AST', 'TRB'}` to `{'AST', 'TRB', 'STL', 'BLK'}`.

### Why
The confidence gate solves the bankroll-policy problem directly: only high-conviction picks can enter slips.

Once that gate exists, STL/BLK no longer need to be excluded purely out of caution. They now have:
- strong retrained under-only hit rates
- positive CLV correlation
- live filtering plus the new hard confidence floor

This is a cleaner rule than trying to bolt on one-off exclusions for prop types that the live quality policy already approves.

### Consequences
- Low-confidence picks are now blocked from betslips even if they are not `volume_fill`.
- Latest verification shows the current live betslips have minimum leg confidence `75.2`.
- STL/BLK can now appear in under-only slips when they survive the live pick filter and clear the hard confidence gate.
- This does not prove bankroll superiority yet; the new mix still needs explicit backtesting / monitoring.

### Alternatives Considered
- Keep AST/TRB-only under slips: rejected because it ignored current retrained evidence.
- Use different confidence floors by prop type: rejected for now as unnecessary complexity without fresh slip-level evidence.
- Leave confidence gating to ranking order only: rejected because ranking alone is not a hard bankroll policy.

---

## [DEC-020] Persist the live injury feed in Step 6 and merge it in Step 8
- Date: 2026-04-11
- Status: Accepted
- Decision owner: Jake

### Context
The repo already had a combined live injury feed in `nba_props.py`:
- official NBA injury report PDF when available
- ESPN injury API as fallback / supplement

But that feed was only used late in the pipeline, inside `nba_props.py predict`.

The earlier daily artifacts still relied on the recent-games absence proxy alone:
- Step 6 wrote only `data/detected_absences.csv`
- Step 8 only merged `absence_recent_gp`
- `data/player_projections_today.csv` could still show live-inactive players as normal projections if the absence proxy missed them

That architectural split was the reason Step 8 still produced misleading live context even after the picker had access to better injury information.

### Decision
Use the existing live injury feed as a shared Step 6 / Step 8 artifact without changing the feed schema itself:

1. During Step 6, fetch the combined official NBA + ESPN injury feed and persist it to:
   - `data/live_injury_status.json`
   - `data/live_injury_status.csv`
2. During Step 8, merge the flattened live injury feed into `data/player_projections_today.csv`.
3. Use live injury buckets (`out`, `doubtful`, `questionable`, `day_to_day`) in the Step 8 luck-suppression rule alongside the absence proxy.
4. Suppress official `OUT` / `DOUBTFUL` players from the Step 8 top-10 display and tag softer live statuses like `QUESTIONABLE`, `DAY TO DAY`, and `PROBABLE`.

### Why
This is the smallest scoped change that fixes the architectural gap without inventing a second live injury system.

It also stays leak-safe:
- the feed is captured pregame during the morning run
- no post-tip information is introduced
- the absence proxy remains available as fallback when live status is unavailable

### Consequences
- `run_daily.py` is no longer relying on the recent-games absence proxy alone for Step 8 availability context.
- `data/player_projections_today.csv` now includes:
  - `live_injury_bucket`
  - `live_injury_status`
  - `live_injury_source`
  - `live_team_status_pending`
- Latest verification merged `188` player statuses into the projection artifact and suppressed luck adjustments for `267` players with either absence-risk or live injury flags.
- The feed is currently shared by importing `fetch_injury_data()` from `nba_props.py` inside `run_daily.py`. That is acceptable as a scoped fix, but it should be refactored into a dedicated shared module later.
- Pending official reports plus missing ESPN coverage can still leave some players unresolved. Kawhi remains the representative example.

### Alternatives Considered
- Leave the live injury feed inside `nba_props.py` only: rejected because Step 8 would remain stale / misleading.
- Overwrite `detected_absences.csv` with live injury statuses: rejected because the absence proxy and the live report solve different problems and should remain distinct artifacts.
- Fully refactor the feed into a new shared module first: rejected for this task because it widened scope beyond the immediate operational gap.

---

## [DEC-021] Adopt context-conditioned σ from quantile spread as the primary edge-filter signal
- Date: 2026-04-12
- Status: Accepted — implemented 2026-04-12
- Decision owner: Jake

### Context
The current `edge_z` filter ([nba_props.py:5941](../nba_props.py#L5941)) divides raw edge by `{stat}_std_l10` — a rolling historical standard deviation computed in the feature pipeline. This is a static, backward-looking proxy for uncertainty. It does not reflect the model's own distributional prediction for this specific game context.

The quantile sidecar already outputs q10/q25/q50/q75/q90 per prediction. These quantiles implicitly encode a full predictive distribution. The normalized IQR `σ = (q75 - q25) / 1.35` is an unbiased σ estimate from that distribution — and it IS context-conditioned, because the quantile models are trained on features including minutes variance (`mp_cv`, `mp_std_l10`), usage rate, matchup, pace, and form signals.

Diagnostic performed 2026-04-12 confirmed:
- σ is never explicitly extracted from quantile spread — the `{target}_sigma` column does not exist
- `edge_z` uses static `std_l10` regardless of how wide or narrow the model's predicted distribution is
- No heteroskedasticity analysis exists — it is not possible to see which contexts drive variance
- Distribution family is not matched to stat type — Poisson sidecar exists but routing is unverified for count stats

### Decision
Three-part upgrade (priority ordered):

1. **Extract and wire `{target}_sigma`**: After quantile prediction, compute `sigma = (q75 - q25) / 1.35` per row; expose as `{target}_sigma` in projections and replace the static `std_l10` lookup in the `edge_z` filter with this value.

2. **σ-context breakdown in calibration diagnostics**: After training, bucket σ by minutes tier, usage tier, pace tier, and matchup tier. Print and store with probability diagnostics so high-variance contexts are visible.

3. **Verify count-stat routing**: Confirm TRB and AST at low lines route through `poisson_probability_sidecar`. If not, add conditional routing in `train_advanced_models.py`.

### Why
Using the model's own predicted uncertainty as the edge-filter denominator is strictly more informative than a static rolling std. A high-σ game context (low minutes stability, difficult matchup) should require a larger raw edge to pass the `edge_z` threshold — and currently it does not. This is a direct path to fewer low-quality picks on high-uncertainty days.

### Consequences
- `{target}_sigma` becomes a new column in `data/player_projections_today.csv`
- The `edge_z` filter becomes distribution-aware — picks in high-σ contexts are harder to pass
- No model retraining required; σ is derived from existing quantile outputs
- The static `std_l10` lookup remains as a fallback when quantile outputs are absent

### Alternatives Considered
- Keep static `std_l10` as edge_z denominator (rejected — ignores the model's own distributional output)
- Train a separate σ prediction model (rejected for now — quantile IQR is already a direct σ estimate with no new artifacts required)
- Switch to NGBoost or other distributional regression (deferred — larger retraining scope; quantile regression already provides the distribution)

---

## [DEC-022] σ-tier confidence and priority bonus for low-σ UNDER picks
- Date: 2026-04-12
- Status: Accepted — implemented 2026-04-12
- Decision owner: Jake

### Context
The σ-context analysis (run after the first retrain with quantile sidecars) produced the following OOF hit rates by predicted σ quartile:

| Stat | σ bucket | Over hit rate | UNDER edge |
|---|---|---|---|
| AST | σ < 1.68 (p25) | 31.8% | +18.2 pp |
| AST | σ > 2.40 (p75) | 45.3% | +4.7 pp |
| TRB | σ < 2.10 (p25) | 37.1% | +12.9 pp |
| TRB | σ > 2.87 (p75) | 41.7% | +8.3 pp |
| PTS | all buckets | 45–49% | near-efficient |

Low-σ UNDER picks (tight predicted distribution, below p25 threshold) have 3–4× the UNDER edge of high-σ picks. The system was not distinguishing between these contexts in the pick selection layer — all UNDER picks of the same `combined_edge` were treated equally.

### Decision
Add a σ-tier bonus in `filter_best_picks()` (`nba_props.py`):

1. `_load_sigma_p25_thresholds()` reads `models/sigma_context_analysis.json` at the top of `filter_best_picks()`. Thresholds at time of implementation: `ast=1.68, trb=2.10, pts=5.58`.
2. `low_sigma_under = (direction == 'UNDER' and sigma_val < p25_threshold)` is set per pick.
3. `base_conf += 0.04` (+4 conf_pct points) applied for `low_sigma_under` picks.
4. `selection_priority += 0.50` stacks with the existing `priority_under_lane` bonus (which gives 1.00 for AST, 0.70 for TRB), putting low-σ UNDER picks at the very top of the card sort.
5. `sigma` and `low_sigma_under` columns added to all pick output paths for forward tracking.

### Why
The σ p25 threshold is the natural cutoff: it defines the bottom quartile of model-predicted uncertainty. Picks below that threshold are in the highest-predictability game contexts. The +0.04 / +0.50 magnitudes are conservative — matched to the scale of existing bonuses (anchor_agrees=+0.02, l10_agrees=+0.03, priority_lane=+0.03–0.045) so the signal contributes without overriding the probability and edge signals.

Thresholds are read from disk at runtime, so they automatically update when `calibrate_confidence.py` is re-run after a retrain without requiring code changes.

### Consequences
- Low-σ UNDER picks receive a small confidence boost and sort above equivalent non-σ picks.
- `picks_latest.csv` now carries `sigma` and `low_sigma_under` columns for forward validation.
- If σ-context hit rates shift over time (e.g. the book adjusts to close the mispricing), thresholds and bonus magnitudes can be updated via a calibration re-run without a code change.
- The bonus only applies when `{prop}_sigma` is populated (requires the quantile sidecar to have run). Falls back to no bonus silently when the column is absent.

### Alternatives Considered
- Hard gate: reject UNDER picks with σ > threshold (rejected — discards otherwise-valid picks without enough forward evidence yet)
- Larger bonus magnitude (±0.08+): rejected — too aggressive without per-pick forward validation data on σ-split performance
- Apply bonus to OVER direction too (rejected — σ-context data shows near-efficient PTS OVER market; no OVER mispricing evidence for AST/TRB)

---

## [DEC-023] σ-tiered dynamic `min_edge_under` gate in `filter_best_picks()`

- Date: 2026-04-12
- Status: Accepted — implemented 2026-04-12
- Decision owner: Jake

### Context
DEC-022 added a confidence/priority bonus for low-σ UNDER picks but left the `min_edge_under` gate static (loaded from the quality policy, same value regardless of distributional width). This meant a low-σ pick with a modest edge (e.g. combined_edge=0.75 vs AST policy floor of 0.90) was still rejected even though the σ-context data shows it is in the highest-predictability tier with a 68% UNDER hit rate.

Conversely, high-σ UNDER picks (σ > p75) were admitted at the same bar as low-σ picks despite having materially more real variance (UNDER edge shrinks to ~5–8 pp).

### Decision
In `filter_best_picks()`, after loading `min_edge` from the quality policy, apply a σ-tiered multiplier to UNDER picks when σ is available:

- **Low-σ UNDER** (σ < p25): `min_edge = max(0.50, min_edge * 0.75)` — lowers bar 25%, floors at 0.50
- **Mid-σ UNDER** (p25 ≤ σ ≤ p75): no change — policy value stands
- **High-σ UNDER** (σ > p75): `min_edge = min_edge * 1.25` — raises bar 25%

`_load_sigma_p25_thresholds()` was renamed to `_load_sigma_thresholds()` and expanded to return both `(p25_dict, p75_dict)`. Both dicts are loaded once at the top of `filter_best_picks()`.

Effective bars at current policy values:

| Target | Policy | Low-σ bar | High-σ bar |
|--------|--------|-----------|------------|
| AST    | 0.90   | 0.675     | 1.125      |
| TRB    | 1.20   | 0.900     | 1.500      |
| PTS    | 2.00   | 1.500     | 2.500      |

### Why
The 0.75/1.25 multipliers are symmetric and conservative — roughly matching the effect sizes in the σ-context data (low-σ UNDER has ~2× the edge of high-σ). The 0.50 floor prevents the low-σ bar from collapsing to noise for targets with very low policy floors. The adjustment is UNDER-only because OVER mispricing evidence is absent from the σ-context analysis.

PTS has near-zero σ signal (all buckets 45–49% hit rate) but the multiplier still applies — it is just less meaningful at PTS policy bars (2.0 → 1.5 / 2.5). No PTS harm risk since PTS OVER is already the primary allowed direction and PTS UNDER is rarely the target.

Adjustment is silently skipped when `{prop}_sigma` is absent (no quantile sidecar run yet) — same graceful fallback as DEC-022.

### Consequences

- Low-σ UNDER picks now pass the edge gate with ~25% less edge required. This will increase pick volume in tight, high-predictability game contexts.
- High-σ UNDER picks now require more edge. This will reduce pick volume in noisy game contexts — a desirable filter.
- Combined with DEC-022 (+conf, +priority), low-σ UNDER picks now rank higher AND are easier to qualify. Monitor `picks_latest.csv` after the next run for σ-tier distribution.
- If the book narrows the low-σ mispricing gap over time, a calibration re-run automatically updates both thresholds and the multiplier breakpoints without a code change.

### Alternatives Considered

- Continuous interpolation (linear ramp from p25 to p75): more principled but harder to audit; discrete tiers are transparent and debuggable
- Separate per-target multipliers (e.g. AST=0.70, TRB=0.80): deferred — insufficient forward data to justify per-target tuning yet; uniform 0.75 is a reasonable prior
- Apply σ-tiered adjustment to volume_fill path too: rejected — volume_fill already applies a blanket 0.65× reduction; double-adjusting would over-permissive for a confidence-capped fallback path

---

## [DEC-024] σ forward validation tracking in picks_history.csv and calibrate_confidence.py

- Date: 2026-04-12
- Status: Accepted — implemented 2026-04-12
- Decision owner: Jake

### Context
DEC-022 and DEC-023 added σ-tiered logic to pick selection (confidence bonus, priority bump, dynamic edge gate), but the σ value and the `low_sigma_under` flag were not guaranteed to persist into `picks_history.csv`. Without σ in history, there was no way to validate — after weeks of real picks — whether the low-σ mispricing edge (68% UNDER hit for AST, 63% for TRB in OOF data) holds up in production data. The OOF analysis uses training data; we need a forward-data counterpart.

### Decision
Two changes:

1. **`save_picks()` column guards** (`nba_props.py`): `sigma` (float, default `np.nan`) and `low_sigma_under` (bool, default `False`) added to the safety-net guard list in `save_picks()`. The primary guards are already in `filter_best_picks()`, but adding them in `save_picks()` ensures the columns survive even if picks arrive via a code path that doesn't pass through `filter_best_picks()`.

2. **`compute_sigma_forward_validation()` in `calibrate_confidence.py`**: Reads `picks_history.csv`, finds graded rows (`WIN`/`LOSS`) with `sigma` populated, buckets by `(prop, direction, σ tier)` using p25/p75 thresholds from `models/sigma_context_analysis.json`, and reports win rate, mean confidence, mean edge, and mean sigma per bucket. Output saved to `models/sigma_forward_validation.json`. Called from `main()` so it runs every time `calibrate_confidence.py` is executed after retraining.

Graceful degradation: if `picks_history.csv` has no `sigma` column yet (pre-DEC-022 runs), or has no graded rows with sigma, the function prints an informative message and returns `{}` without crashing.

### Why
The OOF σ-context analysis is a training-data signal — it measures whether the model's distributional predictions correlate with calibration quality on held-out folds. The forward validation function measures whether the actual real-money pick outcomes stratify by σ tier. These are different questions. The OOF result justifies turning on the σ logic; the forward result is how we know whether to keep it or adjust the multipliers.

Thresholds for tier boundaries come from `sigma_context_analysis.json`, so they auto-update on retrain without code changes.

### Consequences

- `picks_history.csv` will carry `sigma` and `low_sigma_under` for all picks generated after this session.
- After ~50+ graded σ-tagged picks accumulate, `calibrate_confidence.py` will print a stratified forward hit-rate table alongside the OOF calibration diagnostics.
- `models/sigma_forward_validation.json` gives a machine-readable forward performance record that can inform future threshold tuning.
- **Bug fix (2026-04-12, found during testing):** `compute_sigma_forward_validation()` originally used direct column access (`graded['edge']`, `graded['confidence']`) which crashed with `KeyError` when those columns were absent from older history schemas. Fixed with `.columns` guards that fall back to `np.nan`. All 8 test cases pass including the missing-column scenario (Test 8d).

### Alternatives Considered

- Standalone analysis script: rejected — `calibrate_confidence.py` is already the post-retrain diagnostics hub; adding it there keeps all calibration evidence in one place
- Backfill old history rows with σ from archived prediction CSVs: deferred — the prediction archives exist but the join is non-trivial and old data is pre-σ anyway; forward accumulation is sufficient

---

## [DEC-026] PrizePicks promo-line discovery: scraper blindness and UNDER strategy misalignment

- Date: 2026-04-12
- Status: Accepted — field semantics confirmed live; implementation shipped in DEC-028
- Decision owner: Jake

### Context

Operational investigation revealed two linked critical problems:

**Scraper blindness**: `_parse_projections()` in `prizepicks_scraper.py` reads only `line_score` and `stat_type` from each API projection record. The PrizePicks API also returns market-side metadata (later confirmed as `odds_type` plus `is_promo`), but these fields were silently discarded. Every line was stored in `historical_lines.csv` as if it were a 2-way market (both More and Less available).

**UNDER strategy misalignment**: PrizePicks rarely offers UNDER (Less) on promo/Goblin lines — these are artificially low lines designed to attract "More" bets. The system's highest-confidence picks are almost entirely UNDER on AST, TRB, STL, and BLK. If those lines are Goblin/promo, the picks cannot be placed. This was confirmed operationally: top picks (Max Christie, Julian Champagnie, Luguentz Dort, Davion Mitchell) are present in the PrizePicks scrape but the user cannot find UNDER available on those props.

The 65–79% UNDER hit rates in training data are computed on ALL scraped lines, including promo lines. The true UNDER hit rate on standard (2-way) lines only is unknown and may be significantly different.

### Decision Direction

**Fix A — Scraper**: Capture market metadata from the PrizePicks API, persist it through line history, and use it to block structurally unplaceable pick directions.

**Fix B — Pick filter**: In `filter_best_picks()` and `log_betslips()`, block `UNDER` on `goblin` or promo lines and block `OVER` on `demon` lines.

**Fix C — Strategy audit**: After Fix A is live, re-run hit rate analysis restricted to `is_promo=False` lines only. Re-evaluate whether UNDER or OVER edge dominates on true 2-way markets. The model may need to shift focus toward OVER on standard lines if UNDER hit rates collapse when promo lines are excluded.

### Why This Matters

Every UNDER pick placed on a promo line is either unplaceable (user can't find it) or is being placed at a disadvantageous synthetic line (the artificially low Goblin line means the "true" UNDER edge is much smaller or nonexistent). This is a direct money leak.

### Resolved API Check

Direct API smoke on 2026-04-12 resolved the uncertainty:

- `projection_type` was a content label such as `"Single Stat"`
- `odds_type` carried the actual market contract such as `"standard"`, `"goblin"`, or `"demon"`
- `is_promo` was present as a boolean

That means the original field-name assumption was wrong. The implementation record is in DEC-028.

### Consequences (anticipated)

- `historical_lines.csv` schema gains `is_promo` (bool) and `projection_type` (string) columns
- Large fraction of existing training data may be re-labeled as promo — historical UNDER hit rates will need to be recomputed on non-promo subset
- Pick volume may drop significantly once promo lines are excluded from UNDER candidates
- Strategy likely needs to pivot toward OVER picks or identify which prop/player combinations have standard lines consistently

---

## [DEC-025] Betslip vig accounting, true EV, and correlation-aware slip selection

- Date: 2026-04-12
- Status: Accepted
- Decision owner: Jake

### Context

Three structural weaknesses identified in the current `log_betslips()` pipeline:

1. **No vig accounting** — `_PP_MULTIPLIERS = {2: 3.0, 3: 5.0, 4: 10.0}` encodes the payout structure but no code computes the per-leg break-even probability or compares it to `dir_prob`. The implied break-even per leg is √(1/3) ≈ 57.7% for a 2-pick power play — close to but not equal to the system's `min_prob` gates, and never surfaced explicitly.

2. **No true EV** — `potential_payout` is a raw dollar amount. Slip selection uses confidence-sum ranking, not EV. True 2-pick power play EV = `3 × P1 × P2 − 1` per unit wagered. A slip ranked 2nd by confidence could have higher EV than the top-confidence pair if its legs have higher `dir_prob` or favorable correlation.

3. **No correlation handling** — slip construction takes `top.head(size)` by confidence with no awareness of pairwise correlation. For Power Play (no insurance), same-team or same-game UNDER-UNDER pairs are positively correlated, which *increases* P(both win) and therefore increases EV. This is currently ignored.

### Decision

Assume the current live contract is **Power Play only** and implement three linked changes in `nba_props.py`:

**Change 1 — Per-pick vig signal** (`filter_best_picks()` post-processing):

- `break_even_prob` = `math.sqrt(1 / _PP_MULTIPLIERS[2])` ≈ 0.577 (2-pick reference)
- `exceeds_ev_threshold` = `dir_prob >= break_even_prob`
- Added to `picks_latest.csv` and `picks_history.csv`

**Change 2 — Slip-level true EV** (`log_betslips()`):

- New columns per slip: `joint_prob_raw`, `estimated_correlation`, `joint_prob_adj`, `break_even_joint_prob`, `ev_per_unit`, `ev_dollar`
- Correlation tiers: same player ρ=0.70, same team ρ=0.35, same game (opponent match) ρ=0.15, independent ρ=0.00
- Joint prob adjustment: `P1×P2 + ρ×√(P1(1−P1)×P2(1−P2))` (Pearson copula approximation)
- EV formula (Power Play): `joint_prob_adj × gross_mult − 1`
- Keep this implementation scoped to Power Play payout logic; do not add Flex math in this change

**Change 3 — EV-ranked slip selection** (`log_betslips()`):

- For 2-pick slips: enumerate all C(n,2) pairs from top-6 eligible picks, select highest EV pair
- For 3-pick slips: enumerate C(n,3) from top-8 picks, select highest EV triple
- For 4-pick slips: enumerate candidates from the prepared pool and select the highest-EV subset under the same one-prop-per-player rule
- Replaces current `top.head(size)` confidence-ranked selection

### Why

The old confidence-sum ranking was leaving money on the table. Power Play slips are pure joint-probability products, so the right control variable is EV, not rank by confidence alone. Waiting for Flex clarification would have blocked an immediately usable improvement in the already-active Power Play path.

### Consequences

- Betslip selection now shifts toward higher-EV pairs and triples, which can differ from the highest-confidence combinations
- Same-team and same-game pairs can be preferred when the heuristic positive correlation raises joint EV
- `picks_latest.csv` / `picks_history.csv` gain `break_even_prob` and `exceeds_ev_threshold`
- `betslips_history.csv` gains `joint_prob_raw`, `estimated_correlation`, `joint_prob_adj`, `break_even_joint_prob`, `ev_per_unit`, and `ev_dollar`
- `backtest_under_only_slips()` now reuses the same EV-aware slip engine for under-only historical replay

---

## [DEC-016] Fix training_edge_analysis.json format mismatch in quality policy reader
- Date: 2026-04-11
- Status: Accepted
- Decision owner: Jake

### Context
After retraining on 2026-04-11, `models/training_edge_analysis.json` changed its top-level structure from a flat layout `{stat: {...}}` to a nested layout `{"per_target": {stat: {...}}, "cross_target_...": [...]}`. The quality policy builder (`_build_target_quality_policy()` in `nba_props.py`) still called `edge_analysis.get(stat, {})`, which always returned `{}` under the new layout.

Consequence: every stat had `clv_corr=0.0`, `under_hit=None`, `strong_n=0`, `under_n=0`. All three allow-gates (`baseline_allowed`, `strong_allowed`, `under_only_allowed`, `tracking_under_allowed`) failed simultaneously for every prop. The policy set `allowed=False` for all 9 stats — meaning **zero picks would be generated on any run using the retrained models**.

This was discovered during the post-retrain inspection. The broken behavior was silent: `_build_target_quality_policy()` returned a valid-looking dict with `allowed=False` everywhere rather than raising an error or logging a warning.

### Decision
Add a format-detection shim at `nba_props.py:4409-4415`:
```python
if isinstance(edge_analysis, dict):
    if 'per_target' in edge_analysis:
        edge_res = edge_analysis['per_target'].get(stat, {})
    else:
        edge_res = edge_analysis.get(stat, {})
else:
    edge_res = {}
```
This handles both the old flat format and the new nested format without breaking backward compatibility. No schema change to the training output is required.

### Why
The fix is in the reader, not the writer — it makes `nba_props.py` tolerant of both file layouts. Changing the training output format would require touching `train_advanced_models.py` and auditing all downstream consumers. The reader fix is safer and scoped.

### Consequences
- Quality policy now correctly reads `clv_corr`, `under_hit`, `over_hit`, `strong_hit`, `under_n`, `over_n`, `strong_n` from the post-retrain data.
- Post-retrain policy: **AST UNDER, TRB UNDER, STL UNDER, BLK UNDER** — all `allowed=True, allow_under=True`.
- STL UNDER now enabled (74.0% UNDER hit rate, CLV corr 0.324) — was incorrectly disabled before due to this bug.
- BLK UNDER now enabled (79.2% UNDER hit rate) — prior exclusion from betslips was based on 33% raw O/U, which was driven by OVER picks. UNDER-only is strong.
- PTS, PRA, PR, PA, TOV remain disabled — correctly, given low CLV corr and marginal hit rates.

### Open questions
1. **STL UNDER betslip inclusion**: resolved by DEC-019. STL UNDER is now eligible for under-only betslips when it survives the live filter and clears `confidence >= 70`.
2. **BLK UNDER betslip inclusion**: resolved by DEC-019. BLK UNDER is now eligible for under-only betslips under the same hard confidence gate.

### Alternatives Considered
- Fix the training writer to emit a flat layout (rejected — more invasive, breaks other potential consumers)
- Add an assertion/error when format is unexpected (rejected — not deployed in a crash-safe context; a fallback is safer)
- Add a warning log when falling back to `{}` (partially adopted — would be valuable future addition)

---

## [DEC-027] Mark unresolved pending-team rows explicitly in Step 8
- Date: 2026-04-12
- Status: Accepted
- Decision owner: Jake

### Context
DEC-020 brought the live injury feed into Step 8, but there was still a remaining gap when:

- the official NBA report for a team was marked `NOT YET SUBMITTED`
- ESPN had no player-level entry for a specific player on that team
- the recent-games absence proxy also did not flag the player

In that case, `data/player_projections_today.csv` carried only `live_team_status_pending=True` with blank live status fields. The row still looked effectively active in the projection artifact and could survive into the Step 8 top-10 display. Kawhi was the representative example in project memory.

### Decision
Keep the existing Step 8 schema and make the unresolved state explicit inside the existing live status columns:

1. Add a local Step 8 fallback in `run_daily.py` that detects rows where:
   - `live_team_status_pending == True`
   - `live_injury_bucket` is blank
2. Stamp those rows as:
   - `live_injury_bucket = 'team_pending'`
   - `live_injury_status = 'Team Status Pending'`
   - `live_injury_comment = 'Official NBA injury report pending and no player-level ESPN status found'`
   - `live_injury_source = 'team_pending'`
3. Treat `team_pending` as a live-risk bucket for Step 8 luck suppression.
4. Suppress `team_pending` rows from the "active players only" top-10 projection display.

### Why
This is the smallest scoped fix that closes the Step 8 visibility gap without:

- adding a new schema column
- changing pick filtering logic
- inventing a second injury contract

The picker was already protected because `sanity_check_picks()` blocks pending teams using the live injury feed directly. The real issue was the Step 8 artifact still presenting unresolved pending-team players as normal active rows.

### Consequences
- `data/player_projections_today.csv` can now carry `live_injury_bucket='team_pending'` for unresolved pending-team rows.
- Luck-derived Step 8 fields are cleared for those rows the same way they are for other live-risk states.
- The active top-10 Step 8 display no longer shows unresolved pending-team players as normal active projections.
- Pick filtering behavior is unchanged; pending teams were already blocked downstream in `nba_props.py`.
- Live-slate monitoring is still needed on the next real pending-team day because today's saved projection artifact had `0` fallback rows.

### Alternatives Considered
- Add a new `live_status_unresolved` column (rejected — wider schema change than needed)
- Remove all pending-team players from the full projection artifact (rejected — too destructive; the artifact should remain diagnostic)
- Leave the gap in Step 8 and rely only on `sanity_check_picks()` (rejected — still misleading for manual review of `player_projections_today.csv`)

---

## [DEC-028] Capture PrizePicks market type from `odds_type` and gate promo-line picks

- Date: 2026-04-12
- Status: Accepted
- Decision owner: Jake

### Context

DEC-026 identified that the PrizePicks scraper was blind to market-side metadata. Direct API verification on 2026-04-12 then resolved the exact field semantics:

- `projection_type` was a content label such as `"Single Stat"`
- `odds_type` carried the actual market contract (`"standard"`, `"goblin"`, `"demon"`)
- `is_promo` was present as a boolean

Every scraped line had been stored as if it were a 2-way market even though PrizePicks offers three line types:

- `"standard"` — normal 2-way market (OVER and UNDER both available)
- `"goblin"` — artificially low line, **More (OVER) only** — no UNDER offered
- `"demon"` — artificially high line, **Less (UNDER) only** — no OVER offered

The system's UNDER-heavy strategy (AST/TRB/STL/BLK UNDER at 65–79% hit rates) was generating UNDER picks on goblin lines that are structurally unplaceable on PrizePicks. Picks were being sent to real-money betslips against lines that don't offer the bet direction.

### Decision

1. **`prizepicks_scraper.py`** — `_parse_projections()`: log attribute keys plus both `projection_type` and `odds_type` samples for diagnostics, then normalize the market contract from `attrs.get("odds_type") or attrs.get("projection_type") or "standard"`.
2. Store that normalized market contract in the existing `projection_type` column for backward compatibility, and also store `is_promo`.
3. **`nba_props.py`** — `_normalize_historical_lines_frame()`, `fetch_prizepicks_lines()`, `save_lines_snapshot()`, `merge_vegas_lines()`, and the predict paths now carry both `projection_type` and `is_promo`.
4. **`nba_props.py`** — `filter_best_picks()`: block `UNDER` on `goblin` or promo lines and block `OVER` on `demon` lines immediately after direction is established.
5. Re-apply the same gate in the `volume_fill` and `emergency_fill` paths so unplaceable directions cannot leak back into the pick pool through fallback logic.
6. Add `projection_type` and `is_promo` to pick outputs for debugging and future post-gate analysis.

### Why

The old implementation was sending structurally unplaceable picks into both `picks_latest.csv` and betslip construction. That is a direct revenue leak. Preserving the normalized market contract under the existing `projection_type` column keeps the schema change small while still fixing the logic.

### Consequences

- `historical_lines.csv` schema gains two columns for fresh saves: `projection_type` and `is_promo`
- UNDER picks on goblin lines are blocked at the filter stage — will reduce daily pick volume on days with many goblin lines.
- OVER picks on demon lines are blocked similarly.
- Legacy and Odds API rows still normalize to `projection_type='standard'` when no market metadata is available
- `python prizepicks_scraper.py` succeeded on 2026-04-12 and materially rewrote `historical_lines.csv` with `projection_type` and `is_promo`
- `python nba_props.py predict` then materially rewrote `picks_latest.csv` / `picks_history.csv` with `projection_type`, `is_promo`, `break_even_prob`, and `exceeds_ev_threshold`
- Odds API fallback lines and legacy data remain unaffected (`"standard"` default).

### Alternatives Considered

- Block only at betslip construction (rejected — pick still pollutes `picks_history.csv` as a false positive)
- Use raw `projection_type` from the API as the contract field (rejected — live API verification showed it is a content label, not the market-side contract)
- Gate only on `is_promo` and ignore `goblin` / `demon` (rejected — `odds_type` carries more precise directionality and avoids under-blocking demon/standard distinctions)
- Audit historical lines manually first (rejected — adds delay while leaving the live money leak in place)

---

## [DEC-029] Extract the shared injury feed into `injury_feed.py`

- Date: 2026-04-12
- Status: Accepted
- Decision owner: Jake

### Context

The live injury feed introduced in DEC-020 was still physically implemented inside `nba_props.py`, while `run_daily.py` needed to call the same merge logic during Step 6. That forced a cross-import from the daily runner into the picker module:

- `run_daily.py` imported `fetch_injury_data()` from `nba_props.py`
- `nba_props.py` also owned the helper constants and parsing code for official NBA and ESPN injury feeds

That coupling was the wrong boundary. The injury feed is a shared data contract, not picker-only logic.

### Decision

Create a dedicated `injury_feed.py` module that owns:

- official NBA injury-report parsing
- ESPN injury parsing
- merge precedence between official NBA and ESPN
- preservation of `not_yet_submitted` / pending-team state

Then import `fetch_injury_data()` from `injury_feed.py` in both `run_daily.py` and `nba_props.py`.

### Why

This keeps the live injury contract in one place, removes the cross-import from the daily runner into the picker, and makes the feed testable without dragging in the larger `nba_props.py` surface.

### Consequences

- `injury_feed.py` is now the canonical home of live injury-fetch and merge logic
- `run_daily.py` Step 6 and `nba_props.py` now read the same feed contract directly
- Official NBA statuses continue to override ESPN when both sources mention the same player
- Pending-team state continues to survive the merge, which is necessary for the later Step 8 `team_pending` fallback
- The shared module is covered by `tests/test_injury_feed_merge.py`

### Alternatives Considered

- Leave the helpers inside `nba_props.py` and keep the cross-import (rejected — wrong ownership boundary)
- Duplicate the injury merge logic in `run_daily.py` (rejected — guaranteed drift between Step 6 and picker behavior)

---

## [DEC-030] Harden the live PrizePicks fetch path before relying on schema materialization

- Date: 2026-04-12
- Status: Accepted
- Decision owner: Jake

### Context

After the market-type fix in DEC-028, the next required step was to materialize the new columns into the canonical artifacts:

- `data/historical_lines.csv`
- `output/picks_latest.csv`
- `output/picks_history.csv`

The existing `prizepicks_scraper.py` path still had two operational weaknesses:

1. `_fetch_direct()` only tried a narrow set of header/param combinations, even though manual live checks showed the simpler `league_id` request shape could succeed where the heavier query failed.
2. If direct fetch failed and Playwright tried to launch a persistent Chrome profile, `TargetClosedError` could bubble out and crash the entire fetch rather than failing closed.

That combination meant the live save objective could fail even when PrizePicks was only intermittently blocking requests.

### Decision

Keep the market-type logic unchanged and harden only the fetch mechanics:

1. Expand `_fetch_direct()` to try multiple request shapes, starting with the lightest `league_id`-only variant, then progressively richer browser-style header/param variants.
2. Add small backoff after `403` / `429` responses in the direct loop.
3. Wrap the Playwright persistent-context launch in a broad exception handler so profile-launch failures log a warning and return `None` instead of crashing the whole save.

### Why

The objective here was operational reliability, not a new product rule. The canonical artifacts could not reflect DEC-028 until one successful live PrizePicks-backed save actually completed. A narrowly hardened fetch path was the smallest change that unblocked that.

### Consequences

- `python prizepicks_scraper.py` succeeded live on 2026-04-12 via the new minimal direct variant and rewrote `data/historical_lines.csv`
- The newest verified PrizePicks snapshot contained `77` validated NBA rows across `2026-04-14` / `2026-04-15`
- The captured market mix for that snapshot was `76 demon`, `1 standard`, `0 promo`
- Playwright profile failures now degrade to warnings instead of aborting the line-save process
- `python nba_props.py predict` could then rewrite the pick artifacts under the new schema

### Alternatives Considered

- Leave the fetch path alone and wait for a lucky unblocked direct call (rejected — too fragile for a required artifact migration step)
- Remove the Playwright fallback entirely (rejected — direct API remains intermittent and the browser path is still a useful fallback when it works)
- Build a separate one-off migration script for `historical_lines.csv` without a live fetch (rejected — would not provide true current market labels)

---

## [DEC-031] Exclude pre-materialization `standard` labels from the clean standard-line audit

- Date: 2026-04-13
- Status: Accepted
- Decision owner: Jake

### Context

On 2026-04-13, `output/picks_history.csv` verified that the full `2026-04-12` card went `5W / 0L / 0P`, and all five picks carry:

- `projection_type = standard`
- `is_promo = False`
- `line_source = prizepicks`

However, the repo memory already established that the active `2026-04-12` slate used during the 2026-04-12 `predict` run came from legacy cached history. Those rows were normalized to `projection_type='standard'` after schema materialization, not captured as true market-type labels from the post-fix PrizePicks feed.

If these rows are counted as clean standard-line evidence, the standard-line audit will silently mix:

- true-labeled standard rows
- legacy rows defaulted to standard during normalization

That would defeat the whole purpose of the post-gate audit.

### Decision

For the clean standard-line study:

1. Do **not** treat pre-materialization `projection_type='standard'` rows as audit-eligible evidence.
2. Count only rows that come from the post-fix true-labeled PrizePicks feed when measuring standard-line OVER/UNDER edge.
3. Keep the `2026-04-12` `5/5` card as an operational performance fact, but do not use it as proof that standard-line edge is now validated.

### Why

This preserves evaluation integrity. The whole point of the market-type work was to separate true standard lines from goblin/demon boards. Using rows that only look standard because they were default-filled later would recreate the same contamination under a cleaner name.

### Consequences

- The `2026-04-12` `5/5` result is still valid as a betting outcome
- It does **not** advance the clean standard-line audit sample
- The current blocker remains graded true-labeled sample depth, especially for true `standard` rows
- Future audit scripts should filter on provenance, not just `projection_type='standard'`

### Alternatives Considered

- Count all rows marked `standard` regardless of provenance (rejected — mixes true labels with normalized legacy defaults)
- Ignore the `2026-04-12` performance entirely (rejected — it is still a real operational result and should remain in performance tracking)


---

## [DEC-032] Absence threshold: require missed most-recent game, not just low total gp

- Date: 2026-04-15
- Status: Accepted
- Decision owner: Jake

### Context

`detect_recent_absences()` in `usage_injury_model.py` flagged any player with `gp <= max(1, last_n_games - 3)` (i.e. ≤ 2 of 5 recent games). This threshold was too broad: it treated load-managed stars (Curry playing 2 of 5 while appearing in the most recent game) as structurally absent and fed that signal into Step 8 luck suppression and the projection display.

### Decision

Add a recency gate after the gp threshold: if a player with low gp appeared in the **most recent team game date** (`last_played >= most_recent_team_date`), skip the flag — they are being rested, not injured. Only flag when gp is low **and** the player was absent from the most recent game.

Key logic:

- `gp > max(1, last_n_games - 3)` → continue (played enough overall)
- `gp > 0` and `last_played >= most_recent_team_date` → continue (played recently = load management)
- Otherwise → flag as likely absent

### Why

Structural absences (injury, illness, trade) cluster at the end of the observation window. Load management produces scattered misses across the window. A player who played the last game is available for the next game regardless of how many games they skipped earlier in the window.

### Consequences

- Curry-type rest patterns no longer contaminate the absence list or trigger luck suppression
- Players genuinely injured (missed last 1-2 games) are still correctly flagged
- Step 8 absence-risk gate (`absence_recent_gp <= 2`) now reflects actual availability concern, not general game-load scheduling

### Alternatives Considered

- Check last-2-games instead of last-1 (more conservative; rejected — adds complexity with minimal benefit given the gp≤2 base gate)
- Raise the gp threshold to ≤1 (rejected — misses players who played once long ago and have been injured since)

---

## [DEC-033] Step 8 layer merges: deduplicate source files and use normalized names as join keys

- Date: 2026-04-15
- Status: Accepted
- Decision owner: Jake

### Context

Two bugs were identified in `step8_merge_projections` in `run_daily.py`:

1. **Duplicate rows**: Only Layer 5 deduped its source file before merging. Layers 3 and 4 did not, so if `player_profiles.csv` or `player_luck_scores.csv` had duplicate player rows, the merge produced duplicate rows in `proj`. The final `proj.drop_duplicates(subset=["player"])` masked this by keeping the first occurrence, but the wrong row could be first.

2. **17.7% missing features**: All name-based merges (Layers 3, 4, 5) used raw player name strings as join keys. Players with diacritics (Nikola Jokić, Luka Dončić, etc.) stored differently in different source CSVs (with vs. without accent marks) produced NaN rows in `proj`. Those NaN cells were filled with zeros downstream, corrupting predictions for high-value players.

### Decision

1. Add `drop_duplicates(subset=["player_norm"], keep="first")` to all source DataFrames before merging in Layers 2, 3, 4, and 5.
2. Add `proj["player_norm"] = proj["player"].apply(_normalize_player_name)` once before the Layer 5 merge, then use `player_norm` (NFKD ASCII-stripped, lowercase) as the join key for all three name-based layers (5, 3, 4).
3. Layer 2 (`adj_ppp`) keeps `player_id` as the primary join key (more reliable); add `drop_duplicates(subset=["player_id"])` there.

### Why

`_normalize_player_name` already existed in `run_daily.py` (used at line 1352 for the live injury merge) and applies the same NFKD normalization used elsewhere in `nba_props.py`. Using it consistently across all layer merges ensures diacritic-variant names resolve to the same key regardless of source encoding. Deduping before merge (not only after) prevents the first-merge-wins problem.

### Consequences

- Layer feature miss rate drops from ~17.7% to near-zero for diacritic-affected players
- Luka Dončić, Nikola Jokić, and similar names now get their Layer 3/4/5 features instead of zeros
- No schema changes — `player_norm` is an intermediate merge key, not written to the final CSV
- The post-merge `drop_duplicates(subset=["player"])` remains as a final safety net

### Alternatives Considered

- Manual name mapping table (rejected — brittle, hard to maintain)
- Merge on `player_id` for all layers (rejected — Layers 3, 4, 5 source CSVs don't reliably carry NBA API player IDs)

---

## 2026-04-15 — April 14 Picks Missing from picks_history.csv (Manual Backfill)

### Context

The April 14, 2026 pipeline produced 4 picks (Toumani Camara TRB UNDER, Moussa Diabaté AST UNDER, Bam Adebayo STL UNDER, Andrew Wiggins BLK UNDER) and archived them correctly to `output/archive/2026-04-14/picks_2026-04-14_080224.csv`. However, none of these picks appeared in `output/picks_history.csv` after the April 15 morning run.

### Root Cause

The April 15 pipeline's Step 1 fetched NBA game logs but the latest `game_date` in `nba_data.csv` remained `2026-04-12` — the NBA API had not yet published April 14 box scores. This caused the grader (Step 1.5) to report "All picks already graded" (no ungraded rows existed for April 14 because those rows were never written). The `picks_latest.csv` was overwritten by the April 15 run before the April 14 picks could be recovered automatically.

The `predict` command appends picks to `picks_history.csv` at prediction time (nba_props.py:8162–8172). If the prediction step fails or is never reached for a given date, those picks are silently lost from the history file. The archive tree (`output/archive/`) retains the snapshot but the history file does not self-heal.

### Decision

1. Manually backfill April 14 picks using the archived file (`picks_2026-04-14_080224.csv`), with actuals fetched directly from `nba_api` individual player game logs.
2. Grades: all 4 picks WIN (4-0 card). Actuals: Camara REB=4, Diabaté AST=1, Adebayo STL=1, Wiggins BLK=0.
3. Rows written to `picks_history.csv` with `graded_at=2026-04-15`.

### Preventive Measure to Consider

Add a startup check to `run_daily.py` (or Step 1.5) that detects archived picks folders with no corresponding `picks_history.csv` rows and warns loudly — or auto-backfills — before proceeding. This would catch silent history gaps caused by API lag or predict-step failures.

### Consequences

- `picks_history.csv` now has 1,405 rows (4 April 14 rows restored)
- April 14 performance: 4-0
- No model or pipeline code was changed

---

## 2026-04-15 — PTS Model Root-Cause Analysis

### Context

User flagged weak PTS prediction performance. Data pulled from `picks_history.csv` (263 graded PTS picks).

### Findings

| Prop | Direction | n | WR |
|------|-----------|---|----|
| PTS | OVER | 137 | **46.0%** |
| PTS | UNDER | 126 | 54.8% |
| AST | UNDER | 52 | 69.2% |
| TRB | UNDER | 67 | 67.3% |

PTS OVER is below coin-flip. Key data points:
- Model mean prediction: 17.5 pts. Mean line: 14.5. Mean actual: 14.7. **Systematic +2.8 pt upward bias on OVER picks.**
- Confidence correlation with outcome: r=0.02 (noise). Edge correlation: r=0.03 (noise). Model has zero predictive signal on PTS.
- `sigma` is NaN for nearly all PTS picks — the low-sigma UNDER gate is not activating for PTS.

### Root Causes

1. **Market efficiency** — PTS is the highest-volume prop. Books price it better than AST/TRB. Edge is illusory.
2. **Multiplicative complexity** — Points = minutes × usage × FGA × FG% + 3PA × 3P% + FTA × FT%. AST/TRB are additive counting stats. More variance, harder to model.
3. **Game-script sensitivity** — Star player expected at 22 pts plays 28 min in a blowout and scores 11. AST is less sensitive (PG runs offense regardless).
4. **Upward-biased projection** — Season/rolling averages overestimate points due to selection bias and insufficient line-anchoring. Model predicts 2.8 pts above actual on average.
5. **Sigma not populating for PTS** — Volatility filter (`low_sigma_under`) is supposed to gate PTS but sigma is NaN for most PTS rows. The filter is inert.

### Decision

1. **Disable PTS OVER immediately** — same mechanism as STL/BLK exclusion. 46% WR is net-negative; no confidence gate fixes zero feature signal.
2. **Keep PTS UNDER with stricter gates** — 54.8% WR is marginal but acceptable with proper sigma + market-edge agreement filter.
3. **Investigate sigma NaN for PTS** — sigma must be computed and populated for PTS picks so the low-sigma gate can activate.
4. **Add PTS-specific downward calibration** — recalibrate projection by anchoring more heavily to the line (~50% weight) to reduce the +2.8 pt upward bias.
5. **Do not add shot-quality features yet** — only after sigma gate and calibration are fixed, since those are lower-risk and higher-impact.

### Alternatives Considered

- Raise PTS confidence threshold (rejected — confidence has r=0.02 correlation with outcome; no threshold helps)
- Apply market-edge-only filter for PTS OVER (rejected — market_edge is NaN for most PTS rows; same data gap as sigma)
- Retrain model specifically for PTS (deferred — not enough forward-valid graded data yet to evaluate a retrain)

---

## 2026-04-15 — PTS Model Policy Fix

### Context

Follow-up to the root-cause analysis. Implemented code changes to harden the PTS quality policy.

### Root-cause clarification (from deeper data pull)

- The dynamic quality policy was already blocking ALL PTS picks (`allowed: False, allow_over: False, allow_under: False`) based on training-set metrics.
- Sigma NaN for historical PTS picks is not a live bug — PTS stopped generating picks on 2026-04-02, before sigma was added on 2026-04-12. The PTS sidecar correctly produces sigma; it simply has no recent picks to attach it to.
- The +2.8 pt bias is **selection-induced**, not a systematic model bias. Overall PTS bias across all picks is +0.13 pts (near zero). OVER picks show +2.8 (model went high, we picked OVER → regression) and UNDER picks show -2.61 (model went low, we picked UNDER → regression). Correcting bias_advanced.json would not help.
- Market hit accuracy for PTS is 0.4915 (below coin-flip) — the market model itself has no signal on PTS direction.

### Changes Made

**nba_props.py — fallback policy (line ~5135)**
- PTS: `allow_over: False` (was `True`), `shrunk_acc: 0.54` (was `0.60`), `min_edge_over: 9.9` (hard block, was `2.0`)

**nba_props.py — dynamic policy (_build_target_quality_policy)**
- Added `pts_under_tracking_allowed` gate: activates when `n_real ≥ 2000`, `under_n ≥ 400`, `under_hit ≥ 0.54`, `market_hit ≥ 0.50`
- Added `allow_under` lower-threshold override for PTS when `pts_under_tracking_allowed`
- Added hard `allow_over = False` for PTS regardless of dynamic metrics — PTS OVER cannot be re-enabled by model eval data alone
- `pts_under_tracking_allowed` stored in policy dict for visibility

### Current PTS Policy State

```
allowed: False | allow_over: False | allow_under: False
pts_under_tracking_allowed: False (blocked: market_hit=0.4915 < 0.50)
under_hit: 0.5376 | over_hit: 0.4851 | market_hit: 0.4915
```

### When PTS UNDER Will Re-Enable

Automatically, when model training data shows:
- `under_hit ≥ 0.54` (currently 0.5376, close) AND
- `market_hit ≥ 0.50` (currently 0.4915, close)

A model retrain with more data and better features (FGA, usage rate) could achieve this.

### Consequences

- No change to immediate picks output — PTS remains fully disabled
- PTS OVER is now permanently hard-blocked (not just dynamically)
- PTS UNDER gate is in place for future auto-enable when data quality improves
- Fallback is consistent with forward-observed performance

---

## [DEC-034] UNDER-only strategy: block OVER on all composite/defensive props; re-enable STL/BLK structural picks

- Date: 2026-04-15
- Status: Accepted
- Decision owner: Jake

### Context

Live performance analysis across 1,313 graded picks revealed a structural asymmetry:

| Direction | Live WR | Training WR |
|-----------|---------|-------------|
| UNDER     | 58.9%   | 63.1%       |
| OVER      | 48.5%   | 48.3%       |

OVER performance by prop (live graded picks):

| Prop | Live OVER WR | n  | clv_corr |
|------|--------------|----|----------|
| PTS  | 46.0%        | 137 | 0.068   |
| PRA  | 48.0%        | 152 | 0.055   |
| PR   | 47.8%        | 134 | 0.097   |
| PA   | 52.5%        | 118 | 0.084   |
| BLK  | 33.3% (training, n=222) | — | 0.243 |
| STL  | 61.0% (training, n=146) | — | 0.324 |

The fallback quality policy still had `allow_over: True` for all props. If training data is ever missing or sparse, the fallback would allow OVER picks on props with confirmed sub-breakeven live performance. The dynamic policy already blocked most of these (over_hit < 0.52 threshold), but the fallback was a gap.

Additionally, `allow_structural_micro_props` was set to `False`, silently disabling STL/BLK structural UNDER picks despite their 74%/79% training hit rates.

### Decision

1. **Fallback policy**: Set `allow_over: False` and `min_edge_over: 9.9` for `pts`, `pra`, `pr`, `pa`, `stl`, `blk` — permanently blocked in the safe-default path.
2. **Dynamic policy**: Add `_over_hardblocked = stat in ('pts', 'pra', 'pr', 'pa', 'stl', 'blk')` gate that short-circuits `allow_over = False` regardless of training metrics. This prevents re-enabling if model performance fluctuates.
3. **Keep `trb` and `ast` OVER eligible**: Both have positive clv_corr and TRB OVER is 50% (small n). The mathematical gate (over_hit >= 0.52) still filters them. Not hard-blocked.
4. **Re-enable `allow_structural_micro_props: True`**: STL/BLK structural UNDER picks (L10 avg < line = UNDER) are restored. These use the 73%/78% historical win rate as confidence, clear the 70-confidence betslip gate, and represent the system's highest-confidence pick class.

### Why

- PrizePicks lines are systematically set too high. Players underperform their lines far more often than they exceed them. The OVER models are fighting the structural bias of the market.
- clv_corr for all blocked OVER props is near zero — the market is not providing information that improves OVER predictions.
- The composite props (PRA/PR/PA) are sums of stats the model already tracks individually. The UNDER edge on composites comes from each component having independent downside risk. The OVER requires ALL components to exceed expectations simultaneously — that's a product of probabilities, not a sum.
- STL/BLK structural picks don't need a model. The market sets lines at 1.5 for players averaging <1.0. The pick writes itself.

### Consequences

- OVER picks will only appear for TRB and AST (and TOV), and only when training data confirms over_hit >= 0.52
- STL/BLK structural UNDER picks re-enter the card, with confidence = historical win rate (73/78%)
- These picks are betslip-eligible since they clear the hard 70-confidence floor
- Expected effect: cards become near-exclusively UNDER; slip quality improves

### Alternatives Considered

- Block OVER for TRB/AST too (rejected — insufficient evidence yet; TRB/AST OVER data is ambiguous at small live sample)
- Add explicit UNDER-only config flag (considered; rejected — the per-prop gates are more precise and don't require a global kill switch that could interfere with future re-evaluation)

---

## [DEC-035] Soften pending-team block in sanity_check_picks: allow UNDER picks through

- Date: 2026-04-16
- Status: Accepted
- Decision owner: Jake

### Context

`sanity_check_picks()` in `nba_props.py` was blocking **every pick** from any team that had not yet submitted its official NBA injury report (`not_yet_submitted=True`), regardless of direction or individual player status. On 2026-04-16 this fired for Charlotte (CHA) and Orlando (ORL), removing 4 UNDER picks and leaving the card empty.

The blanket team-level block was documented in DEC-027 as intentional, but the intent was to prevent betting on players with genuinely uncertain availability. It over-fired because:

1. ESPN had no individual injury flag on any of the 4 players — they were almost certainly healthy.
2. All 4 picks were UNDER bets. If a player is unannounced-absent, PrizePicks typically voids the pick (no loss). If a *teammate* is absent, the player may see more usage — which makes the UNDER slightly riskier, but not categorically invalid.

### Decision

Change the pending-team guard in `sanity_check_picks` from a blanket block to a direction- and status-aware block:

- **OVER picks** on pending teams with no individual player status → still blocked (hidden teammate absence could inflate usage and push stats over the line).
- **UNDER picks** on pending teams with no individual player status → **allowed through** (void risk is manageable; blanket removal leaves zero picks).
- **Any pick where the player has a known individual status** (injured_out, doubtful, questionable, day_to_day, probable, or in status_map from ESPN) → individual status takes precedence regardless of team pending flag.

### Implementation

`nba_props.py` `sanity_check_picks()` (second definition, ~line 8446):

```python
player_has_known_status = (
    player_norm in injured_out or player_norm in doubtful_players
    or player_norm in questionable_players or player_norm in day_to_day_players
    or player_norm in probable_players or player_norm in status_map
)
if team in pending_teams and not player_has_known_status and direction != 'UNDER':
    reason = "TEAM STATUS PENDING ... skip OVER picks until that team submits"
```

### Consequences

- UNDER picks from players on teams with late-filed reports are no longer blanket-removed.
- OVER picks on those teams are still blocked.
- Players with actual injury designations continue to be caught by the elif chain below.
- DEC-027's Step 8 `team_pending` fallback in `run_daily.py` is unaffected.

### Alternatives Considered

- Keep blanket block but add a minimum-card override (rejected — complexity; masking the real problem)
- Allow all picks through regardless of direction (rejected — OVER risk from hidden absences is real)
- Block pending-team picks only when < N picks remain (rejected — order-dependent; hard to reason about)

---

## [DEC-036] Retrospective audit: model edge is not distinguishable from a naive always-UNDER baseline

- Date: 2026-08-11
- Status: Accepted (findings); strategy response OPEN — pending Jake's decision
- Decision owner: Jake
- Reproduce with: `python scripts/audit_naive_under_baseline.py`

### Context

A full-directory audit was run on 2026-08-11 (first examination since the season ended; last pipeline artifacts are dated 2026-05-19). The audit was asked to answer one question: was this system actually accurate, or not. Four findings materially contradict what STATUS.md, TASKS.md, and CLAUDE.md previously recorded.

**Finding 1 — grading and forward-logging integrity are sound.** Every live-era pick result was independently recomputed from `data/nba_data.csv` rather than trusting the pipeline's own `actual` column: **180/180 agreement**, zero disagreements. DNPs (`n=80`) carry null `actual` and are correctly excluded from win-rate denominators rather than being silently scored as UNDER wins. One cosmetic mislabel exists (Donovan Clingan PTS OVER 14.0, actual 14.0, logged `LOSS`, should be `PUSH`, 2026-03-25). `output/archive/` contains intraday timestamped pre-game snapshots (e.g. 2026-05-06 at `00:45`, `08:28`, `09:05`, `12:51`, `17:56`), so the live-era record is a genuine forward record, not a backtest. **This part of the system is trustworthy and should not be rebuilt.**

**Finding 2 — the headline win rate in CLAUDE.md is not measuring the live system.** `output/picks_history.csv` contains two distinct eras that were being averaged together:

| Era | Picks | Win rate | 95% CI (Wilson) |
|---|---|---|---|
| 2026-03-17 → 2026-03-29 | 1,249 | 54.04% | [51.27%, 56.79%] |
| 2026-03-30 → 2026-05-18 | 173 | **68.21%** | [60.94%, 74.69%] |

The March era is **backfilled, not live**: `pick_source`, `line_source`, and `prob_source` are all null for those rows, `graded_at` is null for 1,288 of 1,528 rows, and the `output/archive/2026-03-*` folders contain *only* `betslips_*.csv` files, all written `Jun 13 18:53` — no morning `picks_*` or `predictions_*` snapshots, unlike every live-era folder. That backfilled era is 88% of the graded sample and it is what dragged the documented figure down to "53–57%".

**Finding 3 — the models add no measurable edge over betting UNDER on everything.** This baseline had never been computed. Across 21,101 graded lines reconstructed from `data/historical_lines.csv`:

| Prop | Model `avg_bet_accuracy` (`models/training_results.json`) | Naive always-UNDER | Model minus naive |
|---|---|---|---|
| PTS | 50.78% | 52.8% | **−2.0** |
| TRB | 58.46% | 59.4% | **−0.9** |
| AST | 61.42% | 62.9% | **−1.5** |
| STL | 73.20% | 70.0% | +3.2 |
| BLK | 76.02% | 75.8% | +0.2 |

On PTS, TRB, and AST the trained models are *worse* than a rule that ignores every feature. Over the live window itself the model card hit 68.21% (`n=173`) against a naive always-UNDER rate of 66.52% (`n=9,588`) — an edge of `+1.7pp`, with the naive baseline falling **inside** the model's 95% confidence interval. The two are statistically indistinguishable.

**Finding 4 — most of the UNDER edge is a minutes/availability bet, not a stat-projection edge.** UNDER win rate conditioned on minutes actually played:

| Minutes | <10 | 10–20 | 20–25 | 25–30 | 30–35 | 35+ |
|---|---|---|---|---|---|---|
| n | 457 | 2,957 | 4,124 | 4,872 | 4,764 | 3,927 |
| UNDER win rate | 95.8% | 78.7% | 68.5% | 61.6% | 54.8% | **50.2%** |

Restricting to players who logged 30+ minutes (i.e. removing the minutes-collapse effect) drops the all-prop UNDER rate to **53.19%** (`n=9,632`). Per prop at 30+ minutes: AST 53.0%, TRB 48.5%, PTS **40.3%**, STL **64.0%**, BLK **71.0%**.

The surviving signal is therefore narrow and specific: **STL and BLK retain a large structural skew even at full minutes** — mean posted BLK line `0.975` vs mean actual `0.698`; mean STL line `1.493` vs mean actual `1.228`. That is a genuine right-skew mispricing (line set near the mean while the median sits below it) and it is the system's only durable modeled edge. AST/TRB/PTS UNDER performance was a bet on who sits, and PTS UNDER at full minutes is actively backwards (40.3%).

Realized live-era money, for the record: 124 slips, `$2,480` wagered, `$5,740` returned, **`+$3,260` (+131% ROI)**. Backfilled-era slips lost money (`n=31`, `−$140`, `−22.6% ROI`), which is consistent with them not being real.

### Decision

1. **Retire the "53–57% individual pick win rate" figure.** It averages a backfilled era with a live era and describes neither. Report the two eras separately, always with sample size and a confidence interval.
2. **Adopt naive always-UNDER as the mandatory benchmark.** No prop, model, gate, or strategy may be described as having edge unless it is compared against always-UNDER *on the same window and the same prop*, minutes-controlled where the claim is a stat-projection claim. Beating 50% is not evidence of anything for these props.
3. **Treat `avg_bet_accuracy` in `models/training_results.json` as non-evidence in isolation.** The existing STL 73.2% / BLK 76.0% figures that drove DEC-019 and DEC-034 are ~equal to the naive rate for those props; the policy those decisions produced happens to be correct, but it was justified by a metric that cannot distinguish a model from a constant.
4. **Preserve the grading, archiving, and forward-logging machinery unchanged.** It was independently verified and it is the reason this audit was possible at all.
5. **Strategy response left OPEN for Jake.** The audit does not unilaterally change pick logic, disable models, or alter gates. Recorded options: (a) narrow to STL/BLK structural unders plus an explicit minutes-availability model and drop the composite stat models; (b) keep current behavior and treat the ML layer as a redundant filter; (c) rebuild around CLV measurement first. No code path was modified by this audit.

### Implementation

- Added `scripts/audit_naive_under_baseline.py` — read-only, writes no artifacts, reproduces every table above from `data/nba_data.csv`, `data/historical_lines.csv`, and `output/picks_history.csv`. Supports `--live-start` and `--min-minutes`.
- No changes to `nba_props.py`, `run_daily.py`, model artifacts, gates, or policy.
- `output/STATUS.md` and `output/TASKS.md` updated to carry these findings forward.

### Consequences

- The next session inherits corrected numbers instead of the misleading 53–57% figure.
- Any future "the model improved" claim now has a defined bar to clear.
- The live-era `+$3,260` result stands as real and verified, but is explicitly a 7-week, 173-pick, postseason-only sample driven mainly by a baseline anyone could have bet, and must not be extrapolated.
- Open exposure: `clv_corr` is `0.074` for PTS in `models/training_results.json` — there is still no evidence any pick beats a closing line. That, not more model tuning, is the highest-value unanswered question.

### Alternatives Considered

- Silently correcting the CLAUDE.md win-rate figure without a decision entry (rejected — the discrepancy is the finding; hiding it recreates the problem).
- Disabling the composite models immediately as part of the audit (rejected — that is a live-money strategy change and belongs to Jake, not to an audit pass).
- Treating the March backfill as valid history and keeping the blended rate (rejected — the missing `pick_source`/`graded_at`/archive-snapshot evidence is conclusive).
- Deleting the backfilled March rows from `picks_history.csv` (rejected — destroys history and violates the project's append-only discipline; era-splitting achieves the same analytically).

---

## [DEC-037] Line store is destructive by design; Odds API backfill has no checkpointing

- Date: 2026-08-11
- Status: Accepted (findings); schema fix OPEN — pending Jake's decision
- Decision owner: Jake
- Related: DEC-036 (the audit that prompted this trace)

### Context

Follow-up to DEC-036 asked why `data/historical_lines.csv` retained only `259` rows tagged `source='odds_api'` out of `46,586`, given the credits spent on Odds API pulls. The trace found two independent defects.

**Defect 1 — `historical_lines.csv` physically cannot hold more than one price per prop, and the sportsbook always loses.**

`_merge_historical_lines()` (`nba_props.py` ~line 1476) ends with:

```python
merged = merged.drop_duplicates(subset=['game_date', 'player_norm', 'prop'], keep='last')
```

The sort immediately above orders by `source_priority` ascending, and `_line_source_priority()` (~line 1388) assigns `fanduel=3 > prizepicks=2 > odds_api=1 > unknown=0`. With `keep='last'`, the highest-priority source wins and every other row for that `(game_date, player_norm, prop)` key is discarded. `_write_historical_lines()` (~line 1500) then rewrites the whole CSV, so this is not an append — it is a replace.

Measured impact (archives vs master):

| | rows | unique `(date, player, prop)` |
|---|---|---|
| Odds-API-family rows in `data/line_archive/` (235 files) | `3,514` | `1,454` |
| Odds-API-family rows surviving in `historical_lines.csv` | `822` | — |
| Keys absent from master entirely | — | `238` |

Only `238` keys disappeared outright (dates `2026-05-12`, `2026-05-13`). The other **~1,200 keys still exist in the master file but carry the PrizePicks number instead of the sportsbook number** — the row survived, the book's price was overwritten. Total across all sources, `616` unique archive keys are absent from master.

This is the exact data required to answer the DEC-036 CLV question. The schema is structurally incapable of holding it: one line per player/prop/day means no multi-book comparison and no intraday line movement, ever.

**Defect 2 — `backfill_historical_lines()` can spend the entire credit balance and write nothing.**

`backfill_historical_lines()` (`nba_props.py` ~line 2258, CLI `python nba_props.py backfill [days]`, default `days=30`) accumulates results in a local `all_rows` list and only calls `_write_historical_lines()` **after the full day-loop completes**. There is no per-day checkpoint. It also performs unguarded `r.json()['data']` and `pr.json()['data'].get('bookmakers', [])` accesses — a single malformed or rate-limited response raises `KeyError`, the function dies, and every credit already spent yields zero rows on disk. `Ctrl-C` or an empty `all_rows` produce the same outcome.

**Observed credit loss (cause unproven).** Logged balances: `2026-05-13 13:09 → 19,364 remaining`; `2026-05-16 16:32 → 63 remaining`. ~`19,300` credits consumed in three days. It did not come from the daily pipeline: there is no `run_20260514.log`, `morning_task.log` jumps from `2026-05-12` to `2026-05-18`, and `run_20260515.log` shows PrizePicks succeeding with zero Odds API calls. `backfill_historical_lines()` stamps `source='odds_api_backfill'`; **that string appears nowhere in `data/` or `output/`**, so if the backfill ran it wrote nothing.

Leading hypothesis, explicitly **not proven**: a manual `python nba_props.py backfill` run against the `/v4/historical` endpoint, which is billed at a multiple of the standard rate (9 markets × up to 10 events × 30 days ≈ 27,000 credits — more than the 19,364 available, which would drain the balance to near-zero and stop). The authoritative record is the Odds API usage dashboard, which shows per-request history; that is outside the repo and has not been checked.

### Decision

1. **Do not run `python nba_props.py backfill` again until Defect 2 is fixed.** In its current form it is capable of consuming an entire credit balance and persisting nothing.
2. **Do not use `python nba_props.py rebuild-lines` to recover the archived Odds API rows.** `rebuild_historical_lines_from_local_sources()` routes through the same `_merge_historical_lines()` dedup and will re-clobber them against PrizePicks.
3. **Treat `data/line_archive/` as the authoritative surviving record of sportsbook prices** (`3,514` odds-api-family rows across 235 files), not `historical_lines.csv`.
4. **Recognise that the DEC-036 CLV task is blocked on schema, not on data purchase.** Buying more Odds API history before changing the line store means paying historical-tier rates for prices the merge is built to discard.
5. **Schema fix left OPEN for Jake.** Proposed shape: make the line store append-only keyed on `(game_date, player_norm, prop, source, snapshot_ts)` so every book and every snapshot is retained, with source-priority selection moved to *read* time (a view) instead of *write* time. Backfill should checkpoint per day and guard its JSON access. No code was changed by this trace.

### Implementation

- No code changes. This entry is diagnostic only.
- Evidence commands: `grep -n "drop_duplicates" nba_props.py`, `grep -rl "odds_api_backfill" data/ output/` (no matches), archive-vs-master key comparison over `data/line_archive/*/*.csv`, and credit-balance timeline from `logs/run_2026*.log`.

### Consequences

- The `+$3,260` live result and the DEC-036 findings are unaffected; this concerns line storage and API spend only.
- Any future CLV work must fix the schema first or it will silently lose the sportsbook side of the comparison.
- Roughly `1,200` prop-keys have a recoverable sportsbook price in the archives that is currently masked in the master file, plus `238` keys missing outright.
- Open question: the actual cause of the ~`19,300` credit burn is still unconfirmed and can only be settled from the Odds API usage dashboard.

### Alternatives Considered

- Changing the dedup to prefer `odds_api` over `prizepicks` (rejected — inverts the problem; the live card legitimately needs the PrizePicks line. The defect is that only one row may exist at all).
- Immediately rewriting the line schema as part of this trace (rejected — it is a data-contract change under the project's non-negotiable rules and requires a plan plus Jake's sign-off).
- Deleting the surviving `odds_api` rows as unreliable (rejected — they are evidence, and the archives are the only sportsbook history that exists).
- Re-running the backfill to "just get the data again" (rejected — that is what is suspected of costing 19,300 credits for zero rows).

---

## [DEC-038] The May decline was a market switch, not model decay; DEC-036's aggregate reading is revised

- Date: 2026-08-11
- Status: Accepted (findings); no strategy change made
- Decision owner: Jake
- Revises: DEC-036 (aggregate model-vs-naive conclusion). Does not supersede it.
- Reproduce with: `python scripts/audit_naive_under_baseline.py` plus the per-source split recorded below

### Context

DEC-036 reported that the live card's 68.2% (`n=173`) was not distinguishable from a naive always-UNDER baseline of 66.5%. Splitting the same live era by `line_source` shows that aggregate was hiding a step change, and that the two halves behave completely differently.

**The live card changed markets around 2026-05-11.** `CONFIG['live_sportsbook_key']` is `'fanduel'`, and the Step 2 log line reads `PrizePicks Snapshot (history only; live card uses configured sportsbook)`. Every pick from `2026-05-11` onward carries `line_source='fanduel'`; every pick from `2026-04-13` through `2026-05-10` carries `line_source='prizepicks'`.

Weekly live-era win rate against that seam:

| Week | Picks | Win rate | Line source |
|---|---|---|---|
| 2026-03-30 | 37 | 62.2% | (unrecorded) |
| 2026-04-06 | 23 | 60.9% | mixed → prizepicks |
| 2026-04-13 | 20 | 85.0% | prizepicks |
| 2026-04-20 | 15 | 93.3% | prizepicks |
| 2026-04-27 | 27 | 63.0% | prizepicks |
| 2026-05-04 | 28 | 82.1% | prizepicks |
| **2026-05-11** | 18 | **44.4%** | **fanduel** |
| **2026-05-18** | 5 | **40.0%** | **fanduel** |

**Mechanism: the edge is a PrizePicks pricing artifact and FanDuel does not have it.** Naive always-UNDER by book:

| Naive always-UNDER | all lines | 30+ minutes |
|---|---|---|
| PrizePicks-sourced | **67.1%** (`n=7,972`) | **58.4%** (`n=3,824`) |
| FanDuel-sourced | **52.0%** (`n=304`) | **44.3%** (`n=176`) |

At 30+ minutes FanDuel's lines favour OVER (TRB 36.2%, PTS 34.7%, BLK 44.4% UNDER). The pick engine is hard-blocked to UNDER-only by DEC-034. Running an UNDER-only strategy into an efficiently-priced book is directionally backwards, and the May results are what that produces — not a model that stopped working.

**Revision to DEC-036.** Split by source, the live record is:

| Subset | Picks | Win rate | 95% CI (Wilson) | Naive baseline | Verdict |
|---|---|---|---|---|---|
| On PrizePicks lines | 95 | **80.0%** | [70.9%, 86.8%] | 67.1% | **beats baseline** |
| On FanDuel lines | 23 | 43.5% | [25.6%, 63.2%] | 52.0% | not distinguishable |

DEC-036's aggregate (68.2% vs 66.5%, "not distinguishable") was diluted by the 23 FanDuel picks and 55 early picks with no recorded source. **On the market the system was actually built for, the card clears the naive baseline with the baseline outside the confidence interval.**

Two caveats that must travel with that number, or it will be over-trusted:

1. This is a **post-hoc subgroup** — the split was made after seeing outcomes, which is the standard way spurious significance is manufactured. It is suggestive, not confirmatory.
2. The 95 picks are a **confidence-filtered selection** compared against an **unfiltered** baseline. A clean test compares the engine's selection against random selection from the same eligible PrizePicks pool.

**What DEC-036 still holds unchanged:** training-time `avg_bet_accuracy` in `models/training_results.json` tracks the naive always-UNDER rate per prop (PTS −2.0, TRB −0.9, AST −1.5 versus naive), the minutes decomposition (UNDER win rate 95.8% at <10 min → 50.2% at 35+ min), and the narrowness of the surviving structural edge (STL/BLK).

**Money exposure:** `betslips_history.csv` ends `2026-05-10`; FanDuel-sourced picks begin `2026-05-11`. The `+$3,260` realized profit was earned entirely on PrizePicks lines, and the losing stretch was logged but never wagered.

### Decision

1. **Record that the May decline has a known cause** — the `live_sportsbook_key` market switch — and is not evidence of model or feature decay. Any future post-mortem should start from `line_source`, not from the models.
2. **Never run the UNDER-only strategy against sportsbook lines.** The DEC-034 hard-block on OVER is calibrated to PrizePicks' recreational pricing. Against an efficiently-priced book it inverts the edge. If the live card is ever pointed at a sportsbook again, the direction policy must be re-derived for that book first.
3. **Treat sportsbook prices as a fair-value reference, not as a betting target.** FanDuel's efficiency is precisely what makes it useful for the DEC-036 CLV question — it is the yardstick, not the market to bet into. This is also the corrected answer to "should Odds API snapshots be pulled into this project": yes for measurement, never for pick generation.
4. **Report the live record split by `line_source` from now on.** The blended figure is misleading in both directions.
5. **Do not treat the 80% (`n=95`) as established.** It is a post-hoc subgroup on a small sample; it raises the priority of a proper selection-vs-random test on the PrizePicks pool, it does not replace one.

### Implementation

- No code, config, or model changes. `CONFIG['live_sportsbook_key']` was left as-is at `'fanduel'` — changing it is a live-card behaviour change and belongs to Jake.
- Diagnostic only; evidence derived from `output/picks_history.csv`, `output/betslips_history.csv`, `data/historical_lines.csv`, and `data/nba_data.csv`.

### Consequences

- The single largest open question from DEC-036 ("was the good stretch real or variance?") is now better posed: the good stretch was on PrizePicks lines and clears its baseline; the bad stretch was a different market.
- `live_sportsbook_key` is now known to be a live-money-critical setting with no guard around it. Nothing in the pipeline warns that flipping it invalidates the direction policy.
- The fragility warning in DEC-036 is *strengthened*: the edge is confirmed to live entirely inside one book's pricing.

### Alternatives Considered

- Folding this into DEC-036 by editing it (rejected — DEC-036 was an accurate reading of the aggregate at the time; the project's log is append-only and revisions belong in their own entry).
- Reverting `live_sportsbook_key` to PrizePicks as part of this finding (rejected — live-card behaviour change, Jake's call, and the system is dormant so there is no urgency).
- Presenting the 80% (`n=95`) as the headline number (rejected — post-hoc subgroup; publishing it unqualified would repeat the exact error that produced the original "53–57%" figure).

---

## [DEC-039] Make the line store lossless; gate paid history behind a capped CLV plan

- Date: 2026-08-12
- Status: Accepted and implemented
- Decision owner: Jake
- Related: DEC-036, DEC-037, DEC-038

### Context

Jake approved the DEC-037 data-contract repair and confirmed that 100,000 paid
Odds API credits are available. The constraint is therefore no longer the tier;
it is preserving every paid observation and buying snapshots that answer a
specific model question.

The old physical grain, `(game_date, player_norm, prop)`, destroyed competing
book prices and intraday movement. The old backfill buffered the entire run in
memory and had unguarded JSON access, so a late error could spend all credits and
write nothing. Downstream readers also assumed that the master file was already
one row per player/prop/day, so simply retaining multiple snapshots without a
read view would have caused many-to-many training joins.

### Decision

1. Make `data/historical_lines.csv` the raw append-only store keyed by
   `(game_date, player_norm, prop, source, snapshot_ts)`.
2. Move source priority and latest-snapshot selection to
   `_select_preferred_line_rows()` at read time. Known post-tip observations are
   excluded. Training, backtest, audit, and live cached-line readers use this
   view before joining.
3. Extend the Odds API record with `bookmaker`, `event_id`, `commence_time`,
   `requested_snapshot_ts`, `over_price`, and `under_price`. Bookmakers are not
   median-collapsed at ingestion.
4. Make paid backfill inert unless the caller supplies a positive hard credit
   cap. Use a conservative preflight estimate, guarded JSON, request timeouts,
   response-header cost accounting with conservative fallback costs, and an
   atomic master/archive checkpoint after every event. Disable the legacy
   uncapped entry point.
5. Recover local archives only after the new grain and read view pass tests.
6. Spend the first paid tranche on a purpose-built three-snapshot CLV panel for
   the genuine live era, not dense all-day polling: archived card-decision time,
   60 minutes pre-tip, and 5 minutes pre-tip for every game from 2026-03-30
   through 2026-05-18.

### Implementation and measured result

- Updated `nba_props.py`, `prizepicks_scraper.py`,
  `train_advanced_models.py`, and `scripts/audit_naive_under_baseline.py`.
- Added/updated tests for multi-source retention, intraday retention,
  read-time preference, paired over/under price parsing, idempotence, and the
  zero-cap/no-request gate.
- Rebuilt from the 235 local line-archive files plus morning and prediction
  snapshots. No paid Odds API request was made.
- Materialized master: `158,603` raw rows, zero duplicate physical keys.
- Preferred view: `46,824` rows, zero duplicate
  `(game_date, player_norm, prop)` keys.
- Recovered Odds/FanDuel family: `3,357` raw rows and `1,454` unique
  player/prop/day keys; `907` keys join to box scores, versus the prior report
  that only `9` surviving Odds API rows joined.
- Baseline audit after recovery: `21,113` graded lines; live always-UNDER
  `66.77%` (`n=9,600`) versus model `68.21%` (`n=173`). The conclusion remains
  statistically indistinguishable.

### Paid snapshot budget

The live window contains 187 NBA games over 45 game dates. The historical event
odds endpoint costs 10 credits per returned market per region. With all nine
existing prop markets and one region / up to ten named books:

| Plan | Event-odds credits | Purpose |
|---|---:|---|
| decision time only | 16,830 | Reconstruct market state when the card was made |
| decision + 5-minute close | 33,660 | Minimum viable CLV panel |
| decision + 60-minute + 5-minute close | 50,490 | Recommended; adds interpretable late movement |

Historical event-list calls add approximately 135 credits for the recommended
three-snapshot plan. Set the operational cap at 55,000, leaving about 45,000
credits for missing-market retries or expansion. Do not poll every five minutes;
that would exceed the tier without answering a materially different first
question.

### Consequences

- More snapshots are now useful because the physical store no longer erases
  them, but they are evidence for CLV and market features, not proof that the
  existing model improves on its baseline.
- The next analytical deliverable is a CLV table with line CLV, no-vig price
  CLV, consensus movement, book dispersion, and market-withdrawal flags.
- Model work after CLV should be source-specific and should separate
  availability/minutes risk from conditional stat production. PrizePicks and
  FanDuel labels must not be mixed into one direction policy.
- The unaccounted May credit burn remains unresolved; only the Odds API usage
  dashboard can prove its cause.

### Alternatives Considered

- Keep one physical row and prefer Odds API at write time (rejected: destroys
  PrizePicks and intraday evidence instead).
- Store raw snapshots but let every consumer read them directly (rejected:
  creates join fanout and inconsistent source choice).
- Use the full 100k on dense five-minute history (rejected: high cost before the
  minimum decision-vs-close question is answered).
- Add more box-score/model features immediately (deferred: the current model has
  not cleared the same-window naive baseline, and CLV/availability labels are the
  missing evidence).

---

## [DEC-040] Post-recovery per-stat edge map: STL/BLK are market-wide, AST/TRB are PrizePicks-shading, composites are inverted

- Date: 2026-08-12
- Status: Accepted (findings); no strategy or code change made
- Decision owner: Jake
- Corrects: DEC-038 (claim that the edge lives *entirely* inside PrizePicks' pricing)
- Builds on: DEC-039 (lossless line store made the book-vs-book comparison possible)
- Reproduce with: `python scripts/market_edge_map.py`

### Context

The DEC-039 recovery raised Odds/FanDuel joins from `9` to `907` and the store to `158,603` raw rows. That made a direct PrizePicks-vs-sportsbook line comparison possible for the first time. `973` keys now carry both a PrizePicks line and a sportsbook line for the same player/prop/day (`2026-04-17` → `2026-05-18`).

**Constraint that bounds every conclusion below: prices are still empty.** `over_price`, `under_price`, and `bookmaker` are present in the schema but populated on `0` of `158,603` rows — local archives never carried them. No no-vig, implied-probability, or CLV comparison is attempted here; those remain blocked on the DEC-039 paid three-snapshot panel. Everything below is line-vs-line and outcome-vs-line only.

**Finding 1 — PrizePicks shades exactly three stats.** Mean gap is outlier-inflated; median is the honest statistic:

| Stat | n | Median gap | PP higher | Exactly equal |
|---|---|---|---|---|
| AST | 125 | **+1.00** | 84.0% | 12.0% |
| TRB | 153 | **+0.75** | 77.8% | 13.1% |
| BLK | 55 | **+0.50** | 65.5% | 34.5% |
| PA | 129 | 0.00 | 45.7% | 42.6% |
| PR | 155 | 0.00 | 45.2% | 39.4% |
| PRA | 151 | 0.00 | 41.1% | 44.4% |
| PTS | 168 | 0.00 | 45.2% | 41.1% |
| STL | 37 | 0.00 | 37.8% | **62.2%** |

**Finding 2 — the PrizePicks-minus-book gap predicts UNDER outcomes monotonically.** UNDER bet on the PrizePicks line, bucketed by distance above the book:

| Gap bucket | n | UNDER win rate |
|---|---|---|
| PP below book | 98 | 53.1% |
| equal | 317 | 59.0% |
| PP +0 to +0.5 | 152 | 68.4% |
| PP +0.5 to +1 | 239 | 66.5% |
| PP +1 or more | 120 | **73.3%** |

This is computable pre-tip, is market-relative rather than a stat projection, and does not depend on forecasting minutes. It is the most actionable signal found to date.

**Finding 3 — three distinct regimes, not one edge.** Minutes-controlled (30+ min) naive always-UNDER, by stat and by book:

| Stat | all | 30+ min | PP 30+ | Book 30+ | Regime |
|---|---|---|---|---|---|
| BLK | 75.9% | **71.3%** | 72.9% | 75.0% (`n=40`) | market-wide |
| STL | 70.4% | **64.6%** | 67.1% | 50.0% (`n=30`) | market-wide |
| AST | 63.2% | 53.3% | 59.4% | 53.2% | PrizePicks shading |
| TRB | 60.0% | 49.1% | 53.7% | 44.4% | PrizePicks shading |
| TOV | 52.1% | 47.9% | 47.9% | — | dead |
| PA | 54.2% | 42.0% | 45.0% | 43.0% | inverted |
| PTS | 53.1% | 40.5% | 42.9% | 38.6% | inverted |
| PR | 53.8% | 39.9% | 43.8% | 45.8% | inverted |
| PRA | 53.6% | 38.0% | 41.4% | 41.8% | inverted |

**Correction to DEC-038.** DEC-038 stated the edge "is confirmed to live entirely inside one book's pricing." That is wrong for STL and BLK. PrizePicks does not shade steals at all (median gap `0.00`, exactly equal to the book in `62.2%` of keys) yet STL UNDER still returns `64.6%` at 30+ minutes; BLK UNDER runs `71–75%` at *both* books. These two are a whole-market mispricing of low-count defensive stats, and are more durable than DEC-038 claimed. The DEC-038 statement remains correct for AST and TRB, whose edge collapses when priced at the book.

**Caveat carried with that correction:** book-side samples are small (`BLK n=40`, `STL n=30` at 30+ minutes). "Both books are soft" is suggestive, not established, and needs the paid panel to confirm.

**Finding 4 — the live card, per stat.** Only one cell clears its baseline at conventional significance:

| Stat | Dir | Book | n | WR | 95% CI | Baseline | Verdict |
|---|---|---|---|---|---|---|---|
| TRB | UNDER | prizepicks | 19 | 84.2% | [62.4, 94.5] | 60.0% | **beats** |
| TRB | UNDER | book | 11 | 27.3% | [9.8, 56.6] | 60.0% | **below** |
| AST | UNDER | prizepicks | 34 | 79.4% | [63.2, 89.7] | 63.2% | not distinguishable |
| STL | UNDER | prizepicks | 32 | 75.0% | [57.9, 86.8] | 70.4% | not distinguishable |
| BLK | UNDER | prizepicks | 10 | 90.0% | [59.6, 98.2] | 75.9% | not distinguishable |

The TRB row pair is the DEC-038 market switch in miniature: the same pick type is the best cell on PrizePicks and the worst cell on the book.

**Finding 5 — archived line movement is not usable.** Across `32,485` multi-snapshot keys, lines that moved *down* produced `55.8%` unders, moved *up* `60.9%`, unchanged `61.6%` — backwards from an informative-movement hypothesis. The archive holds repeated morning captures, not genuine pre-tip movement, and "moved" largely flags news-driven repricing that cuts against the under. Movement features must wait for the paid panel.

### Decision

1. **Stop describing this as one edge.** Report three regimes separately: market-wide (STL/BLK), book-specific (AST/TRB), and inverted (PTS/PA/PR/PRA/TOV). A single blended "UNDER-only" narrative hides that the composites are actively losing at full minutes.
2. **Promote the PrizePicks-minus-book gap to the primary research target.** It is pre-tip computable, monotonic, and market-relative. It should be the first feature evaluated after the CLV panel lands, ahead of any stat-model retraining.
3. **Keep the DEC-034 hard-block on PTS and composites.** `PRA` UNDER at `38.0%` on `n=2,255` at 30+ minutes independently confirms the block is correct.
4. **Do not weaken STL/BLK exposure on the theory that it is a PrizePicks quirk.** Per the correction above it appears market-wide, which makes it the most durable component — subject to the small-sample caveat.
5. **No code, config, model, or gate changed.** Diagnostic only.

### Implementation

- Added `scripts/market_edge_map.py` — read-only, writes nothing, reproduces all five findings. Flags explicitly when prices are absent so a no-vig comparison is never silently skipped. Covers all nine props including the composites and TOV, which the earlier `audit_naive_under_baseline.py` alias map omitted.
- No changes to `nba_props.py`, `run_daily.py`, models, or policy.

### Consequences

- The DEC-036 CLV task now has a concrete first hypothesis to test rather than an open-ended "measure CLV".
- `scripts/audit_naive_under_baseline.py` is known to omit `PA`/`PR`/`PRA`/`TOV` from its alias map; `market_edge_map.py` supersedes it for per-stat work.
- Book-side sample sizes are the binding constraint on the STL/BLK correction; the paid panel resolves it.

### Alternatives Considered

- Editing DEC-038 in place to fix the "entirely one book" claim (rejected — append-only log; corrections belong in their own entry).
- Reporting mean gap rather than median (rejected — mean is outlier-inflated and would have shown all eight props as "shaded", which is false for five of them).
- Treating the `90.0%` BLK and `84.2%` TRB card cells as established (rejected — `n=10` and `n=19`; only TRB-on-PrizePicks clears its baseline, and both remain post-hoc per-stat slices).

---

## [DEC-041] Freeze an auditable CLV request manifest; never invent historical decision times

- Date: 2026-08-12
- Status: Accepted and implemented
- Decision owner: Jake
- Builds on: DEC-039

### Context

DEC-039 authorized a three-snapshot Odds API panel for 187 games in the genuine
live era: card decision, 60 minutes pre-tip, and 5 minutes pre-tip. That budget
was only a plan, not an executable list. An exact request requires a stable game
identity, a scheduled tip, an explicit UTC request time, the nine markets, a
credit ceiling, and a checkpoint key. Most importantly, the repository does not
contain a trustworthy decision timestamp for every historical slate.

### Decision

1. Materialize one instruction per `(nba_game_id, snapshot_kind)` in
   `data/clv_snapshot_manifest_2026_live.csv`.
2. Reconcile every local game by date/home/away to the ESPN scoreboard schedule;
   use its scheduled UTC start only after the full 187-game join is one-to-one.
3. Define `decision` as the first timestamped archive whose bytes equal the
   final daily pick card. Interpret archive filename timestamps in
   `America/Chicago` and convert them to UTC.
4. If no final-card archive exists, or the recovered timestamp is not before a
   game's tip, block that decision row. Do not substitute 09:00, the first run,
   file modification time, or another unlabelled proxy.
5. Define T-60 and T-5 as exact arithmetic offsets from tip. Generate the
   manifest without calling The Odds API; resolve Odds API event IDs during the
   separately capped execution step.

### Implementation and measured result

- Added `scripts/build_clv_snapshot_manifest.py` and two focused tests.
- Cached the normalized schedule evidence in
  `data/clv_schedule_2026-03-30_2026-05-18.json`.
- Wrote `561` unique rows: 187 games x three snapshots, covering 45 game dates.
- Schedule reconciliation: 187/187 local games matched; zero missing and zero
  duplicate date/matchup keys.
- Ready: all 187 T-60 rows, all 187 T-5 rows, and 100 decision rows.
- Blocked decision rows: 46 with no final daily card and 41 whose only provable
  final-card timestamp is at/after that game's tip.
- Ready ceiling: 42,660 event-odds credits plus 120 grouped event-list credits,
  42,780 total. Full theoretical ceiling remains 50,490 + 135 = 50,625.
- Odds API requests sent while building the manifest: zero.

### Consequences

- The paid collection can begin with 474 evidence-safe rows without waiting for
  a policy choice on the 87 unavailable decision observations.
- Missing decision snapshots remain explicit missingness, which downstream CLV
  analysis must preserve.
- Any later proxy panel must use a different snapshot label and cannot be merged
  into `decision` as though it were observed.

### Alternatives Considered

- Use the final daily alias file's modification time (rejected: files were later
  copied or rewritten, so metadata is not immutable decision evidence).
- Use the earliest archived run for every slate (rejected: it may not contain the
  final selected card and answers a different question).
- Force all 561 rows to be executable with a fixed morning timestamp (rejected:
  creates false precision and contaminates the CLV label).

---

## [DEC-042] Correct the naive-baseline audit from five markets to all nine

- Date: 2026-08-12
- Status: Accepted and implemented
- Decision owner: Jake
- Corrects: DEC-036 and DEC-039 aggregate coverage figures
- Does not change: DEC-040 per-stat findings

### Context

`scripts/audit_naive_under_baseline.py` silently mapped only PTS, TRB, AST,
STL, and BLK. It omitted the Odds API aliases for PA, PR, PRA, and TOV. The TOV
defect was deeper: the script also did not load the `tov` box-score column or
define the derived-stat formula. As a result, every "all props" figure cited from
that script covered five markets while being labelled as the complete universe.

### Decision and implementation

1. Add all four missing Odds API aliases.
2. Add TOV to the box-score column contract and `PROP_FORMULAS`.
3. Add a regression test that requires all nine configured Odds API markets to
   map to auditable box-score formulas and specifically requires `tov` input.
4. Preserve the old DEC-036/DEC-039 text under append-only discipline; carry the
   corrected aggregates here and in current STATUS/TASKS.

### Corrected measurements

| Metric | Incomplete five-market audit | Corrected nine-market audit |
|---|---:|---:|
| All-date graded lines | 21,113 | **37,935** |
| All-date always-UNDER | 62.56% | **58.51%** |
| 30+ minute graded lines | 9,636 | **17,453** |
| 30+ minute always-UNDER | 53.30% | **47.74%** |
| Live-window graded lines | 9,600 | **17,129** |
| Live-window always-UNDER | 66.77% | **61.67%** |
| Model point estimate vs naive | +1.44pp | **+6.53pp** |

The live model remains 68.21% (`n=173`, 95% CI [60.94%, 74.69%]). The corrected
61.67% naive baseline is still inside that interval, although narrowly. The
honest conclusion remains "selection edge not established," not "no possible
edge." The larger point estimate strengthens the case for a properly matched
selection-vs-random test.

### Consequences

- DEC-036's PTS/TRB/AST/STL/BLK per-stat measurements remain valid because those
  aliases were always present.
- The old all-prop aggregates are invalid and must not be cited.
- The corrected minutes-controlled aggregate is below 50% because PA/PR/PRA
  strongly invert at full minutes. This reinforces DEC-040's three-regime view
  and makes blended all-prop reporting even less useful.
- No production model, live gate, sportsbook setting, or acquisition data was
  changed by this correction.

### Alternatives Considered

- Add only the missing aliases (rejected: TOV would still fail without its input
  column and formula).
- Quietly update the old decision entries (rejected: the decision log is
  append-only and the correction must remain auditable).

---

## [DEC-043] Execute the paid CLV panel; include line in the physical key and recover from checkpoints

- Date: 2026-08-12
- Status: Accepted and implemented
- Decision owner: Jake
- Builds on: DEC-039 and DEC-041

### Context

Jake authorized use of the 100,000-credit Odds API tier. The evidence-safe
manifest contained 474 executable event snapshots and 87 blocked decision rows.
A one-event canary returned 634 valid paired-price rows, but the DEC-039 key
`(game_date, player_norm, prop, source, snapshot_ts)` retained only 386 because
Bovada supplied alternate lines for the same player/market/book/snapshot.

The canary also proved two other ingestion hazards before the bulk run was
accepted: 250 snapshot files initially inherited the next-day UTC date for
evening games, and the standalone name normalizer did not match production
punctuation handling. Raw JSON and parsed per-event checkpoints preserved every
paid response, so both defects were repairable without repeating a paid call.

### Decision

1. Expand the lossless physical key to
   `(game_date, player_norm, prop, line, source, snapshot_ts)` everywhere.
2. Retain all alternate lines physically. For the one-row read view, select the
   paired-price line closest to a balanced no-vig market before applying source
   and recency priority.
3. Treat the manifest's local game date as authoritative; never derive the NBA
   game date by slicing a UTC commence timestamp.
4. Match paid name normalization to production NFKD/punctuation normalization.
5. Save raw JSON and parsed CSV before marking a manifest key complete, batch
   the atomic master rewrite, and recover completed parsed checkpoints on resume.
6. Execute only the 474 evidence-safe rows under a 45,000 lifetime cap. Preserve
   empty responses as availability observations and do not submit blocked rows.

### Measured result

- Completed: 474/474; 469 with odds and 5 genuine empty observations.
- Spend: 35,880 credits; API-reported balance: 64,120.
- Paid rows: 253,725; paired over/under prices: 253,680 (99.98%).
- Master rows: 412,328; corrected physical-key duplicates: zero.
- Actual API snapshots after requested time or at/after tip: zero.
- Wrong local game dates after repair: zero.
- Unique player/prop/date box-score join rate: 95.28%.
- Returned markets: eight. `player_turnovers` was never returned and is not a
  productive retry target.

### Consequences

- The canary prevented 248/634 valid rows from being silently destroyed and
  exposed that DEC-039's purportedly lossless key was still incomplete.
- The raw/parsed checkpoint boundary is now the authoritative paid-data recovery
  layer; the master CSV is a reproducible materialization, not the sole copy.
- The five empty observations are valid market-withdrawal/unavailability labels.
- No additional Odds API spend is justified until feature analysis names a
  specific coverage gap.

### Alternatives Considered

- Collapse alternate lines to one row at ingestion (rejected: destroys paid
  price ladders and makes the main-line rule irreversible).
- Retry after each defect (rejected: raw checkpoints already contained the data).
- Spend the full 100,000 because it was available (rejected: budget availability
  is not an analytical objective).

---

## [DEC-044] Promote the PP-minus-consensus gap; retract market-wide STL softness; reserve remaining credits

- Date: 2026-08-12
- Status: Accepted (findings and data policy); model integration pending
- Decision owner: Jake
- Corrects: DEC-040's market-wide STL/BLK interpretation
- Builds on: DEC-043

### Context

The paid panel made price-bearing consensus features possible at decision,
T-60, and T-5. `data/clv_features_2026_live.csv` contains 16,989
game/player/prop rows, including 15,391 T-60-to-T-5 line movements, 7,920
decision-to-T-5 movements, 11,551 T-60 PrizePicks-minus-consensus gaps, no-vig
probabilities, dispersion, and withdrawal flags.

### Findings

1. **The PP gap survives at scale in-sample.** On 10,943 graded T-60 gap rows,
   PrizePicks UNDER is 49.5% when PP is below consensus, 58.8% when equal, 65.1%
   at +0 to +0.5, 67.2% at +0.5 to +1, and 67.1% at +1 or more. The original
   973-key dose-response was not a small recovered-archive accident.
2. **DEC-040's market-wide claim does not survive the larger book sample.** At
   30+ minutes, T-5 sportsbook-consensus UNDER is STL 51.9% (`n=621`) and BLK
   58.7% (`n=965`). STL is efficiently priced at the book; BLK retains only a
   modest skew. The prior 64.6%/71-75% reading came from only 30/40 book rows.
3. **The selected card beat the book line cross-sectionally, not temporally.** On
   159 covered picks, direction-adjusted line advantage versus T-5 sportsbook
   consensus averages +0.671 units and is positive on 71.7%. But the closing
   no-vig probability of the selected side averages only 50.18% and exceeds 50%
   on 47.8%; sportsbook prices do not independently confirm selection skill.
4. **Same-book PrizePicks CLV remains unmeasured.** Archived PP references are a
   median 534 minutes old at T-5; only 3 selected-pick references are within the
   final hour. Odds API cannot supply a PrizePicks close.
5. **Late sportsbook movement is weak.** UNDER is 51.5% after the line moves
   down, 54.0% unchanged, and 53.6% after it moves up (`n=14,985`).

### Decision

1. Make PP-minus-consensus gap, no-vig consensus probability, book dispersion,
   and availability the first candidate market features in chronological tests.
2. Retract the claim that STL softness is market-wide and weaken the equivalent
   BLK claim. Preserve DEC-040 unchanged as the small-sample precursor.
3. Do not treat this same-window confirmation as an out-of-sample model win.
   Evaluate against same-date random UNDER and per-stat always-UNDER controls.
4. Capture PrizePicks itself at T-60 and T-5 going forward next season.
5. Reserve the remaining 64,120 Odds API credits. Do not buy generic density,
   retry unavailable TOV, or attempt to solve missing PrizePicks closes with the
   wrong vendor.

### Consequences

- The clearest current edge is source-specific line shopping: PrizePicks above a
  sportsbook consensus, not a broad belief that every book misprices STL/BLK.
- The live sportsbook setting remains strategy-critical. A policy trained on the
  PP gap must not silently publish against FanDuel lines.
- The next model action is a frozen chronological selection-vs-random evaluation,
  followed by source-specific integration only if it clears that test.

### Alternatives Considered

- Spend remaining credits on earlier book-only dates (rejected: there are no
  matching historical PrizePicks rows before April 3, so this cannot widen the
  PP-gap test).
- Buy a fourth sportsbook snapshot immediately (rejected: existing movement and
  dispersion coverage is already ample; same-book PP close is the actual gap).
- Keep the DEC-040 market-wide label despite the larger sample (rejected: the
  paid evidence directly contradicts it for STL and materially weakens it for BLK).

---

## [DEC-045] Do not promote CLV features on thin standard-line history; capture PrizePicks closes directly and reserve Odds credits

- Date: 2026-08-12
- Status: Accepted and implemented
- Decision owner: Jake
- Builds on: DEC-038, DEC-043, DEC-044

### Context

DEC-044 identified six candidate pre-tip market feature groups and required a
chronological selection-versus-random test before model integration. The frozen
CLV table has broad sportsbook coverage, but the production card's eligible pool
is narrower: PTS/TRB/AST, UNDER, standard non-promo PrizePicks lines, archived
model under-edge of at least 5%, and a graded non-push outcome. Most archived
PrizePicks references are demon lines that the production policy would not have
treated as the same eligible market. Mixing them into the primary test would
answer a different question.

### Findings

1. The exact eligible pool contains `889` T-60 rows across `8` dates and `902`
   T-5 rows across `9` dates. The expanding development window has only `3`
   validation dates / `15` selected rows for feature selection.
2. None of the requested feature groups clears the predeclared survivor gate:
   mean top-five lift at least `+1pp`, positive lift in at least `3` folds,
   non-negative mean AUC lift, at least `12` validation dates, and at least `60`
   validation selections. PP-minus-consensus gap shows an apparent T-60
   `+26.7pp` top-five lift and positive lift in `3/3` folds, but the entire claim
   rests on `15` selections; it remains a hypothesis, not a production feature.
3. T-60-to-T-5 movement and withdrawal are not available at T-60 and were
   excluded there as future leakage. All six groups were eligible at T-5.
4. Untouched future results are too small and inconsistent for promotion. T-60
   shadow ranking is `5/9` (`55.6%`) versus current model-edge `4/9` (`44.4%`)
   on two dates; matched random is `51.0%` / `48.3%`. T-5 shadow and current
   ranking are both `4/10` (`40.0%`) on three dates; matched random is `46.5%` /
   `45.0%`. The always-UNDER eligible pool is `54/82` (`65.9%`) in both views.
5. The Odds API cannot provide true PrizePicks closing CLV. More book snapshots
   cannot repair the missing same-source PrizePicks observation.

### Decision

1. Promote none of the six market feature groups. Do not alter or retrain the
   production regression models on this evidence. Persist the fitted selection
   overlays as research-only shadow artifacts with `enabled=false`.
2. Make the standard, non-promo production pool the authoritative primary test.
   Promotional line types may be analyzed separately only when sample sizes
   support distinct policies.
3. Capture PrizePicks directly at T-60 and T-5 going forward. Save raw payloads
   first, parsed per-event snapshots second, then merge into the master line
   store and checkpoint the event/window.
4. Restore the live source to PrizePicks and refuse to publish a card when the
   observed source differs from the configured source-specific policy.
5. Disable implicit live Odds API fallback, read its key only from the
   environment, and reserve all `64,120` remaining credits. A future tranche
   requires a specific missing observation named by the frozen evaluation.

### Implementation and Reproduction

- Evaluation: `python scripts/evaluate_clv_walk_forward.py`
- Frozen eligible rows: `data/clv_eligible_pool_frozen.csv`
- Feature/fold/holdout evidence: `output/clv_walk_forward_*.csv` and
  `output/clv_walk_forward_summary.json`
- Research policy: `models/clv_selection_policy.json`; both overlays disabled
- Direct capture: `scripts/capture_prizepicks_closing_snapshots.py`
- Daily integration: `run_daily.py` Step 2.5
- Focused verification: `39` tests passed before the publication-guard test was
  added; the final full-suite result is recorded in STATUS after verification.

### Consequences

- The project does not manufacture improvement by accepting a positive result
  from three dates. The current model remains the production comparator.
- The next useful data is true same-book PrizePicks timing, not generic Odds API
  density. Forward collection can resolve that gap at zero Odds credits.
- Promotion can be reconsidered after at least `12` validation dates / `60`
  validation selections and an untouched holdout of at least `15` dates / `100`
  picks, including at least `20` standard picks.

### Alternatives Considered

- Promote PP gap from its apparent T-60 lift (rejected: only three validation
  dates and fifteen selected rows).
- Pool demon/goblin and standard lines for power (rejected: changes the eligible
  market and does not estimate the current production policy).
- Spend the remaining credits on more blind historical snapshots (rejected: the
  missing variable is a direct PrizePicks close, which Odds API does not sell).

---

## [DEC-046] Expand to 80 dates; accept T-60 no-vig for sportsbook consensus and require PrizePicks source transfer

- Date: 2026-08-13
- Status: Accepted and implemented
- Decision owner: Jake
- Builds on: DEC-043, DEC-044, DEC-045

### Context

DEC-045 could not promote any market feature because exact standard, non-promo
PrizePicks history supplied only eight/nine eligible dates. Jake authorized an
immediate paid expansion to replace the thin sportsbook evidence with a real
chronological sample. The specific retrievable gap was older T-60/T-5
sportsbook consensus history; Odds API still cannot sell historical PrizePicks
lines.

### Acquisition

An isolated manifest covered `2026-02-23` through `2026-03-29`:

- `35` dates, `264` games, `528` ready T-60/T-5 event snapshots.
- `528/528` completed with data; no empty snapshots or failures.
- `314,162` parsed rows; `314,054` paired-price rows (`99.97%`).
- Zero physical-key duplicates, date mismatches, post-request snapshots, or
  at/after-tip snapshots.
- `94.81%` box-score join coverage (`24,462 / 25,801` unique keys).
- Checkpoint spend: `42,129`; API-reported balance: `13,436`.

The combined original and expansion feature table contains `42,790` rows over
`80` dates. The primary pool is PTS/TRB/AST consensus UNDER, two or more books,
at least 90% paired-price completeness, and graded non-push outcomes.

### Chronological Findings

Feature selection used `54` development dates and `24` expanding validation
dates (`119` top-five selections). The latest `20` dates with archived model
predictions were untouched and supplied exactly `100` top-five holdout
selections.

1. **T-60 no-vig probability survives.** Validation lift is `+21.9pp`, positive
   in `3/4` folds, with `+0.0374` mean AUC lift. Holdout is `61/100` versus
   current model-edge `56/100`, date/prop-matched random `51.94%`, and full-pool
   always-UNDER `52.43%`. The one-sided matched-random probability is `3.45%`.
   Holdout halves are `58%` then `64%`.
2. **T-5 no-vig does not transfer.** It passed development selection but went
   `52/100` versus current model `55/100` and matched random `50.77%`.
3. **Movement is redundant after no-vig.** T-5 movement screened positively in
   the first selection step, then produced `-2.47pp` incremental lift after
   no-vig entered. It is rejected.
4. Sportsbook consensus line alone, dispersion, and availability/withdrawal did
   not survive the sequential validation gate.

### Decision

1. Accept `consensus_p_over_t_minus_60` as the only validated sportsbook
   selection feature from this panel.
2. Reject T-5 no-vig, line movement, consensus line alone, dispersion, and
   availability for integration on this evidence.
3. Keep both sportsbook shadow artifacts disabled for PrizePicks production.
   The accepted outcome is defined against a sportsbook consensus line, while
   the live card is placed at PrizePicks. A feature must not cross that source
   boundary without a paired line test.
4. Use the remaining `13,436` credits only to pair future direct PrizePicks
   T-60 captures with T-60 sportsbook consensus. Do not spend them on T-5, which
   already failed a powered holdout.
5. Correct the sample contract: a five-pick card needs `20` holdout dates to
   produce `100` selections. The prior `15 dates / 100 picks` wording was
   internally impossible.

### Credit/Executor Incident

The prior recorded balance (`64,120`) minus expansion checkpoint spend
(`42,129`) implies `21,991`, but the API reports `13,436`: an `8,555`-credit
gap. Two command invocations timed out during execution, and a later atomic
master replacement failed with file contention. The strongest hypothesis is
that timed-out shells left overlapping Python executors briefly alive, causing
paid duplicate requests that the shared checkpoint could not serialize. This
is not proven without request-level dashboard history.

The executor now creates a PID lock beside each checkpoint, rejects an active
second process, detects stale locks, and retries transient Windows atomic
replacement failures. Raw and parsed files preceded every master merge, so no
purchased expansion observation was lost.

### Reproduction

- Manifest: `data/clv_snapshot_manifest_2026_expansion.csv`
- Checkpoint/ledger: `data/clv_snapshot_execution_expansion_*`
- Audit: `output/clv_panel_expansion_audit.json`
- Features: `data/clv_features_2026_expansion.csv` and
  `data/clv_features_2026_combined.csv`
- Evaluation: `python scripts/evaluate_sportsbook_expansion.py`
- Holdout selections: `output/sportsbook_expansion_holdout_selections.csv`
- Policy: `models/sportsbook_selection_policy.json` (`enabled=false` for both)

### Alternatives Considered

- Spend the balance on more T-5 snapshots (rejected: T-5 failed the powered
  holdout and adds no present decision value).
- Treat consensus-line performance as directly transferable to PrizePicks
  (rejected: differing lines can change the outcome and recreate the source
  mistake diagnosed in DEC-038).
- Continue reporting that every feature lacks evidence (rejected: T-60 no-vig
  now clears the declared chronological validation and holdout gates at the
  sportsbook-consensus grain).

---

## DEC-047 — Spend the Remaining Reserve Only on Paired PrizePicks/Sportsbook T-60 Captures

**Date:** 2026-08-13
**Status:** Accepted and implemented
**Scope:** Forward data acquisition; no production model or pick gate change

### Context

DEC-046 accepted sportsbook T-60 no-vig probability after a chronological
validation and untouched 100-selection holdout, but correctly blocked production
because the tested outcome used the sportsbook consensus line while the real card
uses PrizePicks. Jake directed that the remaining reserve be used for PrizePicks
T-60 capture and challenged the interpretation that the model was useless outside
this feature.

### Decision

1. Use the remaining `13,436` Odds API credits only for current sportsbook
   event-prop snapshots paired to a direct PrizePicks T-60 capture of the same
   game.
2. Continue direct PrizePicks T-5 capture because it is free, but make no paid
   Odds API request at T-5. The powered T-5 sportsbook holdout already failed.
3. Require a unique event match using tip time and matchup identity before a
   paid request. An ambiguous or missing match is a zero-credit failure, not a
   guessed join.
4. Enforce a durable `13,436` lifetime cap in
   `data/prizepicks_t60_pair_checkpoint.json`. Preflight each request at a
   nine-credit ceiling and record actual spend from `x-requests-last`.
5. Keep sportsbook T-60 raw and parsed artifacts separate from PrizePicks raw
   and parsed artifacts, then append the parsed book rows to the lossless master
   only after raw persistence.
6. Do not change the production model. The result of DEC-046 is that current
   model-edge ranking is weaker than the accepted T-60 selector on that holdout,
   not that every model component is useless. Projection, minutes, injury, and
   eligible-pool logic remain in place pending the paired source-transfer test.

### Cost and Availability Verification

The official current event list endpoint is quota-free. Current event odds cost
one credit per returned market per region; the nine-market, one-region request
therefore has a conservative ceiling of nine credits per game, versus 90 for the
historical equivalent. `13,436 // 9 = 1,492` complete game captures with eight
credits left at the ceiling; unavailable markets reduce actual cost.

An authenticated free event-list call on 2026-08-13 returned HTTP 200,
`x-requests-last=0`, `x-requests-remaining=13,436`, and 34 future NBA events.
ESPN listed no NBA games for August 13. The earliest currently listed event is
October 20, so no paid T-60 observation could honestly be captured today.

### Implementation Contract

- `run_daily.py` launches `scripts/capture_prizepicks_closing_snapshots.py` with
  `--pair-sportsbook-t60 --odds-credit-budget 13436`.
- PrizePicks rows retain `game_description` solely to make the cross-source event
  match deterministic.
- Sportsbook raw: `data/odds_api_prizepicks_t60_raw/`.
- Sportsbook parsed: `data/odds_api_prizepicks_t60_parsed/`.
- Spend/completion checkpoint: `data/prizepicks_t60_pair_checkpoint.json`.
- A completed key is never requested twice; terminal 4xx errors block automatic
  retries; a request is refused if its ceiling would cross the lifetime cap.

### Verification

- Direct script execution and the no-game schedule plan complete successfully.
- Unit coverage verifies matchup disambiguation, raw/parsed/price persistence,
  actual header-cost accounting, refusal before the paid request at the cap, and
  zero sportsbook calls for T-5.
- A free live account check confirmed API authentication and the full balance
  without spending a credit.

### Alternatives Considered

- Spend on more historical snapshots (rejected: DEC-046 already powered the
  sportsbook feature decision; historical data cannot recover PrizePicks close).
- Buy paired T-5 observations (rejected: T-5 failed the powered holdout).
- Treat all rejected market features as proof the entire model is worthless
  (rejected: those tests measured incremental selection value at a specific
  sportsbook grain, not every projection or operational component).

---

## DEC-048 — Build Availability and Minutes as a Leakage-Safe Pregame Hurdle Model

- **Date:** 2026-08-13
- **Status:** Accepted for prop-model retraining; not approved for live use
- **Decision owner:** Jake

### Context

The legacy minutes model is trained only on box-score appearance rows. It cannot
represent plausible pregame candidates who did not appear, so it cannot learn
availability risk. Prop retraining needs strict out-of-fold availability and
minutes estimates rather than realized participation or minutes.

### Decision

Use a two-stage pregame hurdle model: estimate the probability that a player
appears, estimate minutes conditional on appearing, and retain availability
probability, conditional minutes, and soft expected minutes as separate prop-model
features. Use a fixed `0.50` availability gate for a point minutes estimate when
the loss function is all-candidate MAE.

Reconstruct the eligible pool from prior same-team appearances only. A player
enters after two prior team appearances with recent rotation minutes of at least
eight and remains eligible for at most 45 days after the last appearance. Features
must be computable before the current game. Player identity, current-game
appearance, and current-game minutes are excluded.

Histogram gradient boosting was selected independently for availability and
conditional minutes using four expanding calendar-date folds over the first two
seasons. The latest season remained outside fitting and family selection. The
model produced `74,965` strict out-of-fold rows with zero chronology violations.

### Evidence

The reconstructed table contains `98,217` candidates, including `26,704`
non-appearances, across 508 dates, with no duplicate keys or null feature cells.
Every acceptance measure improved in all four development folds. On the 2025-26
holdout (`35,444` candidates across 193 dates):

- availability Brier improved `0.146247 → 0.111513`; AUC was `0.886173`;
- conditional-minutes MAE improved `5.027538 → 4.787715`;
- fixed-0.50-gated all-candidate MAE improved `6.157249 → 5.902702`; and
- soft expected-minutes RMSE improved `9.877959 → 8.728322`.

All four paired daily gains had positive 95% date-clustered bootstrap intervals.
The likely-prop-pool proxy (`n=19,398`) improved on all four measures. Maximum
absolute availability calibration-decile gap was `0.022683`.

Soft expected minutes did **not** beat the strongest last-status baseline on
all-candidate MAE (`6.206798` versus `6.157249`). It is retained as an
expectation/RMSE feature, not substituted for the fixed-gate point estimate when
MAE is the objective.

### Consequences

- Prop models may be retrained using only
  `data/availability_minutes_oof_predictions.csv`.
- The full-fit artifact remains disabled for live scoring until an upcoming-game
  roster/candidate adapter exists and the downstream prop stack passes its own
  chronological future comparison.
- This validates a model component; it does not establish profitable prop
  selection or betting performance.

### Limitations

- Historical candidates are inferred because the source lacks a point-in-time
  transaction or roster feed.
- Candidates before their second team appearance and returns after more than 45
  days are outside this reconstruction.
- The likely-prop-pool subgroup is a rotation proxy, not an archived PrizePicks
  board.
- The `0.50` gate is frozen and was not optimized on the holdout.

### Artifacts and Verification

- Trainer/scorer: `scripts/train_availability_minutes_model.py`
- Independent audit: `scripts/audit_availability_minutes_model.py`
- OOF predictions: `data/availability_minutes_oof_predictions.csv`
- Policy: `models/availability_minutes_policy.json`
- Report: `output/AVAILABILITY_MINUTES_REPORT.md`
- Executed notebook: `notebooks/availability_minutes_validation.ipynb`
- Contract/leakage tests: `tests/test_availability_minutes_model.py`

---

## DEC-049 — Join Strict OOF Availability/Minutes Features into Prop Training

- **Date:** 2026-08-13
- **Status:** Accepted for isolated candidate retraining; production unchanged
- **Decision owner:** Jake

### Context

DEC-048 produced valid out-of-fold availability and minutes predictions, but the
prop trainer did not join or register them. Running the existing command would
therefore have retrained the old feature set while appearing to use the new model.

### Decision

Add a dedicated adapter that joins `data/model_data_advanced.csv` to
`data/availability_minutes_oof_predictions.csv` one-to-one on
`(GAME_ID, PLAYER_ID, team)`. Retain matched appearance rows and carry exactly
three candidate model features: availability probability, minutes conditional on
appearing, and probability-weighted expected minutes.

The adapter must fail on duplicate keys, date or actual-minutes disagreement,
null/invalid feature values, OOF training dates at or after the target game, input
dates newer than the OOF table, or less than 90% match coverage within the OOF
window. The trainer registers the fields but may discard an individual field
during its per-prop walk-forward feature selection. Availability-aware training
must pass `--require-availability-features`, which repeats the chronology and
feature-contract checks and writes a contract beside candidate model artifacts.

Train candidates in an isolated directory. Do not replace production models until
the full prop comparison and future-game feature adapter pass.

### Evidence

The materialized table contains `52,775` appearance rows from `2024-03-08` through
`2026-05-18`, with zero duplicate keys, null feature cells, or chronology failures.
Coverage inside the OOF window is `52,822/58,435` (`90.39%`) overall and
`44,032/45,685` (`96.38%`) for the `mp_l10 >= 15` rotation proxy. Exclusions are
explicit: `15,096` rows before OOF begins, `5,613` unmatched rows inside the window,
and `47` zero-minute non-appearances.

An isolated STL smoke retrain consumed the 52,775-row artifact successfully.
Conditional and expected minutes survived its 40-feature cap; standalone
availability probability did not. That verifies plumbing and selection behavior,
not superiority over the frozen STL model.

### Consequences

- The project can now run all nine availability-aware prop retrains without a
  silent old-feature fallback.
- Earlier history is intentionally excluded because it has no strict OOF value.
- Fringe-player coverage is lower than rotation-player coverage; the exact counts
  are retained in the join summary.
- Production/live activation remains blocked on downstream comparison and an
  upcoming-game availability feature adapter.

### Artifacts and Verification

- Adapter: `scripts/build_prop_availability_training_data.py`
- Materialized table: `data/model_data_availability_advanced.csv`
- Join audit: `output/availability_prop_training_join_summary.json`
- Trainer contract: `train_advanced_models.py --require-availability-features`
- Tests: `tests/test_prop_availability_training_integration.py`
- Smoke artifacts: `models/availability_retrain/`

---

## DEC-050 — Availability/Minutes Prop Retrain Does Not Clear Promotion; No Model Replaced

- **Date:** 2026-08-13
- **Status:** Candidates rejected for promotion; production unchanged; STL carried forward as the only follow-up
- **Decision owner:** Jake

### Context

DEC-049 authorized retraining all nine prop candidates in the isolated
`models/availability_retrain/` directory using the joined availability/minutes
features, with explicit instructions not to overwrite `models/*_advanced.pkl`
until the candidates beat production and the same-prop always-UNDER baseline on
a chronological comparison. That retrain has now run for all nine props.

### Decision

**Do not promote any candidate to production.** Verified against
`models/availability_retrain/selected_features_advanced.json`:

| Prop | New features retained by walk-forward selection |
|---|---|
| STL | `conditional_minutes_prediction`, `expected_minutes_prediction` |
| PRA | `conditional_minutes_prediction` |
| PTS, TRB, AST, PR, PA, BLK, TOV | none |

Standalone `availability_probability` was rejected for every prop. For the seven
props with no retained feature, the availability/minutes experiment is closed —
there is nothing to promote because nothing survived selection.

For STL and PRA, a same-row OOF comparison aligned production and candidate
predictions on identical player/date/prop/line observations:

| Prop | Production | Candidate | Change | Always UNDER (same rows) |
|---|---:|---:|---:|---:|
| PTS | 51.22% | 51.82% | +0.59pp | 53.79% |
| PRA | 50.86% | 51.19% | +0.33pp | 54.32% |
| STL | 70.99% | 71.14% | +0.15pp | 69.82% |
| PR | 49.67% | 49.81% | +0.14pp | 52.28% |
| BLK | 74.30% | 74.43% | +0.13pp | 74.69% |
| PA | 51.54% | 51.54% | 0.00pp | 54.33% |
| TRB | 53.88% | 53.75% | −0.12pp | 56.15% |
| AST | 59.41% | 59.12% | −0.29pp | 61.42% |
| TOV | 58.51% | 57.61% | −0.90pp | 52.26% |

No candidate-vs-production change has a positive 95% confidence interval — every
delta is consistent with noise. Seven of nine candidates also lose to the naive
always-UNDER baseline on the same rows, repeating the DEC-036 pattern that a more
complex model is not automatically beating the simplest possible strategy.

**STL is the only prop worth a further look.** Both minutes features survived
selection and the candidate beat production and always-UNDER on this comparison,
but the margin over production (+0.15pp) is far too small to promote on its own.

### Why this comparison is not yet a clean promotion test

The candidate artifacts were fit through `2026-05-18` (the end of the OOF window
established in DEC-048/049) — there is no untouched future holdout for this
retrain the way DEC-046/047 built one for the sportsbook feature. The same-row OOF
table above is chronologically valid (no row's features postdate that row's game),
but it is not a reserved period the candidate never influenced during selection.
A real promotion decision for STL requires: (a) an ablation using identical
training rows and folds, with vs. without the two minutes features, isolated to
STL only, and (b) evaluation on a newly reserved future period the STL candidate
has not touched, per the DEC-049 condition and the DEC-036/042 always-UNDER
benchmark rule.

### Consequences

- `models/*_advanced.pkl` (production) are untouched — verified: production
  `selected_features_advanced.json` (mtime 2026-06-07) contains none of the three
  availability/minutes fields, while the candidate copy in
  `models/availability_retrain/` does. The live daily pipeline is unaffected.
- The availability/minutes feature experiment is closed for PTS, TRB, AST, PR,
  PA, BLK, and TOV. Do not re-attempt without a new signal — repeating the same
  join and features would repeat the same null result.
- PRA retained a feature but lost to always-UNDER on the same rows; also not
  promotable as-is.
- STL is the sole open follow-up, and it is explicitly not authorized for
  promotion yet — it needs the isolated ablation and reserved holdout described
  above before any live change.

### Artifacts and Verification

- Candidate artifacts: `models/availability_retrain/` (all nine props, isolated,
  untracked, not wired into `run_daily.py`)
- Candidate feature selection: `models/availability_retrain/selected_features_advanced.json`
- Production feature selection (unchanged): `models/selected_features_advanced.json`
- Verified via `git status` + mtime check that no tracked production model file
  was modified by this retrain, and via direct inspection that
  `availability_probability` / `conditional_minutes_prediction` /
  `expected_minutes_prediction` are absent from the production selection file
  and present only where noted above in the candidate file.

### Addendum — Why the new minutes features mostly didn't matter

Checked `models/availability_retrain/feature_importance_advanced.json` against
the question of whether minutes information itself is useless (it is not). Every
prop model already carries pre-existing rolling minutes features (`mp_l3`,
`mp_season_avg`, `mp_ewma5`, `mp_trend_ratio`, `mp_recency_anchor`), and those
already rank highly out of 183 total features: `mp_l3` is #22 for PTS, `mp_trend_ratio`
is #11 for TRB, `mp_l3`/`mp_trend_ratio` are #20-21 for AST. The new
`conditional_minutes_prediction` / `expected_minutes_prediction` fields ranked
near the bottom for PTS (#158, #82), TRB (#170, #158), and BLK (#120, #137) —
i.e., they added little beyond what the existing simple rolling-minutes average
already captured, consistent with the null selection result above.

The exception is exactly where selection kept the new features: for STL,
`conditional_minutes_prediction` ranked **#3 of 183** (ahead of every existing
minutes feature); for PRA it ranked **#9**. Both are stats that scale close to
linearly with playing time and have comparatively little other signal (shot
luck, matchup) to compete with it. For PTS/TRB/AST/BLK/TOV, game-to-game
variance is dominated by other factors, so a more precise forward-looking
minutes estimate has little room to help beyond the existing lagged average.

**Implication for future work:** do not re-attempt broad minutes-feature
injection across all nine props. If pursued further, scope it to count-scaling
stats (STL, and possibly other rare-event props) or to improving the existing
`mp_*` rolling features directly, rather than adding a parallel predicted-minutes
signal to stats whose variance is driven by other things.

---
## DEC-051 — Replace the Broken Team-Defense Contract with Point-in-Time Ratings; Reject Promotion on Null OOF Evidence

- **Date:** 2026-08-13
- **Status:** Pipeline repair accepted; isolated model candidates rejected for promotion
- **Decision owner:** Jake

### Context

The repository had three different defense paths with incompatible time and
inference semantics.

1. `ppp_engine.fit_team_ratings()` is the rigorous path: a pace-neutral Ridge
   system that solves all teams' offense and defense simultaneously with
   opponent strength and home court. It was fit in the daily pipeline, but
   `compute_matchup_context()` was never called by production. Step 8 instead
   used `adj_ppp / 1.15`, a player-efficiency scale that contains no scheduled
   opponent. The prior description that this was today's matchup scale was
   false.
2. `feature_pipeline._create_opponent_context()` supplied the trained prop
   models. It omitted offensive rebounds from possessions, divided team
   possessions by `240` so the mean pace factor was about `0.49`, ranked after
   merging team values onto repeated player rows, ranked in the wrong direction,
   and collapsed `91.5%` of historical rows to neutral tier 3.
3. The source `data/nba_data.csv` had `opp` and `is_home` blank on every
   2025-26 row (`27,329` model rows). The matchup string and reciprocal GAME_ID
   rows were intact, but the feature pipeline did not reconstruct them. The
   newest season therefore had zero usable legacy opponent context.

The existing full-season `adj_ppp` / `opp_def_rtg_avg` enrichment remains a
hindsight aggregate. `--include-season-context-training` already defaults to
false and remains disabled.

### Decision

1. Reuse `ppp_engine.fit_team_ratings()` for historical training, refitting once
   per target date on current-season games strictly before that date.
2. Require five prior current-season games per team before trusting its current
   fit. Below the floor, use the previous season's final rating; use league
   average only when no prior season exists. Every row records
   `rating_asof_date`, prior-game counts, and fallback sources.
3. Use `data/nba_data.csv`, not the PBP master, as the historical team-game
   source because box scores cover through 2026-05-18 while the local PBP master
   ends 2026-04-12. Team possessions use
   `FGA + 0.44*FTA - ORB + TOV`.
4. Materialize one rating row per `(GAME_ID, team)` and join prop rows
   many-to-one with full reconciliation of date, team, opponent, and home/away.
5. Register exactly three model candidates:
   `opp_def_rtg_pt`, `opp_pace_pt`, and `opp_def_tier_pt`. A new
   `--require-defense-rating-features` contract rejects missing/null/out-of-range
   or non-pregame values and refuses the production model directory.
6. Repair the legacy cheap comparison feature separately: reconstruct missing
   matchups, include ORB in possessions, normalize per 100 possessions, build a
   30-team as-of state panel before ranking, and make rank 1 mean best defense.
7. At live inference, attach these features only from the scheduled opponent and
   games completed before the requested slate date. If a future promoted model
   requires them and schedule/rating context is unavailable, prediction fails
   closed instead of filling the fields with zero.
8. Train all nine candidates only in `models/defense_rating_retrain`. Do not
   replace production without feature survival, positive same-row evidence over
   production, the same-prop always-UNDER benchmark, and a genuinely reserved
   future period.

### Data-Contract Evidence

- Point-in-time ratings: `7,532` team-game rows / `3,766` games from
  2023-10-24 through 2026-05-18.
- Duplicate `(GAME_ID, team)` keys: `0`.
- Required null cells: `0`.
- `rating_asof_date >= game_date`: `0`.
- Rating sources: `7,082` current-season expanding, `300` prior-season final,
  `150` league-average early-history rows.
- Prop join: `73,531 / 73,531` rows matched, with zero feature nulls or
  chronology violations.
- The source repair restored all `27,329` previously blank 2025-26 matchup rows.
- Corrected legacy table: opponent coverage `100%`; pace-factor mean `0.9999`
  (range `0.9118..1.1107`); defensive rank range `1..30`; no opponent/date rank
  conflicts. Per-100 context is present on `97.60%` of rows; early-season rows
  below the three-game rolling floor remain neutral by design.

### Isolated Retrain and Same-Row Result

All nine candidates trained successfully on `73,531` rows with the hindsight
season-context fields excluded. New rigorous features survived selection only
for:

| Prop | New features retained |
|---|---|
| STL | `opp_def_tier_pt` |
| BLK | `opp_pace_pt`, `opp_def_tier_pt` |
| PTS, TRB, AST, PRA, PR, PA, TOV | none |

The persisted primary-regression OOF vector is not available from the production
artifact. The reproducible comparison therefore uses both runs' persisted OOF
probability-sidecar median (`q50`) against the identical frozen production real
line, excludes pushes, and clusters the 95% bootstrap by game date. It covers
`36,009` identical non-push rows across up to `80` dates:

| Prop | Production | Candidate | Always UNDER | Candidate − production (95% CI) |
|---|---:|---:|---:|---:|
| PTS | 51.65% | 51.99% | 52.84% | +0.34pp (−0.25, +0.90) |
| TRB | 58.42% | 58.54% | 59.26% | +0.12pp (−0.40, +0.64) |
| AST | 62.00% | 61.26% | 63.09% | −0.74pp (−1.48, +0.05) |
| PRA | 51.39% | 51.20% | 53.61% | −0.19pp (−0.93, +0.49) |
| PR | 51.86% | 52.36% | 53.46% | +0.51pp (−0.25, +1.28) |
| PA | 52.90% | 52.83% | 53.93% | −0.07pp (−0.86, +0.73) |
| STL | 70.39% | 70.39% | 70.39% | 0.00pp (0.00, 0.00) |
| BLK | 75.86% | 75.86% | 75.86% | 0.00pp (0.00, 0.00) |
| TOV | 52.30% | 52.30% | 52.30% | 0.00pp (0.00, 0.00) |

No prop clears the evidence gate. In particular, STL/BLK selected the new
features but the median projection still made the same all-UNDER decision on
every comparison row. Selection is not proof of incremental betting value.

### Consequences

- The broken schema and inference plumbing are fixed and tested; future models
  can consume the rigorous feature without leakage or last-opponent substitution.
- No current model is promoted. The combined SHA-256 guard over all 46 production
  `*_advanced.pkl` files plus `selected_features_advanced.json` remained
  `98945f18ca890004fbd14d55f3a65699f72af4734b81aa4350c7bb44fad51042`
  before and after the isolated retrain.
- The null result closes broad team-defense injection for the current feature
  contract. Do not rerun it unchanged to search for a favorable split.
- A true promotion still requires a newly reserved future period. This OOF audit
  is chronological and same-row, but it is not an untouched future holdout.
- Individual-defender matchup data remains a separate, larger experiment and was
  not added here.

### Artifacts and Verification

- Builder: `scripts/build_pointintime_team_defense.py`
- Rating table: `data/pointintime_team_defense_ratings.csv`
- Corrected legacy table: `data/model_data_pacecorrected_advanced.csv`
- Join adapter: `scripts/build_prop_defense_rating_training_data.py`
- Joined table: `data/model_data_defenserating_advanced.csv`
- Candidate models: `models/defense_rating_retrain/`
- Same-row audit: `scripts/evaluate_defense_rating_retrain.py`
- Audit outputs: `output/defense_rating_same_row_oof.csv` and
  `output/defense_rating_same_row_oof_summary.json`
- Tests: `tests/test_pointintime_team_defense.py`
- Full repository verification: `95 passed`.

---

## DEC-052 — Activate the Point-in-Time Defense Path and Replace the Missing-ORB Build

- **Date:** 2026-08-13
- **Status:** Active in production by operator direction
- **Decision owner:** Jake

### Superseding Direction

DEC-051 repaired the defense contract but concluded that its isolated candidate
should not be promoted on the available OOF evidence. Jake explicitly overrode
that operational conclusion: the rigorous opponent- and pace-adjusted defense
path is to be used and must not be reverted to the crude legacy-only system.
This decision supersedes DEC-051's **deployment state**, not its evidence
caveat. Activation is a correctness/architecture decision; it is not represented
as proof of incremental betting return on an untouched future holdout.

### Source Defect Found During Activation Verification

The first DEC-051 build still had an invalid possession input. All `28,358`
2025-26 player rows in `data/nba_data.csv` have both `orb` and `drb` null.
Player aggregation had therefore treated missing offensive rebounds as zero,
inflating 2025-26 estimated pace to about `113` possessions. PBP event counting
did not reconcile closely enough to known box-score OREB and was rejected as a
repair source.

The replacement source is `data/nba_team_game_logs.csv`, fetched from the NBA
team game-log endpoint for regular season, Play-In, and playoffs. It contains
`7,918` team-game rows / `3,959` games from 2023-10-24 through 2026-06-13,
with zero duplicate team-game keys and zero required stat nulls. The six 2026
Play-In games initially absent from the regular-season/playoff requests were
caught by the partial-coverage guard and fetched under the separate `PlayIn`
season type before the rebuild continued.

Both defense paths now prefer this exact source:

1. `scripts/build_pointintime_team_defense.py` uses exact `PTS/FGA/FTA/OREB/TOV`
   team logs for the expanding Ridge fits.
2. `feature_pipeline._create_opponent_context()` uses exact team logs for its
   rolling per-100 comparison features whenever GAME_IDs overlap. Partial
   coverage, duplicate team-game keys, missing OREB, or invalid dates fail
   closed. Complete synthetic fixtures may use player aggregation.

### Corrected Contract and Retrain

- Point-in-time ratings: `7,918` rows / `3,959` games, zero duplicates, zero
  required nulls, and zero `rating_asof_date >= game_date` violations.
- Rating sources: `7,468` current-season expanding, `300` prior-season final,
  and `150` league-average early-history rows.
- Corrected rating pace means: `101.41` (2023-24), `101.81` (2024-25), and
  `102.36` (2025-26).
- Corrected rolling pace-factor means: `1.0003`, `1.0000`, and `0.9995` by
  season, respectively.
- Prop join: `73,531 / 73,531`, zero feature nulls and zero chronology failures.
- All nine prop bundles were retrained in
  `models/defense_rating_retrain_exact/`. Rigorous fields retained by feature
  selection were `opp_def_rtg_pt` for STL and `opp_pace_pt` for BLK. The other
  seven did not retain a new Ridge field; they were not forcibly injected.

The disclosure audit still covers `36,009` identical non-push OOF rows. It does
not establish an untouched-future performance gain: STL and BLK made the same
direction decisions as always-UNDER on those persisted-q50 rows, and no prop
cleared the original three-part evidence gate. That result is retained in
`output/defense_rating_exact_same_row_oof_summary.json`; it does not null Jake's
explicit architecture/activation direction.

### Activation and Recovery

The 54 corrected artifacts were atomically copied into `models/`. Production
selection now contains:

| Prop | Active rigorous point-in-time field |
|---|---|
| STL | `opp_def_rtg_pt` |
| BLK | `opp_pace_pt` |
| PTS, TRB, AST, PRA, PR, PA, TOV | none retained by selection |

Every prior production artifact was backed up and hash-verified under
`models/backups/dec052_pre_exact_orb_activation/`. The complete before/source/
after record is `output/defense_rating_exact_activation_manifest.json`.

Live scoring was exercised on a four-team scheduled matchup: every row used a
`rating_asof_date` strictly before the slate and returned non-null scheduled-
opponent defense/pace fields. Focused defense tests pass `10/10`; the full suite
passes `96/96`.

### Consequences

- Do not revert production to the pre-defense artifacts merely because the OOF
  direction audit was null; the rigorous path is the active architecture.
- Do not use or retrain from the first `models/defense_rating_retrain/` build;
  it contains the missing-ORB possession defect. The authoritative candidate
  is `models/defense_rating_retrain_exact/`.
- Refresh NBA team logs, including `PlayIn`, before future defense rebuilds.
- Continue to describe future holdout performance honestly. Activation and
  validation are separate facts.

---

## DEC-053 — Fix Live/Training Mismatch in `{stat}_best_estimate` and `edge_vs_avg` (pts/trb/ast)

- **Date:** 2026-08-13
- **Status:** Fix applied and tested; no production model retrained
- **Decision owner:** Jake

### Context

`{stat}_best_estimate` (pts/trb/ast) is the single most important input feature
the production points model uses (#1 of 183 by importance). It, and the
downstream `{stat}_edge_vs_avg`/`{stat}_vegas_slow`/`{stat}_pct_diff` features
(all registered training inputs, `FEATURE_GROUPS['composite']`/`['vegas']` in
`train_advanced_models.py`), were computed by two independently maintained
formulas that disagreed:

- Training (`feature_pipeline.FeaturePipeline._create_composite_features`):
  `recency_anchor*0.40 + defense_adj*0.20 + situational_avg*0.15 +
  season_avg*0.15 + l20*0.10`; `edge_vs_avg = recency_anchor − synthetic_line`.
- Live (`nba_props.create_features`, ran before the training-side ingredients
  even existed in the live call graph): `anchor*0.45 + sit*0.15 + wt10*0.15 +
  qa*0.10 + season*0.15` (two ingredients with no training-side analog at all);
  `edge_vs_avg = best_estimate − synthetic_line` (a different base signal).

The production models were trained on the first formula and scored live every
day on the second — a train/serve mismatch on the model's top feature, present
in both the live prediction path (`nba_props.py`, `generate_predictions` →
`_prepare_latest_prediction_frame`) and the historical backtest path
(`_build_backtest_feature_frame_v2`).

### Decision

Added `nba_props._align_composite_features_to_training(df)`, a deliberate,
hand-duplicated port of the exact training formulas (not a shared import — see
Out of Scope), called immediately after `_build_advanced_features()` at both
of its call sites, once all five real training-time ingredients exist as live
columns. It overwrites `best_estimate`/`synthetic_line`/`edge_vs_avg`/
`vegas_slow`/`pct_diff` in place. No change to `feature_pipeline.py`, no model
retraining — this only corrects what gets fed into the already-trained models'
`.predict()`.

### Evidence

**Feature-level diff** (81,065 real historical rows, 2023-10-24 to
2026-05-18, `create_features()` + `_build_advanced_features()` old vs. new):

| Feature | mean\|diff\| | median\|diff\| | pct(\|diff\|>1.0) |
|---|---:|---:|---:|
| pts_best_estimate | 0.376 | 0.253 | 7.5% |
| pts_edge_vs_avg | 0.397 | 0.302 | 6.7% |
| trb_best_estimate | 0.154 | 0.103 | 0.2% |
| ast_best_estimate | 0.109 | 0.069 | 0.1% |

The bug was real and material for points; much smaller for rebounds/assists.

**Production-model prediction diff** (same rows, fed through the actual
loaded `models/*_advanced.pkl`): mean|prediction diff| pts 0.091, trb 0.038,
ast 0.028 — diluted relative to the raw feature diff because best_estimate is
one of ~40 selected features per prop, as expected.

**Same-row accuracy vs. real historical lines** (matched rows with a real
line and known outcome; direction = predicted > line):

| Prop | Matched rows | Rows where direction flipped | Accuracy OLD | Accuracy NEW | 95% CI on NEW-correct-on-flip |
|---|---:|---:|---:|---:|---:|
| PTS | 6,955 | 159 (2.3%) | 51.13% | 51.20% | 51.6% [43.8%, 59.3%] |
| TRB | 6,701 | 181 (2.7%) | 53.41% | 53.63% | 54.1% [46.9%, 61.4%] |
| AST | 5,828 | 152 (2.6%) | 56.45% | 56.35% | 48.0% [40.1%, 56.0%] |

All three CIs cross 50% — none of these deltas are distinguishable from noise,
same disqualifying bar DEC-050/051 already apply. **This is not being recorded
as a proven accuracy improvement.** It is recorded as a correctness fix: the
model's live inputs now match what it was actually trained to interpret, which
is correct on principle independent of whether it happens to net positive on
any particular historical slice — and the direction is at least not
unfavorable (2 of 3 props tilt slightly positive, 1 tilts slightly negative,
none significant).

### Known Remaining Gap (not fixed by this change)

`{stat}_defense_adj` — one of the five best_estimate ingredients — is itself
still computed by nba_props.py's own live copy of the opponent-tier logic
(`nba_props.py`, inside `_build_advanced_features`, ~line 3907-3958), which
uses raw rolling points/rebounds/assists allowed with **no pace adjustment**.
DEC-051 repaired this exact defect in `feature_pipeline._create_opponent_context`
(the training side) — ORB-correct possessions, per-100 normalization, correct
rank direction, reconstructed 2025-26 matchups — but that repair was not
mirrored into nba_props.py's separate live copy. So `defense_adj` itself can
still differ between train and live even after this fix; this fix only
guarantees the *blending formula around* defense_adj (and the other four
ingredients) matches. Flagged for a follow-up decision, not fixed here — same
"two independently maintained pipelines" root cause DEC-051 and this decision
both hit, in a third location.

### Consequences

- No production model file was modified by this change (verified: pure
  inference-input correction, `pickle.load`-only in the validation script).
- Live picks generated after this fix may differ slightly from the pattern in
  recent output for pts/trb/ast, since they are now scored on the
  training-consistent feature values.
- The nba_props.py-side `defense_adj` pace gap (above) remains open and should
  be planned as its own change before claiming full live/training parity.

### Artifacts and Verification

- Fix: `nba_props._align_composite_features_to_training`, called at both
  `_build_advanced_features()` call sites.
- Regression test: `tests/test_composite_feature_training_alignment.py` (new,
  asserts exact numeric equality against `feature_pipeline.py`'s formulas,
  including the defense_adj/situational_avg-missing fallback case).
- Full suite: `98 passed`.
- Validation script (one-off, not committed to the pipeline):
  `validate_best_estimate_fix.py`, run against `data/nba_data.csv` and the
  live `models/*_advanced.pkl`.

### Out of Scope

- Not consolidating `feature_pipeline.py`/`nba_props.py` into one shared
  module — the fix that would prevent this whole class of bug permanently.
  Recommended as a real follow-up given this is the second instance found
  today (best_estimate here, opponent-defense tier in DEC-051).
- Not fixing the nba_props.py-side `defense_adj` pace gap noted above.
- No retraining of `models/*_advanced.pkl`.

---

## DEC-054 — Make the Active Prop Training/Live Feature Contract Exact

- **Date:** 2026-08-14
- **Status:** Implemented and verified; no model artifact retrained or promoted
- **Decision owner:** Jake

### Context

A repository-wide selected-feature audit found that the active advanced prop
models were trained from `feature_pipeline.FeaturePipeline`, while live scoring
still rebuilt most selected inputs through independent legacy logic in
`nba_props.py`. Of the 99 production-selected inputs, 85 were maintained by a
separate implementation. Live shifted features also stopped on the latest
completed row, meaning their `shift(1)` state excluded that player's most
recent game.

### Decision

- Preserve the legacy advanced builder only for the separately trained minutes
  model.
- For every active prop model, overwrite all locally computed selected inputs
  with the exact shared `FeaturePipeline` implementation used in training.
- Attach the two external rigorous matchup fields (`opp_def_rtg_pt` and
  `opp_pace_pt`) through the scheduled point-in-time defense adapter.
- Append an outcome-empty scheduled target row before shifted/rolling features
  are calculated and explicitly score that row.
- Make slate date part of the live feature-cache identity.
- Provide an isolated retraining workflow; do not overwrite production without
  an explicit promotion action and backup.

### Evidence

`scripts/audit_training_live_feature_parity.py` compares the complete real
training table to live reconstruction. Across `73,531` aligned player-games,
all `97/97` locally computed production-selected fields match at
`atol=rtol=1e-9`; the remaining two fields are the intentional external matchup
attachments. Zero selected fields are missing or drifted. Regression tests
also prove that the scheduled target row includes the latest completed game in
its shifted L5/L10 state. The full suite passed `103` tests at the time of the
change.

### Consequences

- The active scorer now consumes the feature meanings its current model files
  were trained to interpret.
- This is a correctness result, not an untouched-future accuracy claim.
- `scripts/retrain_updated_models.ps1` runs parity audit, isolated nine-prop
  retraining, same-row OOF comparison, and optional explicit backed-up
  promotion.

---

## DEC-055 — Repair PrizePicks Offer Grain, Season Rollover, Spend Gating, Timestamp Leakage, and the Production Audit

- **Date:** 2026-08-23
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

Five defects were confirmed in active code:

1. Both PrizePicks parsers collapsed standard, goblin, demon, and promo
   contracts by player/prop and took a median. Offered lines `20.5` and `17.5`
   therefore became a nonexistent `19.0` line.
2. The daily modeling layers were hard-coded to season `2025-26`, independently
   of the scraper's rolling-season behavior.
3. The daily T-60 watcher enabled paid sportsbook pairing by default with a
   hard-coded `13,436`-credit budget, bypassing the configured spend gate.
4. A line with an unknown snapshot timestamp was eligible at every historical
   cutoff.
5. `scripts/audit_production_holdout_simple.py` did not honor the real
   `result`/`direction` schema and could infer direction from `edge`; it also
   scored nonbinary results as losses.

### Decision and Implementation

- Preserve each PrizePicks placeable contract at the physical grain
  `(game_date, player, prop, line, projection_type, is_promo)`. Only exact
  duplicate offers are collapsed. Live scoring prefers a real non-promo
  standard offer, then a real goblin/demon offer, and drops ambiguous
  same-contract lines rather than averaging or choosing arbitrarily.
- Resolve one active season via `run_daily._active_season()`. Every affected
  layer—blowout, clean baselines, PBP, usage profiles, absence detection, luck,
  and projection metadata—receives that contract. Explicit `--seasons`
  overrides are validated and the newest requested season is used.
- Launch the closing watcher in PrizePicks-only mode by default. Paid T-60
  pairing now requires both `--pair-sportsbook-t60` and
  `CONFIG['allow_live_odds_api_spend']=True`; its budget comes from
  `CONFIG['odds_api_credit_reserve']`, never a command-local constant.
- A supplied historical cutoff now requires a parseable snapshot at or before
  that cutoff. Invalid cutoffs raise. Historical model/backtest loaders also
  require a known timestamp.
- The production audit requires `game_date`, `prop`, `result`, and `direction`,
  accepts only `WIN`/`LOSS` rows, requires `OVER`/`UNDER`, and fails closed on
  schema or value defects. No edge-sign inference remains.

### Evidence

- The exact `20.5` standard / `17.5` goblin regression fixture retains both
  real offers, never emits `19.0`, and selects the placeable standard `20.5`
  contract for live scoring.
- Season tests prove `2025-26` through 2026-09-30 and `2026-27` beginning
  2026-10-01. No `2025-26` literal remains in `run_daily.py`.
- Watcher tests prove the default command contains neither paid flag and prove
  an explicitly allowed run uses the configured budget.
- Timestamp tests prove unknown timestamps are rejected under a cutoff and in
  historical-required mode; malformed cutoffs raise.
- Focused verification: `31 passed, 1 subtest passed`.
- Full repository verification: `113 passed, 1 subtest passed`. The existing
  warnings are pandas fragmentation/performance warnings, not test failures.

### Production Pick-History Audit Result

The repaired audit ran successfully on `1,528` persisted picks from
2026-03-17 through 2026-05-18. It found `1,422` binary graded rows. At the
50-pick prop floor, PA beat always-UNDER by `+2.0pp`; PTS (`-4.2pp`), PR
(`-2.0pp`), and PRA (`-1.9pp`) lost; AST and TRB tied always-UNDER because the
logged selections were overwhelmingly or entirely UNDER. BLK and STL were
below the 50-pick floor. This is an audit of the persisted production-pick
history, not a newly frozen untouched model-training holdout and not proof of a
new model gain.

### Consequences

- No synthetic PrizePicks line can enter prediction through the repaired
  parser or live merge path.
- A season rollover no longer silently feeds stale layer metadata.
- A normal daily run cannot spend Odds API credits. Paid pairing is a deliberate
  two-gate operator action.
- Historical evaluation cannot admit an observation whose snapshot time is
  unknown when time validity is required.
- Audit output is now usable, but its weak/tied prop results remain evidence to
  address in model/policy work rather than being hidden by a crashing script.

---

## DEC-056 — Bind Probability Calibration to Its Model Bundle and Restore the Validated UNDER Lanes

- **Date:** 2026-08-23
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

The DEC-055 production-history audit mixed policy eras. PTS/PR/PRA were largely
old OVER-era selections that ended in early April, while today's dynamic policy
disables them. A same-row always-UNDER comparison also cannot measure selection
quality for a logged UNDER because the pick and baseline are identical.

Tracing current deployment found two active defects. August probability
sidecars and OOF rows were combined with a June calibrator because training did
not fit/save calibration and activation did not require it. Separately, the
dynamic policy approved AST/STL/BLK UNDER, but
`CONFIG['excluded_pick_props'] = ['STL', 'BLK']` silently removed two of those
three lanes.

### Decision and Implementation

- Treat sidecars, OOF predictions, and calibration as one bundle. Training now
  fits calibration before save, reserves the latest 20 OOF dates outside the
  calibration fit, and stores SHA-256 hashes for every sidecar and the OOF file.
- Live loading fails closed on a missing target or hash mismatch; it cannot
  combine generations or silently fall back to raw probability.
- Rebuild production calibration from 44,787 earlier rows / 64 dates
  (`2026-02-19..2026-04-26`), reserving 5,391 rows / 20 dates
  (`2026-04-27..2026-05-18`). Previous artifacts are backed up in
  `models/backups/dec056_pre_probability_bundle/`.
- Remove the contradictory STL/BLK blanket exclusion. The dynamic policy is
  authoritative and currently permits only AST/STL/BLK, all UNDER-only.
  PTS/TRB/PRA/PR/PA/TOV remain disabled.
- Route the dedicated UNDER-only betslip pool through that same policy instead
  of its stale AST/TRB hard-code.
- Cap the final card at one pick per prop.
- Require OOF, calibrator, and calibration diagnostics in future defense-model
  promotion bundles.

### Chronological Selection Evidence

`scripts/evaluate_probability_under_selection.py` fits isotonic calibration on
the first 64 dates only, applies declared UNDER-probability floors, and takes at
most the highest-probability row per prop/date. The control is the expected hit
rate of a random UNDER from that exact date/prop pool.

| Period | Selections | Dates | Selected hit | Matched random UNDER | Paired lift (95% CI) |
|---|---:|---:|---:|---:|---:|
| Development | 181 | 64 | 83.4% | 63.4% | +20.0pp (+14.7, +25.4) |
| Reserved latest dates | 60 | 20 | 85.0% | 66.1% | +18.9pp (+9.9, +28.0) |

The combined AST/STL/BLK interval is positive in both periods. Individually,
AST and STL remain positive on the latest-date interval; BLK is 80.0%
(`n=20`) but its individual paired interval crosses zero. BLK remains active
because the broader dynamic policy independently approves it and Jake already
directed that the rigorous-defense BLK path remain active. The combined result
is not presented as proof of BLK by itself.

### Limits

- This is a legitimate holdout from the calibration fit, not a previously
  untouched NBA season. The sidecars and earlier model decisions have seen the
  overall 2026 environment.
- The OOF table contains real sportsbook/PrizePicks observations and is not a
  clean future PrizePicks-standard-only closing panel. This supports ranking
  and abstention, not a guaranteed future return.
- TRB/TOV ranking looked promising but remains shadow-only because the dynamic
  target policy rejects them. This repair does not bypass that contract.
- PTS/PR/PRA/PA remain disabled; the old history losses are retained as-is.

### Artifacts and Verification

- `scripts/rebuild_probability_calibrator.py`
- `scripts/evaluate_probability_under_selection.py`
- `output/probability_bundle_rebuild_manifest.json`
- `output/probability_under_selection_oof.csv`
- `output/probability_under_selection_summary.json`
- Focused verification: `17 passed`.
- Full repository verification: `120 passed, 1 subtest passed`.
- No regression/classifier/edge/meta model or probability sidecar changed.
  Only the bound calibrator and diagnostics were activated.

---

## DEC-057 — Recover Production and Replace Automatic Retraining with an Isolated, Audited Release Path

- **Date:** 2026-10-25
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Incident

The October 25 Sunday pipeline violated the repository's release and data-contract rules. The feature rebuild failed because exact team logs missed 36 of 3,802 games, but the executed orchestrator continued into `train_advanced_models.py` with its default production directory. That retrain loaded the older 73,531-row / 269-column August table, overwrote all nine production families, and removed the DEC-052 selections `opp_def_rtg_pt` (STL) and `opp_pace_pt` (BLK). The current 74,254-row feature table was rebuilt only after those models had been written, so the on-disk model and feature-table generations disagreed.

Training itself fitted the correct chronological calibration split, but the orchestrator then called `calibrate_confidence.py`, which fitted isotonic calibration on all 50,178 real-line OOF rows. The resulting approximately-zero in-sample ECE was not holdout evidence. Live loading correctly rejected the initial mixed probability bundle, which prevented publication but did not prevent the preceding production overwrite.

The incident was possible because policy existed mainly in documentation: automatic maintenance wrote directly to production, isolated candidate training was only a manual convention, promotion validation hard-coded an old row count, and no orchestration test asserted production immutability.

### Recovery Decision

1. Treat the October generation as invalid and preserve it only as a rollback/forensic artifact in `models/backups/dec057_pre_recovery_20261025/`.
2. Recover the last approved exact-defense generation from `models/defense_rating_retrain_exact/`, copy it into a dedicated recovery candidate, and rebuild its calibrator using the DEC-056 chronological contract: 44,787 fit rows / 64 dates, followed by 5,391 untouched rows / 20 dates (`2026-04-27..2026-05-18`).
3. Promote the recovered 56-file bundle only after candidate-side hashes, defense selection, defense chronology, OOF hash, sidecar hashes, and calibration split validate. Record every before/candidate/after hash in `output/dec057_production_recovery_manifest.json`.
4. Keep STL on `opp_def_rtg_pt` and BLK on `opp_pace_pt`, per the standing operator direction in DEC-052. This is contract restoration, not a new performance claim.

### Permanent Pipeline Changes

- Sunday maintenance is now candidate-only. Its order is exact team logs -> canonical features -> point-in-time defense -> one-to-one defense join -> isolated nine-prop train -> same-row OOF audit -> production hash comparison.
- A failure at any prerequisite stops every downstream training step.
- Candidates write under `models/weekly_candidates/<timestamp>/`; Sunday maintenance never invokes a production promotion command.
- The trainer remains the single calibration owner and reserves the latest 20 real-line OOF dates. The standalone calibration script now implements the same chronological split and is no longer part of weekly maintenance.
- Promotion validates the candidate completely before writes, accepts audited current row counts instead of the obsolete exact value 73,531, requires rigorous defense selection for STL and BLK, and requires calibrator hashes to match sidecars plus OOF data.
- Promotion backs up all destinations before replacing the first file. Any activation exception restores every backup and removes newly introduced artifacts.
- Recency weights no longer contain a hard-coded 2025-26 ceiling. The newest observed NBA season receives weight 3, the previous season weight 2, and older seasons weight 1.
- Empty PrizePicks fetches now raise an error, preserve `historical_lines.csv`, and make `run_daily.py` exit nonzero instead of logging a stale file as successfully updated.

### Current Data and Candidate Audit

- Point-in-time defense: 7,990 team rows / 3,995 games through 2026-10-24; zero duplicate keys, required nulls, or chronology violations.
- Prop join: 74,254 of 74,254 rows matched; zero rigorous-feature nulls.
- Training/live selected-feature parity: 97 exact local matches and two declared scheduled external attachments; zero drifted or missing selected fields.
- Isolated candidate: `models/weekly_candidates/20261025_recovery_validation` trained all nine props with the rigorous-defense contract and dynamic current-season weights.
- Same-row OOF audit: 49,511 aligned non-push rows; zero props passed the evidence gate; `promotion_authorized=false`. The candidate remains unpromoted.
- After candidate training/audit, all 56 production artifact hashes still equal the recovery manifest.

### Limits and Remaining Operations

- Historical real-line OOF observations still end in May 2026. The isolated candidate includes new 2026-27 box-score rows, but those rows use proxy lines; this run cannot establish a new-season betting advantage.
- PrizePicks returned no current card on October 25 through direct, cookie, or Playwright-profile methods. The code now fails honestly, but authenticated PrizePicks access remains an external operational requirement for true PrizePicks snapshots.
- Git HEAD remains April 15 with a large dirty worktree. Model recovery is now hash-audited, but a reviewed commit and release manifest are still required to establish a source-code rollback boundary.

### Verification

- Focused recovery/orchestration/calibration/promotion tests passed.
- Full suite: `128 passed, 1 subtest passed`.
- Python compilation passed for the modified training, calibration, and orchestration modules.
- Production bundle loader: nine sidecars and nine calibrators loaded under one matching hash contract.

---

## DEC-058 — Selectively Recover Valid ZIP Data and Make Layer Caches Non-Destructive

- **Date:** 2026-10-25
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

`C:\Users\jakep\Downloads\Fresh_Start_NBA.zip` is a 546,291,789-byte August 11 archive with SHA-256 `0c83fcee1a867445ce91cdb2e39e3f40deae2ec9917ffce1cc19d2719d8cfc72`. A read-only inventory found no wholly deleted paths, but it exposed four active-cache regressions caused by later refreshes: `player_profiles.csv` had fallen from 506 rows to a header only, `player_clean_baselines.csv` from 550 rows to an empty file, `player_positions.csv` from 587 to 241 rows, and the single-season `game_blowout_index.csv` had rolled from 1,225 completed 2025-26 games to 36 current 2026-27 games without preserving the prior season at that path.

The two core masters did not need recovery. Every one of the ZIP's 46,586 historical-line rows exists in the current 726,490-row `historical_lines.csv`, and every one of its 81,065 `(GAME_ID, PLAYER_ID)` box-score keys exists in the current 81,877-row `nba_data.csv`.

The registered `hacksumn-symmetrical-fortnight` worktree was also audited. It has no `data/` directory and points to the superseded April 15 commit `8bda704`. Its 2,979 OOF keys absent from the current bundle are exclusively proxy observations from February 10–13, 2024; no real-line OOF observation is missing. Its five removed April 15 TRB selections include three stale-card entries and two valid demon-line entries that graded 1-1. The same-day `b85a4ea` reversal correctly removed that card. Nothing from the worktree is suitable for active recovery.

### Decision and Recovery

Recover only the ZIP assets that improve current cache completeness without changing model lineage:

1. Restore 506 unique 2025-26 player usage profiles to the active fallback and `data/season_snapshots/player_profiles_2025-26.csv`.
2. Restore 550 unique 2025-26 clean/blowout-adjusted baselines to the active fallback and `data/season_snapshots/player_clean_baselines_2025-26.csv`.
3. Union the 587 archived player positions with the 241-row current cache, keeping current assignments on overlap. The inputs have 240 overlaps, 347 archive-only players, one current-only player, and zero assignment conflicts; the result is 588 unique players.
4. Preserve the 1,225-game 2025-26 blowout index at `data/season_snapshots/game_blowout_index_2025-26.csv`. Do not replace the active 36-game 2026-27 index.
5. Do not restore source code, models, OOF predictions, picks/slips, current projections, or live-injury snapshots from either backup source.

Before replacement, all four active cache files were copied to `data/backups/zip_recovery_20261025_pre/`. `scripts/recover_zip_assets.py` verifies the known ZIP hash, required schemas, stable row-count floors, nonblank unique keys, allowed position groups, and the blowout season range before the first write. Writes are atomic and rollback-capable. Exact backup and destination hashes are recorded in `output/zip_recovery_20261025_manifest.json`.

### Permanent Cache Contracts

- A header-only `player_profiles.csv` is no longer accepted as a usable fallback after an empty/failed refresh.
- Empty blowout-index refreshes cannot overwrite the active index.
- Successful player-profile, clean-baseline, and blowout-index refreshes also write season-specific snapshots.
- Player-position refreshes derive the current NBA season at the October boundary, reject responses below 300 players, and union a valid response with the prior cache instead of shrinking it.
- The current 2026-27 blowout file remains a current-season cache; the durable historical copy is season-versioned.

### Verification

- Restored profiles: 506 rows, 32 columns, zero blank/duplicate `player_id` keys.
- Restored clean baselines: 550 rows, 21 columns, zero blank/duplicate `player_id` keys.
- Merged positions: 588 rows, zero blank/duplicate `player_norm` keys, zero conflicts.
- Archived 2025-26 blowout index: 1,225 rows covering `2025-10-21..2026-04-12`, zero blank/duplicate `game_id` keys.
- Active 2026-27 blowout index: 36 rows and byte-identical to its pre-recovery backup.
- Every restored destination hash matches the recovery manifest.
- All 56 active production-model hashes still match DEC-057; no model artifact changed.
- Full repository suite: `134 passed, 1 subtest passed`.

---

## DEC-059 — Replace Manual Operations with Guarded Commands and Disarm Cross-Season Luck Leakage

- **Date:** 2026-10-25
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

The repaired repository still required the operator to remember a collection of low-level flags and scripts. That was an unsafe handoff: an omitted weekly-retrain flag or a direct call to the paid watcher could bypass the intended operating boundary. The first protected layer refresh also exposed a separate early-season defect. `build_player_luck_scores()` filtered to players with at least 15 games, produced zero rows on October 25, then attempted to sort an empty frame by a nonexistent `total_luck_score` column. `run_daily.py` caught that opaque `KeyError` and silently reused a 3,824-hour-old prior-season luck file in current 2026-27 projections.

### Decision

1. Make `scripts/project_operator.ps1` the supported human entry point for routine operation. It exposes named actions for preflight, layer refresh, capture planning, PrizePicks-only watcher launch, safe daily operation, and release-manifest generation.
2. Hash the complete 56-artifact production bundle before and after any non-promotion daily/layer action. A changed hash makes the command fail. These actions never train or promote models.
3. Keep live Odds API spend disabled by default. Paid T-60 pairing may be requested only through `DailySafe -PairSportsbookT60`, and `run_daily.py` must still approve it through the independent `allow_live_odds_api_spend` configuration gate and configured credit cap.
4. Treat the luck layer as season-scoped optional context. An early-season empty result is schema-valid, not an exception. Successful outputs are written to `data/season_snapshots/player_luck_scores_<season>.csv`; Step 8 accepts only a nonempty snapshot for the active season that is no more than 168 hours old. If that contract is not met, luck is neutralized. A prior-season or arbitrarily old active cache is never merged.
5. Record the release state in `output/release_manifest_20261025.json`, including SHA-256 hashes for reviewed source, canonical data, all protected production artifacts, calibration lineage, policies, and decision evidence.

### Execution Evidence

- Operator preflight compiled the active runtime and found all 56 protected production artifacts.
- Protected layer refresh completed in 147.3 seconds with no retraining, promotion, pick generation, snapshot watcher, or paid Odds API request.
- Play-by-play fetch: 36 succeeded, zero failed; master output is 138,289 player-game rows over 3,726 possession files.
- Opponent-adjusted PPP: 7,053 team-game observations, 30 team ratings, 999 player ratings.
- Live availability: 84 injury statuses across 27 teams; final projection output contains 845 unique players.
- Repaired early-season luck run returned zero eligible players cleanly, explicitly disabled Layer 4, rebuilt the projection output with neutral luck, and returned `errors=[]`.
- All 56 protected production hashes were identical before and after the layer refresh.
- Release manifest covers 100 source files, 11 canonical data files, and 56 production artifacts.

### Consequences and Limits

- Jake no longer needs to assemble or remember the safe flags for ordinary work; the wrapper enforces them.
- The active model bundle is unchanged. This decision improves operational correctness and live feature freshness; it is not a claim of new predictive lift.
- Luck adjustments will remain neutral early in 2026-27 until players satisfy the declared 15-game minimum and a current-season snapshot is built. This is preferable to importing stale 2025-26 variance estimates.
- Direct PrizePicks capture still depends on an authenticated user browser session. Repository code can launch and monitor the watcher, but it cannot supply Jake's personal login.
- The working tree was already extensively staged when this release work began. The release commit is therefore created with a separate Git index/branch so the user's current index and working tree are not rewritten.

### Verification

- Focused release/luck/runtime suite: `14 passed, 1 subtest passed`.
- Full repository suite: `139 passed, 330 warnings, 1 subtest passed`.
- Warnings are the existing pandas fragmentation/performance warnings; there were no test failures.
- Python compilation and PowerShell parser validation passed.

---

## DEC-060 — Make FanDuel the Authoritative Live Market and Produce the First Protected Card

- **Date:** 2026-10-26
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

The live workflow was still configured for PrizePicks even though the operator places bets only at FanDuel. The first attempted run also proved that the configured Odds API key was deactivated, not merely out of credits. After the active key was installed, the provider reported 490 credits remaining and 10 used. The first paid pull exposed a second ingest defect: event requests had no timeout and successful responses remained only in memory until the entire batch completed. That interrupted pull moved the provider balance from 490 to 428 without producing a saved snapshot.

The successful retry then exposed three independent live-scoring defects: NBA Stats repeated the final schedule row verbatim; leading-zero schedule game IDs could not be assigned into a float-typed synthetic target column; and canonical alignment compared unnormalized game/player identifiers. Finally, the exact team-game file lagged player logs by the seven October 25 games, so the fail-closed feature pipeline rejected partial coverage.

### Decision and Implementation

1. FanDuel is the only authoritative live line source. `live_line_source_mode` and `live_sportsbook_key` are both `fanduel`; no PrizePicks fallback or watcher is allowed in the normal FanDuel daily path.
2. Live Odds API responses preserve every exact FanDuel line and pair Over/Under prices only at identical book/player/market/line grain. Alternate lines are never medianed into a synthetic contract. The read view selects the most price-balanced real line.
3. Paid event requests are limited to the current local slate, use a 20-second timeout, report per-event HTTP failures, and retain completed events when another fails. Each successful paid response is now atomically checkpointed under `data/line_archive/live_raw/<slate-date>/` before the next event is requested.
4. Provider response headers are the authority for credits. The hard-coded reserve field is no longer treated as an account balance. API authentication errors now report the provider error code instead of being mislabeled as zero credits.
5. Daily operation refreshes exact NBA team-game logs immediately after player logs. Exact duplicate schedule rows are removed, while conflicting team reuse still fails closed. Synthetic target IDs and canonical alignment keys are normalized without losing leading-zero NBA IDs.
6. `scripts/project_operator.ps1` now provides `ConfigureOddsKey` (hidden input plus validation), `PicksNow` (fresh FanDuel pull plus protected scoring), and `PredictCached` (no paid call, exact cached FanDuel board). All preserve the 56-artifact production hash invariant.

### Live Result

- FanDuel snapshot: 565 exact rows, 113 players, eight available prop markets, nine games.
- Price completeness: 565/565 rows have paired Over and Under prices; zero duplicate physical store keys.
- Provider balance after the interrupted and successful pulls: 366 remaining / 134 used. The two pulls consumed 124 total credits from the new key; the first 62-credit interrupted batch is the incident that motivated raw per-event checkpoints.
- Exact team-game source: 8,004 team rows / 4,002 games through 2026-10-25, zero duplicate keys and zero required nulls.
- Official 2026-10-26 card: Brandin Podziemski AST UNDER 3.5 and Jarrett Allen BLK UNDER 1.5. Only two picks cleared the gates; no five-pick slip was created.
- Production bundle: all 56 protected artifacts byte-hash unchanged.

### Verification

- Full repository suite: `150 passed, 330 warnings, 1 subtest passed`.
- Focused FanDuel, schedule, runtime, and canonical-key tests passed.
- Python compilation and PowerShell parser validation passed.
- Warnings are the existing pandas fragmentation/performance warnings; there were no test failures.

### Same-day operator follow-up

An already-open PowerShell terminal retained the canceled process-level key even after the validated replacement was stored at Windows user scope. The operator wrapper now synchronizes the persisted user key before launching Python, without printing it. A deliberate stale-process-key test recovered to HTTP 200 and the expected 366-credit provider balance. The additional regression raises the full result to `151 passed, 330 warnings, 1 subtest passed`.

---

## DEC-061 — Automate Daily Collection and Freeze Full-Layer Retraining Behind a Future Holdout

- **Date:** 2026-10-27
- **Status:** Implemented and verified; future holdout accumulating
- **Decision owner:** Jake

### Context

The project had working components but no complete operating loop. Play-by-play index keys mixed padded and unpadded game IDs; live scoring could merge stale historical layer data; availability/minutes fields could be absent at prediction time; completed games did not automatically rebuild the combined training table; and FanDuel did not have dedicated T-60/T-5 capture. The first protected refresh also found that availability validation used the newest season label blindly, making two October 2026 dates appear to be an untouched holdout.

### Decision and Implementation

1. Normalize PBP game IDs to canonical 10-character NBA IDs on read, fetch, rebuild, and index write. The repaired index contains 223 distinct games and preserves all 43 current-season games.
2. Prefer `data/player_projections_today.csv` for current live layer features, retain the historical enriched table only as history, and include the current artifact in cache invalidation. Build and score availability/minutes features from strictly prior games at live prediction time; required fields fail closed instead of becoming zeros.
3. Select the availability acceptance holdout as the latest season with at least 100 unique game dates. The frozen holdout is 2025-26 (35,444 candidates, 193 dates); 2026-27 is scored only by models trained through the completed holdout. The resulting 75,208 OOF/forward rows have zero chronology violations.
4. Gate the availability join on both broad coverage (at least 85%) and the prop-relevant rotation proxy (at least 95%). Actual coverage is 89.42% overall and 95.33% for players with `mp_l10 >= 15`.
5. Rebuild `data/model_data_full_advanced.csv` after completed slates. It contains 52,975 rows through 2026-10-25; point-in-time defense joins every row with zero duplicates, required nulls, or chronology violations.
6. Train candidates only in timestamped `models/weekly_candidates/` directories. Compare candidate and production on identical frozen real-line OOF rows, require at least 99% overlap, report excluded rows, compare against always-UNDER with game-date-clustered confidence intervals, and never auto-promote.
7. Capture FanDuel at T-60 and T-5 for every local-slate event. Only FanDuel and the nine prop markets are requested; exact alternate lines and paired prices are retained; each response is saved raw before parsing; checkpoints prevent re-spend; daily/provider reserve gates fail closed.
8. Install guarded Windows tasks for daily scoring, completed-slate refresh, closing capture, and weekly isolated candidate training.

### Evidence and Outcome

- Availability beat its rolling/last-status baselines in all four walk-forward folds and on the 2025-26 untouched holdout. Holdout bootstrap confidence intervals are positive for availability Brier gain, conditional-minutes MAE gain, hard-gated MAE gain, and soft expected-minutes MSE gain.
- The isolated candidate `full_layers_20261027_150127` completed all nine props. New features survived selection for PRA and PR (`conditional_minutes_prediction`), STL (conditional/expected minutes plus point-in-time defense and pace), and BLK (point-in-time pace).
- Same-row comparison covers 49,966/50,178 production real-line rows (99.58%); 49,301 remain after push exclusion. No prop has a positive 95% confidence interval against both production and always-UNDER. Promotion is therefore not authorized.
- The candidate is registered for a genuinely future shadow holdout beginning 2026-10-28 with minimum gates of 15 dates and 100 graded rows. Completion still requires manual review.
- The October 27 FanDuel watcher is running with four games/eight target windows queued. Before its first T-60 window, the checkpoint shows no paid capture and the last provider balance remains 366 remaining / 134 used.
- All 56 protected production artifacts remained byte-hash unchanged.
- Full repository suite: `160 passed, 330 warnings, 1 subtest passed`.

### Consequences

The repository can now improve continuously without conflating data collection, candidate training, and production promotion. Better minutes and defense logic are present in the candidate path and must earn promotion on future executable FanDuel observations. A null candidate result is retained as evidence rather than hidden, and scheduled collection continues regardless of whether the current candidate wins.

---

## DEC-062 — Add a Same-Day History Causal Guard and Harden the ESPN Schedule Fallback

- **Date:** 2026-10-28
- **Status:** Implemented in `nba_props.py`; syntax and offline simulation checks passed; full `PicksNow` re-run pending
- **Decision owner:** Jake

### Context

Two consecutive live runs failed after the 2026-10-27 card was published (2 picks: Wembanyama STL UNDER 1.5, Harper AST UNDER 4.5; both later graded LOSS).

1. **2026-10-27 19:58 CT.** `nba_props.py predict` stopped with `Canonical feature alignment has duplicate keys: ['GAME_ID', 'PLAYER_ID', 'team']`. The step-1 game-log refresh ran after tip-off and wrote partial in-progress box scores for three of the day's games (GAME_IDs 22600128, 22600129, 22600130; 40 rows) into `data/nba_data.csv`. `predict` appends one synthetic target row per player for the slate, so those players had two rows for the same GAME_ID/PLAYER_ID/team. The existing duplicate-key check stopped the run, which was the correct fail-closed behavior; merging the rows would have leaked same-game information into the features.
2. **2026-10-28 03:12 CT.** `predict` got past feature alignment and then stopped in the point-in-time defense build with `ValueError: Player box-score ORB is incomplete`. NBA Stats schedule returned no data, so the ESPN fallback was used (12 games). `build_live_matchup_ratings` uses `data/nba_team_game_logs.csv` only when every scheduled team is one of the 30 known tricodes; otherwise it falls back to player box scores, where every 2026-27 row has a null ORB (1,275 of 1,275 rows). `data/nba_team_game_logs.csv` contains all 30 teams, so the fallback can only have been triggered by an ESPN team code outside the 30. `ESPN_TEAM_ABBR_ALIASES` mapped GS, NO, NY, SA and UTAH but had no entry for WSH (Washington), which is the most likely unmapped code. The exact ESPN code was not directly observed (ESPN was not reachable from the analysis environment), so this is an inference from the code path.

### Decision and Implementation

1. In the `predict` command, immediately after the active slate date is resolved, drop every `nba_data.csv` row dated on or after that slate date before any downstream use (`generate_predictions`, `compute_regression_candidates`, `sanity_check_picks`, `build_live_pick_card`). The guard prints `CAUSAL GUARD: ignoring N history rows dated on/after slate ...` when it acts. `data/nba_data.csv` is not modified, and nothing changes when no same-day rows exist.
2. Add `"WSH": "WAS"` to `ESPN_TEAM_ABBR_ALIASES`.
3. Add `NBA_TEAM_TRICODES` (the 30 canonical tricodes) and make `_fetch_espn_games` raise `ValueError: ESPN schedule has unmapped team code(s) [...]` for any code outside it, so an unmapped code fails with a direct message instead of misrouting the defense build to the incomplete player-box path.
4. No change to gating, ranking, per-prop caps, confidence thresholds, or betslip construction.

### Evidence and Outcome

- Staged `data/nba_data.csv` (82,340 rows) contained 94 rows dated on/after 2026-10-27 across 4 games. For slate 2026-10-27 the guard removed exactly 40 rows from 3 games with zero of those GAME_IDs remaining in history; a simulation of the failure produced 40 duplicate keys without the guard and 0 with it.
- The 30-code list matches the distinct `TEAM_ABBREVIATION` values in `data/nba_team_game_logs.csv` exactly. A simulated ESPN-style code list maps entirely into the 30 with the new alias; without it `WSH` is the only unmapped code.
- `python -m py_compile nba_props.py` passes; the causal guard from item 1 is still present after the item 2-3 edits.
- Not verified: a full `PicksNow` run with these changes. Whether WSH was the specific failing code is unconfirmed until the next run succeeds or the new unmapped-code message names another one.
- Backup of the pre-guard file: `nba_props.py.bak_before_causal_guard` (repository root).

### Consequences

A late re-run on a slate day can no longer leak partial same-day box scores into features or duplicate target rows, and ESPN schedule problems now fail with an explicit message. Operationally, a re-run after tip-off still produces a card from late FanDuel lines, so lines for started games must be checked before betting. Tomorrow's morning refresh refetches the 2026-27 season and is expected to replace the partial rows once the games are final (not yet confirmed).

---

## DEC-063 — Preflight Full-Slate Spend and Never Report a Blocked Watcher as Complete

- **Date:** 2026-10-28
- **Status:** Implemented and verified
- **Decision owner:** Jake

### Context

At 17:59 the guarded `PicksNow` workflow failed with `no current fanduel player-prop lines returned`. That message was false as a diagnosis. `historical_lines.csv` already contained 1,626 October 28 FanDuel rows over 12 events, with snapshots at 03:13, 03:30, and 08:34; the 08:43 pipeline had successfully published Josh Hart AST UNDER 4.5. The active Odds API headers reported 452 credits used and 48 remaining, proving the configured key has 500 total credits rather than the expected 100,000-credit tier.

The same account condition broke the closing-capture promise. The noon watcher planned 24 event windows, reached the configured 100-credit reserve before its first October 28 paid request, captured zero windows, and nevertheless returned `status: complete`. Separately, two full-slate pulls at 03:13 and 03:30 were only 17 minutes apart, so the pipeline had no protection against accidental rapid repurchase.

### Decision

1. Before the first paid event request, calculate a conservative complete-slate ceiling as event count multiplied by requested market count. Require that ceiling plus the configured provider reserve. If the balance is insufficient, raise a dedicated `OddsApiBudgetError` with remaining credits, events, markets, request ceiling, reserve, and total required. Send zero paid event requests.
2. Retain the 100-credit live reserve. Do not solve insufficient balance by silently spending through it or by returning a partial event slate.
3. Reuse an exact-date, same-book snapshot captured within the prior 30 minutes. Log the row count and age explicitly. This prevents an immediate repeated `PicksNow` call from repurchasing the same board.
4. Do not silently call a many-hour-old morning board fresh. A current paid refresh failure still blocks `PicksNow`; `PredictCached` is the explicit operator choice for no-spend cached scoring.
5. Closing watchers must report `blocked_provider_reserve`, `blocked_daily_credit_cap`, or `incomplete_windows` and include planned, completed, blocked, and pending window counts plus the provider balance. `complete` is permitted only when no planned window remains pending.

### Evidence

- Real free endpoint preflight: 48 remaining; 12 events x 9 markets = 108-credit conservative ceiling; 100-credit reserve; 208 required; zero paid event requests sent.
- October 28 retained data: 1,626 FanDuel rows, 132 players, 12 events; latest snapshot 08:34 local.
- October 28 valid morning card remains present: Josh Hart AST UNDER 4.5, prediction 3.8, confidence 61.1.
- Focused spend/watcher/runtime tests: `23 passed, 1 subtest passed`.
- Full repository suite: `162 passed, 330 warnings, 1 subtest passed`.
- Protected production models were not changed.

### Consequences

The pipeline can no longer burn another full slate during a short accidental rerun, begin a paid slate it cannot finish, hide reserve exhaustion behind a generic empty-lines error, or claim that a zero-capture watcher completed successfully. The remaining blocker is external and measurable: the configured key needs the intended paid-tier balance before full morning plus T-60/T-5 collection can resume.

---

## DEC-064 — Gate FanDuel Picks on the Exact Offered Price; Fix Season Stage, Opponent, and Slip Probability Defects

- **Date:** 2026-11-01
- **Status:** Implemented and verified; first live card produced under the gate
- **Decision owner:** Jake

### Context

The operator reported picks on props that did not look bettable (e.g. an unfamiliar player's STL under) and asked why PTS is never picked. Investigation of the 2026-11-01 card (Darius Garland STL UNDER 1.5, Brandon Williams AST UNDER 3.5) found:

1. Both lines were real FanDuel contracts from the 06:38 Odds API snapshot, but `predict` dropped `over_price`/`under_price`/`commence_time` when building the line frame. Every pick was judged against the PrizePicks 2-pick break-even `(1/3)^0.5 = 57.74%` regardless of price. Garland UNDER was priced -240 (70.6% break-even); the model's 72% was treated as a 77%-confidence edge.
2. `_coerce_pick_side_probability` read `dir_prob` (stored as a percentage, e.g. 72.0) as a fraction and clipped it to 0.99, so every betslip leg was scored as near-certain.
3. `_infer_season_stage` labeled any slate after the last logged game as `play_in`/`playoffs`. During the regular season the last logged game is always yesterday, so every regular-season slate (including 2026-11-01) was ranked with postseason weights.
4. Prediction rows did not carry `opp`, so `opponent` was blank on every pick.
5. Betslips used PrizePicks Power Play multipliers and grading, which do not describe FanDuel parlays.
6. PTS is not blocked by a hard-coded rule: the evidence policy rejects it (real-line shrunk accuracy 50.2% on 6,950 rows, meta AUC 0.50). PRA/PR/PA/TRB are rejected for the same reason; only AST/STL/BLK UNDER are allowed.

### Decision and Implementation

1. `_attach_live_market_prices` matches each candidate to its exact FanDuel row on (raw-name normalized player, prop, line) and adds `over_price`, `under_price`, `side_price`, `side_decimal_odds`, `no_vig_side_prob`, `model_side_prob`, `ev_per_unit`, `commence_time`, `event_id`, `line_snapshot_ts`, `price_gate_pass`, `price_gate_reason`. In FanDuel mode `break_even_prob` is the vig-inclusive implied probability of the chosen side and `exceeds_ev_threshold` equals `price_gate_pass` (semantics change from the PrizePicks value; columns unchanged).
2. Publication requires a paired price on both sides, a known start time in the future, side price no worse than `live_max_favorite_american = -200`, and `ev_per_unit >= live_min_ev_per_unit = 0.03`. The filter is applied before the tier cascade because lower tiers bypass the accuracy gates. The -200 cap exists because at heavy juice a few points of calibration error flips the EV sign; it is a risk control, not evidence.
3. `_coerce_pick_side_probability` divides percentage inputs by 100.
4. Postseason context requires April–June and at least a two-day gap after the last logged game.
5. Prediction rows carry `opp`; picks now show the opponent.
6. FanDuel-mode betslips are cross-game parlays priced as the product of leg decimal odds; same-event legs and unpriced legs are rejected (FanDuel prices same-game parlays with its own correlation adjustment). `betslips_history.csv` gains trailing `Pick_1..4_Price` columns; slip types are `N-pick FanDuel Parlay` / `N-pick FanDuel UNDER Parlay`. `run_daily.py` step 1.6 grades priced slips as FanDuel parlays (push/DNP legs drop out); legacy unpriced rows keep Power Play grading.
7. PTS stays blocked for betting. `live_tracking_only_props = ['pts']` logs the model's lean, exact price and EV for every PTS line to `output/tracking_only_history.csv` (never picks_history or betslips) to build a forward record.

### Evidence

- 2026-11-01 live scheduled run (07:39 predict, which loaded the edited module): price gate passed 2/8 candidates before the name-normalization fix; card = Brandon Williams AST UNDER 3.5 at -114 (break-even 53.3%, model 62.6%, EV +0.175), opponent LAC, `season_stage=regular_season`. Garland/Reaves/Nembhard STL unders were blocked at -245/-260/-270; Pritchard AST UNDER 4.5 blocked at EV -0.131 (-174).
- The store's `player_norm` uses a different normalizer ("collin murray boyles") than picks ("collin murrayboyles"); matching now normalizes the raw `player` name. Isolated dry run after the fix (outputs redirected to scratch, spend disabled): 4/8 pass (Williams, Murray-Boyles -106, Oubre +104, Melton -102). The card is still one pick because `live_pick_max_per_prop = 1` and all four are AST.
- PTS tracking: 49 leans logged, model side probabilities cluster near 51%; 1 of 49 clears the gate.
- Tests: new `tests/test_fanduel_price_gate.py` (price gate, juice cap, unmatched/started/one-sided lines, punctuated names, season stage, tracking log) and FanDuel parlay tests in `tests/test_betslip_ev_selection.py`; the Power Play test is pinned to PrizePicks mode.
- Full suite: `174 passed, 1 failed, 330 warnings, 1 subtest passed`. The single failure, `test_failed_live_line_snapshot_blocks_pick_generation`, reads the real `historical_lines.csv`; the scheduled run's 07:32 FanDuel snapshot was under the 30-minute reuse window, so the test's simulated refresh failure was correctly suppressed. Unmodified re-run at 08:03, after the snapshot aged out: `tests/test_daily_runtime_contracts.py` `15 passed, 1 subtest passed`. This test depends on the environment and can fail within 30 minutes of any real pull.

### Consequences

Cards will be smaller and some days empty; that is the honest result when the model cannot beat FanDuel's price. STL/BLK unders, historically the highest hit rates, are mostly priced beyond -200 and will rarely publish; this is consistent with DEC-038's finding that STL is efficiently priced at the book. Open operator decisions: whether to raise `live_pick_max_per_prop`, and whether to rank the card by EV rather than signal score.

---

## DEC-065 — Remove the Card-Size and Per-Stat Caps

- **Date:** 2026-11-01
- **Status:** Implemented and verified; 2026-11-01 live card regenerated
- **Decision owner:** Jake

### Context

Under DEC-064 four AST unders cleared every accuracy gate and the FanDuel price gate, but `live_pick_max_per_prop = 1` published only one. The card was also capped at `live_pick_target = 5`, and the candidate pool at `max(25, live_pick_min * 10) = 50`. The operator directed: publish every confident pick, with no limit per stat category and no limit on the total.

### Decision and Implementation

1. `live_pick_target = None` and `live_pick_max_per_prop = None` mean unlimited; `live_pick_min = 0` (no shortfall warning); `live_pick_candidate_pool = 500`.
2. With no target, `build_live_pick_card` publishes only the `core` tier (every accuracy gate passed). The `market_structured`/`structured` backfill tiers existed only to reach a fixed card size and are not used, so removing the cap does not admit ungated picks. The FanDuel price gate (DEC-064) still applies before the card is built.
3. Kept: one pick per player, and one row per (player, prop). These are correlation/duplication guards, not stat or size caps.
4. The console table prints every published pick (previously the top 10).
5. Integer targets and per-prop caps still work when configured; the existing card tests pin `live_pick_max_per_prop = 1` to keep covering that path.

### Evidence

- 2026-11-01, regenerated from the cached 07:32 FanDuel snapshot (no paid call): 4/8 candidates passed the price gate and all 4 published — Brandon Williams AST U3.5 -114 (EV +0.175), Collin Murray-Boyles AST U1.5 -106 (+0.073), Kelly Oubre Jr. AST U1.5 +104 (+0.277), De'Anthony Melton AST U2.5 -102 (+0.093). Previous card: Williams only.
- No betslips: the slip pool still requires confidence >= 70 and none of the four qualify.
- Tests: new unlimited-card tests (12 same-stat picks all publish; no ungated backfill; production defaults uncapped). Full suite `178 passed, 330 warnings, 1 subtest passed`.

### Consequences

Card size now tracks how many picks clear the gates; concentration in one stat or one game is possible (today: 4 AST, 2 from GSW-LAC). The protection against weak picks is the gates themselves, not volume limits. Prices on the card are from the snapshot time and must be checked at FanDuel before betting.

---

## DEC-066 — Closing-Line Value Tracking and Alternate-Line Contract Selection

- **Date:** 2026-11-01
- **Status:** CLV implemented and running daily; alternate-line selection implemented and verified but dormant (no alternate lines are fetched yet)
- **Decision owner:** Jake

### Context

The operator asked for (a) an honest, fast measure of whether the edge is real and (b) more high-value picks per day. Two facts shaped the design:

1. The paid FanDuel feed requests only main markets. Raw payloads (`data/line_archive/live_raw/`) carry exactly one outcome per player/market/side, and `historical_lines.csv` holds exactly one line per player/prop/snapshot. FanDuel alternate lines are separate `*_alternate` markets that cost extra credits per event and have never been requested.
2. Closing prices exist only where the T-60/T-5 watcher could spend: 2 slates (2026-10-27, 2026-10-31). The 500-credit key (DEC-063) blocked it on the other days.

### Decision and Implementation

**CLV (`scripts/clv_report.py`, run daily as `run_daily.py` step 1.7, diagnostic only):**
1. Bet snapshot = the latest FanDuel snapshot at or before the archived run that produced the card/predictions. Close = the latest FanDuel snapshot after that and before tip (master store, T-60/T-5 captures, alternate store).
2. Metrics per row: `ev_vs_close` (close no-vig side prob x bet decimal − 1; beats the close only if the market moved by more than the vig), `no_vig_move` (vig-free direction signal), `line_clv` when the line moved, `close_minutes_before_tip`, `close_source`, and graded result/profit at the bet price. Rows with no later snapshot are labeled and excluded from the summary.
3. Two populations: published picks and every model lean (the favored side on every scored line, from archived predictions), so skill can be measured on hundreds of rows per day rather than a handful of bets.

**Alternate lines (nba_props.py):**
1. `_line_raw_over_probability` / `_line_over_probability` price any line from the row's stored distribution (quantiles for continuous stats, ordinal tails `P(X>=k)` or Poisson rate for STL/BLK/TOV) through the same isotonic calibrator. Ordinal tail columns (`stl_ge_1..3`, `blk_ge_1..3`) are now persisted in predictions.
2. `_select_best_contract_lines` moves each candidate — whose direction already cleared the model gates on the main line — to the highest-EV FanDuel line in the same direction, subject to paired prices, the -200 favorite cap, `live_alt_line_max_distance = 2.0`, and `live_alt_line_raw_prob_band = (0.30, 0.70)`. Adds `main_line` and `line_choice`. The price gate then evaluates the chosen contract. No new player/prop candidates are created from alternate lines.
3. Fetching `*_alternate` markets is behind `live_fetch_alternate_markets = False` (default off; costs credits). When enabled, alternate rows are written to `data/fanduel_alternate_lines.csv`, never `historical_lines.csv`, because `_normalize_projection_type` maps unknown types to `standard` and the master store feeds training and the main-line view.

### Evidence

- `_line_over_probability` at the main line reproduces production `{stat}_p_over_cal` for AST/TRB/PTS/PRA/PR/PA on the 2026-11-01 predictions (max abs diff 5e-5, i.e. rounding).
- **Calibration limit found:** the AST isotonic calibrator maps every raw probability from 0.012 to 0.265 to 0.293 (fitted on main lines, where a low raw probability means the model disagrees with a balanced market). It is not valid far from the main line; hence the raw-probability band. Alternate-line probabilities outside the band are not used. There is no historical alternate-line data to recalibrate on.
- CLV baseline (2026-10-26 to 2026-11-01): published picks 15 rows, 4 with a real close (mean no-vig move +0.004, ev_vs_close −0.055); model leans 2,793 rows, 472 with an exact-line close (T-5, median 8.5 minutes before tip): no-vig move +0.0012, moved toward the lean 43.4% vs away 36.9%, ev_vs_close −0.057 (≈ the vig). This is noise-level evidence, not a demonstrated edge. 2,175 leans had no later snapshot.
- Dry run on the 2026-11-01 snapshot: `Contract lines: 275 main + 0 alternate`; card unchanged (4 picks).
- Tests: `tests/test_clv_report.py` (6), `tests/test_alternate_lines.py` (8). Full suite `192 passed, 330 warnings, 1 subtest passed`.

### Consequences / Open Items

- CLV becomes meaningful only with daily closing captures, which require the paid-tier Odds API key (DEC-063 blocker).
- Alternate lines produce no picks until `live_fetch_alternate_markets` is enabled. Before enabling: run a one-event canary to confirm FanDuel's alternate format. If FanDuel alternates are Over-only ladders, they cannot serve the current UNDER-only allowed props (and fail the paired-price rule), so the expected volume gain could be near zero.

---

## DEC-067 — Zero-Pick Reason Logging; Read-Only Audit of Absence Flags, Layer Joins, and the PBP Delay

- **Date:** 2026-11-03
- **Status:** Logging implemented and verified. Absence-flag attribution change NOT made (operator decision pending). PBP delay fix NOT made (awaiting approval).
- **Decision owner:** Jake

### Context

A manual `run_daily.py` run at 23:06 CT on 2026-11-03 produced no live picks. Step 2 logged only `Odds API returned no NBA events for today's slate (2026-11-03)`. The provider call itself succeeded (HTTP 200, remaining-credit header present) and `_coerce_slate_date` converts commence times to America/Chicago, so there is no UTC/local date mismatch. The message prints only when the provider lists events but none is dated on the slate date. The log did not say how many events were listed, their dates, the local time, or the credit balance, so a late run, an empty schedule, and a credit problem looked the same (DEC-063 recorded credit exhaustion previously misreported as missing data).

### Decision and Implementation (logging only; no pick-affecting change)

1. `nba_props._describe_empty_slate()` builds the message: events listed, their local date range, local clock time, provider credits remaining, and the possible reasons. `fetch_vegas_lines` prints it in place of the bare line. Return values and control flow are unchanged.
2. `run_daily.step2_prizepicks` logs a `Zero-pick reason` warning before raising the existing `no current <book> player-prop lines returned` error (exception text unchanged).
3. The Step 8 `Top 10 Projected Scorers Today` header states when the live-line snapshot failed and the list is not filtered to today's slate.
4. `tests/test_zero_pick_reason_logging.py` (3 tests) covers the helper and the `fetch_vegas_lines` path with a mocked provider, including that only the single free events lookup is made.
5. Backups: `nba_props.py.bak_before_zero_pick_reason`, `run_daily.py.bak_before_zero_pick_reason`.

### Audit Findings (read-only; nothing below was changed)

1. **Absence flags reach live predictions.** `usage_injury_model.detect_recent_absences` flags any rostered player with 2 or fewer of the last 5 team games (no minutes/role gate). Its roster comes from the cached player profiles, which are the DEC-058 restored 2025-26 profiles, so traded players carry last season's team; the `latest_played_team` filter removes only players who have already played for another team. Flags flow to `detected_absences.csv`, then Step 8 `absence_recent_gp`, then `nba_props._adaptive_fp_blend_weights`, which lowers the first-principles blend weight by 0.10 when `absence_recent_gp <= 2` (PTS/TRB/AST blends; AST is a live-allowed prop), and also suppresses luck columns. Changing absence attribution is therefore a pick-affecting change and interacts with the DEC-061 frozen holdout that began 2026-10-28.
2. **Join keys.** Layer 2 and the roster override join on `player_id`; Layers 3/4/5 join on `_normalize_player_name` (diacritic-safe). The Layer 3 count (508) mirrors the restored 2025-26 profiles. `apply_first_principles_adjustments` joins on `_normalize_name`; unmatched players skip the blend silently (`fp_applied=False`).
3. **Step 8 projects every player ever in `nba_data.csv`.** `raw_l20_*` is built from `base.groupby("player")` with no season filter, so 859 players are projected against 448 active in 2026-27, and each L20 window spans seasons for players with fewer than 20 current-season games.
4. **PBP delay.** `pbp_fetcher.fetch_game` skips games already on disk, but `fetch_season` still sleeps `REQUEST_DELAY = 1.1` s after every game including skipped ones: 55 games is about 60.5 s of the 205 s run (observed 61 s).
5. **CLV definitions** match DEC-066: `beat_close` requires the market to move by more than the vig, so `ev_vs_close` near `-vig` is expected when there is no information beyond the market.

### Consequences / Open Items

- **Operator decision:** re-key absence flags to the current-season roster and add a rotation gate now, or after the DEC-061 holdout window. It changes AST blend weights for affected players; a before/after diff of flagged players and the generated card is required.
- Optional, no pick impact: skip the post-request sleep when `fetch_game` was a cache hit (about 60 s per run).
- Not verified: whether the Odds API events endpoint lists games that have already finished; the new message reports observed facts and states the reasons only as possibilities.
- Verification run in the cloud workspace: new tests pass; 47 related tests passed in a temp copy. The full suite and the 56-artifact production-hash check were not run there and must be run on the operator machine.

---

## DEC-068 — Live Fetch Skips In-Play Events; Low-Credit Alarm; Step 2 Fetch Output Reaches the Run Log

- **Date:** 2026-11-04
- **Status:** Implemented and verified on the operator machine. No pick-scoring, model, or schema change.
- **Decision owner:** Jake

### Context

The 2026-11-04 07:30 CT run published no card: the Odds API key had 109 credits, below the 126 + 100 reserve needed for a 14-event slate (the budget guard worked; nothing was spent). The operator rotated `ODDS_API_KEY` (User env var) to a new 20,000-credit key. A 23:07 CT manual rerun then paid 18 credits and stored **135 in-play rows** (3 events that tipped at 20:48-21:41 CT) in `historical_lines.csv` as ordinary `odds_api_fanduel` rows, because the `/events` endpoint lists live games and `fetch_vegas_lines` filtered only by local slate date. Also, since Step 2 calls `fetch_vegas_lines` in-process, every message it prints (credits, events fetched, DEC-067 empty-slate reason) reached the console only; `grep` found none in `logs/run_202611*.log`.

Downstream impact of the in-play rows was contained: `_select_preferred_line_rows` and `scripts/clv_report.py` already drop post-tip snapshots, and the price gate blocked all candidates ("game started"). Unguarded raw readers remain (`scripts/market_edge_map.py`, `evaluate_clv_walk_forward.py`, `build_clv_features.py`, `player_pool_guard.py`). The 135 existing rows were left in place (no ad hoc edit of the data-contract file); they are identifiable by `snapshot_ts > commence_time`.

### Decision and Implementation

1. `nba_props._split_pregame_events(events, now_utc)` splits events by `commence_time <= now` (started) vs later/unknown (pregame; unknown kept, matching `_select_preferred_line_rows`). `fetch_vegas_lines` applies it before the budget check and before any paid request, logs how many events were skipped, and returns empty with a "none spent" message when every slate event has started. The clock comes from `nba_props._utc_now()` so tests are time-independent.
2. New `CONFIG['odds_api_low_credit_warning'] = 750`: when provider credits are below it, `fetch_vegas_lines` prints a `!!! ODDS API CREDITS LOW` line.
3. `run_daily._call_with_logged_stdout` / `_StdoutToLog` route Step 2's `fetch_vegas_lines` output to the run logger line by line; lines containing `!!!` log at WARNING.
4. `generate_predictions` logs start and elapsed time for point-in-time defense, live availability/minutes, and the per-player scoring loop (the 2m13s silent gap observed at 23:15-23:17).
5. Tests: new `tests/test_pregame_event_filter.py` (8). `tests/test_fanduel_live_lines.py` pins `_utc_now` in two tests whose 23:00Z tip-offs would otherwise fail after 17:00 CT. `tests/test_daily_runtime_contracts.py::test_failed_live_line_snapshot_blocks_pick_generation` now stubs `_recent_bookmaker_snapshot`; it failed whenever a real capture was under 30 minutes old.

### Verification

- `pytest -q tests`: 205 passed, 1 subtest passed (this also closes the DEC-067 operator-machine verification item).
- `scripts/project_operator.ps1 -Action Preflight`: syntax/import smoke clean; 56 protected production artifacts present and hashed.
- Live check (free `/events` call, 19,982 credits): at 23:29 CT, 3 events listed for 2026-11-04, all 3 classified started, 0 pregame; a rerun would spend nothing. This also confirms the DEC-067 open item: `/events` lists in-play games.

### Findings Not Acted On

- Layer 1 PBP columns (`possessions`, `raw_ppp`, `garbage_time_flag`, `competitive_poss`) cover 37/702 prediction rows because only `training_dataset_enriched.csv` (ends 2025-26) carries them. Only the legacy `models.pkl` uses them; none of the 36 production advanced models do. No change.
- Layers 3/4 MIN_GP (10/15) leave usage on restored 2025-26 profiles and luck neutral early in the season (DEC-058/059, by design).
- CLV (2026-11-04): model leans `ev_vs_close` -0.0586 (SE 0.0016, n=857), picks -0.0458 (n=11): consistent with no information beyond the closing market under the DEC-066 definitions.

## DEC-069 — Grade Every Final Candidate, Including Those Blocked by the Last Gates

- **Date:** 2026-11-05
- **Status:** Implemented and verified. Tracking only; no pick-scoring, card, betslip, model, or existing-schema change.
- **Decision owner:** Jake

### Context

Jake wants a forward record of the model's final candidates even when the last gates (FanDuel price gate, accuracy gates, card build) keep them off the card, to judge whether the model and the gates are doing well. `pre_card_candidates` (post injury/sanity check, with exact FanDuel prices and gate flags) was written only to `output/candidates_latest.csv`, which every run overwrites, and was never archived or graded. `tracking_only_history.csv` covers only PTS leans; `clv_model_leans.csv` covers all model leans, not the final shortlist.

### Decision and Implementation

1. **New file `output/candidates_history.csv`** (new artifact; no existing file changed). `nba_props._log_candidate_history()` runs in `predict` after the final card is built. One row per candidate with model fields, exact side price/EV, gate results, `published`, and `gate_bucket` = `published` | `blocked_price` | `blocked_other`.
2. **Integrity rules:** candidates whose game had tipped at run time are skipped (a late rerun cannot replace the pre-game record with in-play prices); rows upsert on (game_date, player, prop, direction, line) so the latest pre-tip run wins; `published` is sticky once True; rows absent from a later run are kept; graded rows are never modified. Failure prints a warning and never affects the card.
3. **New Step 1.8 `run_daily.step1_8_grade_candidates`** (after Step 1.7): grades past rows with Step 1.5 rules (WIN/LOSS/PUSH; player absent = DNP; no box scores and >3 days old = DNP), records `profit_units` at the recorded FanDuel side price (unpriced rows count toward hit rate only), and logs the last-14-days and all-time record per bucket. Never counts as a pipeline error.
4. Tests: `tests/test_candidate_history.py` (9).

### Verification

- `pytest -q tests`: 214 passed, 1 subtest passed. Preflight: 56 protected artifacts hashed, smoke clean.
- End-to-end in a temp directory with the real 2026-11-04 `candidates_latest.csv` (clock set pre-tip) and real `nba_data.csv`: 4 rows logged and graded correctly (Edgecombe STL U1.5 actual 1 WIN; Cunningham AST U8.5 actual 11 LOSS; Robinson AST U2.5 actual 2 WIN; Thompson AST U4.5 actual 3 WIN). Not written to production because those candidates came from a post-tip run.

### Consequences / Open Items

- No backfill is possible: past candidate sets were never archived. The record starts with the next pre-tip run.
- `tracking_only_history.csv` still replaces a whole slate on every run, so a post-tip rerun overwrites the morning PTS leans (it did on 2026-11-04 at 23:17). Same pre-tip rule could be applied there; not changed.
- Interpretation: compare `blocked_price` hit rate and ROI against `published`. Sample sizes will be small for weeks; do not loosen gates without a time-respecting before/after comparison.

## DEC-070 — Data Is Never Deleted: History Files Are Append-Only; Lost Picks and Slips Restored

- **Date:** 2026-11-05
- **Status:** Implemented, recovery applied, verified. Standing rule (also in CLAUDE.md Non-Negotiable Rules).
- **Decision owner:** Jake

### Context

Jake: "All data should be saved and it should be left up to me what's deleted." An audit of every write in the daily pipeline found code that deleted saved data:

1. `save_picks` replaced every row for the slate in `output/picks_history.csv` with only the current run's picks (`existing[existing['game_date'] != slate_date]`). Comparing all timestamped archive run copies with the history found **70 published picks across 19 slates** erased by later same-day reruns, including 2026-10-31 and 2026-11-01.
2. `log_betslips` did the same to `output/betslips_history.csv`; a zero-slip rerun wiped the whole slate (04-11, 04-14, 05-07, 05-11 had no slips left), and `reindex(columns=col_order)` dropped unlisted columns. **116 slips** recovered.
3. `_log_tracking_only_props` replaced the slate in `tracking_only_history.csv`.
4. `_log_candidate_history` (DEC-069) replaced rows on key match.
5. `nba_scraper.incremental_update` deleted all current-season rows of `data/nba_data.csv` before adding the API pull (a partial response would shrink the file); `--full-refresh` overwrote other seasons.
6. `nba_props.py update|train|full` wrote a hardcoded 2023-24..2025-26 fetch over `nba_data.csv`, erasing 2026-27.
7. `scripts/fetch_nba_team_game_logs.py` deleted the refreshed season's rows before merging.
8. `_write_historical_lines` re-sanitized the whole store with `drop_unknown=True` (0 rows dropped to date, but could drop stored rows).
9. `live_injury_status.csv` and `detected_absences.csv` were overwritten with no history.

Unaffected: timestamped archive copies (never rewritten), `*_latest` working files, derived reports, protected model artifacts, `historical_lines.csv` append/dedup on a key including `snapshot_ts`.

### Decision and Implementation

- New `nba_props._append_only_history(path, rows, key)`: existing rows and columns are never deleted or modified; rows are added only when their key is new; new columns are added blank for old rows; raises if the result would shrink. Keys: `PICKS_HISTORY_KEY` (game_date, player, prop, direction, line, normalized) and `_betslip_history_key` (date, slip_type, order-independent legs).
- Items 1-4 use it. New `published_at` column on picks/betslips history. Zero-slip runs leave betslips_history untouched. Tracking-only leans for games already started are not recorded. Candidate history keeps the first pre-tip record; the only in-place change allowed is promoting an ungraded row to `published=True`.
- Items 5-6: `nba_scraper.merge_game_logs_preserving` keeps every stored row and replaces only the same PLAYER_ID+GAME_ID (final box score superseding an in-progress one); empty responses change nothing; atomic write. `_save_nba_data_preserving` routes the `update/train/full` commands through it.
- Item 7: merge keeps all stored team-games; GAME_ID normalized to int on both sides (CSV loses leading zeros) so the keep-last dedup matches; shrink guard. `--replace-all` remains an explicit operator choice.
- Item 8: whole-store sanitize uses `drop_unknown=False`; `_backup_before_shrink` copies the full prior file to `data/backups/` with a `!!!` warning before any rewrite that would hold fewer rows.
- Item 9: `run_daily._save_timestamped_snapshot` keeps `data/snapshots/<stem>/<stem>_<stamp>.csv` each run.
- Recovery: `scripts/recover_lost_history_rows.py` (dry run by default; additive; idempotent) appended 70 picks (1,556 -> 1,626) and 116 slips (160 -> 276) from the archive run copies, tagged `history_source=archive_recovered`, `published_at` = run-copy time, `recovered_from` = source file. Step 1.5/1.6 will grade them on the next run, which will change reported historical win rates; filter on `history_source` to exclude them.

### Verification

- Backup before any change: `output/backups/before_append_only_20261105_041553/` (all affected data and code files).
- Original 1,556 picks and 160 slips are byte-identical after recovery (rows and original columns compared to the backup); re-running recovery finds nothing.
- Real `nba_data.csv` (83,794 rows): re-merging the current season, a 100-row partial response, and an empty response all leave 83,794 rows (old code: a partial response would have cut it to ~81,165).
- `pytest -q tests`: 223 passed (new `tests/test_append_only_history.py`, 8). Preflight clean, 56 protected artifacts hashed.

### Operator Decisions

- Whether restored rows should count in headline performance figures (they are flagged).
- Whether to remove any data; the pipeline never will.

## DEC-071 — Loosen Two Gates on Forward Evidence: Volatility CV 0.70 -> 1.00, Min EV +0.03 -> +0.01

- **Date:** 2026-11-05
- **Status:** Implemented (config values; one-line rollback each). PICK-AFFECTING: more candidates reach and pass the price gate.
- **Decision owner:** Jake (requested loosening because cards were empty)

### Context

Jake reported zero picks getting through. Log review showed most zero-pick slates (10-26, 10-29, 10-30 morning, 11-03, 11-04) had **no lines at all** (Odds API credit exhaustion, fixed by the DEC-068 key rotation), so no gate ran. On slates with lines the card was 1-3 picks (11-02: 12-13). The largest cut was `sanity_check_picks` HIGH VOLATILITY (6-20 removals/slate), then the price gate (~50% pass).

### Evidence (forward, pre-game logged, graded against nba_data.csv)

- Sanity-check removals parsed from all `logs/run_2026*.log` and graded: volatility removals hit 71.4% (n=440, Apr-May 2026) and 65.7% (n=70, 2026-27; ROI +6.1% at recorded FanDuel prices, n=68) vs published picks 69.3% (n=215) and 63.3% (n=30). By CV band (all): 0.70-0.85 69.5% (164), 0.85-1.00 67.1% (140), 1.00-1.25 70.5% (112), >1.25 76.4% (106); this season 1.00-1.25 was weakest (58.8%, n=17). No band shows the filter removing losers.
- `output/clv_model_leans.csv` (3,614 graded 2026-27 leans, 8 slates, exact FanDuel prices, -200 cap applied): EV (0,0.01] ROI +5.2% (215), (0.01,0.02] +8.5% (158), (0.02,0.03] +5.7% (145), (0.03,0.05] -3.3% (123), (0.05,0.08] +2.2% (117), >0.08 +3.8% (361). The model's EV does not rank above zero, so +0.03 mainly discards volume. Prices worse than -200 with EV>=0 lost 13.1% (n=18): cap kept.
- Caveats: 8 slates; SEs 5-9% per bucket; DEC-068 found no edge vs closing lines. Directional evidence, not proof.

### Decision

- `CONFIG['sanity_volatility_cv_limit'] = 1.00` (was hard-coded 0.70); relaxed micro-profiles are never stricter than the base. Spike rule, injury/day-to-day removals and all other sanity rules unchanged.
- `CONFIG['live_min_ev_per_unit'] = 0.01` (was 0.03). `live_max_favorite_american` stays -200. Accuracy gates unchanged.
- DEC-061 shadow holdout is unaffected (it compares model predictions on identical rows, not card gates).
- Tests: `test_fanduel_price_gate.py` thin-EV test moved to the new boundary (52% at -110 blocked, 53% passes) plus a config-driven bar test. Full suite 224 passed.

### Review / Rollback

- Rollback: set the two config values back to 0.70 / 0.03.
- Review after ~3 weeks using `candidates_history.csv` (DEC-069) and graded picks: compare picks that pass only because of the new limits (CV 0.70-1.00, EV 0.01-0.03) against the rest at real prices. Tighten again if they underperform.


## DEC-072 — Scheduler Wrapper No Longer Aborts on Python stderr Output

- **Date:** 2026-11-07
- **Status:** Implemented (`scripts/project_operator.ps1`, `Invoke-PythonStep` only). No pick, model, gate, schema or data change.
- **Decision owner:** Jake (requested the wrapper fix after the read-only project audit)

### Context

`logs/scheduled_dailysafe.log` recorded `FAILED DailySafe` on all 10 days from 2026-10-28 to 2026-11-06, with no COMPLETE line, even on days whose `run_*.log` ended `All steps completed successfully` (e.g. 11-06). Six failures carried a Python `[INFO]` line as the error message, such as `Fitting Ridge Regression over 2642 observations...`. That line is written to stderr by `ppp_engine.py` (`logging.basicConfig`, default stream) when the last DailySafe step (`scripts/capture_candidate_future_holdout.py`, which imports `nba_props`) runs. `run_scheduled_action.ps1` sets `$ErrorActionPreference='Stop'` and merges streams with `*>&1 | Tee-Object`; under Windows PowerShell 5.1 (the interpreter used by `install_project_automation.ps1`) native stderr then becomes a terminating error.

### Evidence

Reproduced in an isolated scratch test (outside the project, 5.1.26100): a native command that wrote one harmless stderr line and exited 0 aborted the step before the next stdout line and logged `FAILED :: 12:00:00 [INFO] harmless` through the same `*>&1 | Tee-Object` wrapper pattern. The patched function was then extracted from the real file by PowerShell AST and run under that wrapper with a `python.cmd` shim: stderr+exit 0 continues (output logged), plain stdout continues, stderr+exit 3 throws `... failed with exit code 3`, and `$ErrorActionPreference` is restored to `Stop` afterward. The file parses with zero parser errors.

### Decision

`Invoke-PythonStep` temporarily sets `$ErrorActionPreference='Continue'`, runs `& python @PythonArgs 2>&1`, converts ErrorRecords to plain strings, restores the preference in `finally`, and decides success from `$LASTEXITCODE` only. Real failures still throw with the exit code. `Invoke-ProtectedPythonStep` hash checks are unchanged.

### Consequences / Not done

- Expect DailySafe to log COMPLETE when the pipeline and holdout step succeed. If the holdout step then writes `data/candidate_future_holdout_predictions.csv` for the first time, the DEC-061 holdout starts accumulating from that date, not from 2026-10-28; check the capture QA item in TASKS.md. The earlier "FAILED" lines did not prove the capture itself failed (cause unproven).
- Not verified on the real scheduler: the next scheduled run is the confirmation.
- `scripts/retrain_updated_models.ps1` has the same latent pattern (`$ErrorActionPreference='Stop'` plus bare `& python`). It is a manual script and was left unchanged.
- Unrelated failures seen in the same logs remain: 2026-10-29 Application Control block of `python.exe`; 2026-11-06 `WinError 5` on a `.pyc` rename and an availability-join traceback (probably overlapping manual runs); real Odds API credit shortages.

### Rollback

`git checkout -- scripts/project_operator.ps1` (or restore the three-line `& python @PythonArgs` / `$LASTEXITCODE` form).


## DEC-073 — Loosen Final-Card Meta Floor and FanDuel Volatility Ceiling

- **Date:** 2026-11-10
- **Status:** Implemented (config values in `nba_props.CONFIG`); unit tests pass. Same-day rerun check passed (see Verification). PICK-AFFECTING: more AST (and potentially TRB/STL/BLK/TOV) candidates pass the final accuracy gates.
- **Decision owner:** Jake (asked to loosen the "blocked by other gates" filters after seeing their record)

### Context

Step 1.8 (DEC-069) reported candidates blocked by other gates at 8-1 and price-blocked at 25-16. As of 2026-11-10 `output/candidates_history.csv` showed published 26-14 (+7.0u), blocked_price 39-18, blocked_other 11-2 (+7.35u). Every blocked_other row was an AST UNDER. Four causes: meta_prob below the AST floor of 46.0 (10 rows), volatility above the AST ceiling of 2.92 (4 rows, one also meta), volume_fill/emergency_fill excluded from the card (5 rows).

### Evidence

- Candidates (4 graded slates, 2026-11-06..11-09): meta 42-46 went 3-1 (plus 1 DNP), meta < 42 went 1-1, volatility 2.92-3.30 went 4-0 (one overlapping with meta), and fills went 4-0.
- `picks_history.csv` (published picks, mostly Apr-May 2026): AST meta (0,35] 2/6, (35,40] 2/5, (46,50] 3/7, (50,55] 20/29, >60 46/63. All props meta (40,46] 4/5. AST hit rate does not fall as vol/ceiling rises to 1.0 (0.9-1.0: 16/23). Consistent with DEC-071 finding that volatility removals did not remove losers.
- Fill picks published historically: volume_fill 1-2, emergency_fill 0-1.
- Price-gate blocks: favorite cap (beyond -200) 17-10, -2.80u; EV floor 22-8, +5.43u, but nearly all of those had model EV between -0.15 and -0.01 at -130 to -198. Loosening that would mean betting negative model EV.
- Caveats: n=13 graded blocked_other, 4 slates, single lane (AST UNDER), which also hit 22-11 when published. This is directional evidence, not proof.

### Decision

- `live_pick_meta_floor_min` 45.0 -> 42.0 and `live_pick_meta_floor_slack` 10.0 -> 14.0. Resulting floors are AST 42, TRB 45, TOV 45, STL/BLK 42, with PTS/PA/PR/PRA unchanged (meta gate off).
- `live_fanduel_volatility_ceiling_mult` 1.25 -> 1.50. Resulting ceilings are AST 3.51, TRB 4.36, STL/BLK 1.50, TOV 1.88.
- Unchanged: price gate (-200 cap, EV >= +0.01), fill-pick exclusion, hit-rate floor, sanity checks, models. `meta_floor` is not used in signal scoring. `volatility_ceiling` feeds only the postseason role score, so regular-season ranking is unchanged.
- Replay against logged candidates: 7 of 13 graded blocked_other rows would have published (6-1, +4.83u). The rows still blocked went 5-1 (+2.53u: 4 fills, 2 meta < 42).
- Verification: `python -m pytest -q tests`, 224 passed. No test pinned the old values.
- Same-day rerun: isolated rerun check PASSED 2026-11-10: `nba_props.py predict` run twice in a scratch copy, both exit 0 with no errors; run 1 published 2 extra picks under the new limits (Desmond Bane AST U2.5 meta 44.4, Isaiah Hartenstein AST U3.5 meta 44.8; card 5 -> 7); run 2 added 0 rows and left picks/candidates/betslips/historical_lines byte-identical; every original row in all five history files unchanged (only the 2 candidate rows flipped published False->True, the DEC-070-allowed promotion); real project files' hashes unchanged.

### Review / Rollback

- Rollback: restore 45.0 / 10.0 / 1.25 in `nba_props.CONFIG`.
- Review around 2026-11-30: compare published picks with meta 42-46 or volatility in (1.25x, 1.50x] of the reference against the other picks at recorded prices. Revisit fills only after 30+ graded fill candidates.


## DEC-074 — Step 1.5 Prints the Last Slate's Picks With W/L on Every Run

- **Date:** 2026-11-10
- **Status:** Implemented (`run_daily.py`). Display only: no pick, grading, data, or schema change.
- **Decision owner:** Jake (asked to see yesterday's picks with a W/L beside each in the Step 1.5 output)

### Context

Step 1.5 printed a W/L table only for picks graded in that run. A same-day rerun hit `All picks already graded` and returned early, so the previous slate's results were never shown again.

### Decision

- New helper `_log_last_slate_results(hist)` logs every pick for the latest `game_date` before today (so off days show the last played slate) with `W`/`L`/`P`/`DNP`, or `?` if ungraded, and that slate's record. It is called on the early-return path and at the end of Step 1.5.
- It reads the in-memory frame only, writes nothing, and wraps itself in try/except. A failure logs one warning and never affects grading or later steps. Output goes through `logger`, so it reaches `logs/run_*.log`.

### Verification

- The function was extracted by AST and run against the real `output/picks_history.csv` with a console-only logger, without importing `run_daily` (which would append to the real log). Output: 2026-11-09, 7 picks, 3W/4L. File md5 unchanged (`9a7d1b2e...`). Empty input printed a message; malformed input logged a warning with no crash. `pytest -q tests`: 224 passed.
- Ordering note: the DEC-073 same-day rerun check (two isolated `nba_props.py predict` runs) was done **before** this change and covered Step 9 only. This change was not exercised by a full `run_daily.py` run. The first real confirmation is the next scheduled run's log.

### Rollback

Remove the two `_log_last_slate_results(hist)` calls (or the helper) in `step1_5_grade_picks`.



## DEC-075 ? Repair timestamp and validation integrity before further accuracy tuning

- **Date:** 2026-11-10
- **Owner/request:** Jake requested a comprehensive project review and improvements to pick-generation accuracy.
- **Status:** Implemented and verified integrity repairs; isolated retraining/forward acceptance remains open.

### Evidence and decision

Mixed legacy/ISO timestamp columns exposed pandas format-inference failures: 584,364 of 742,967 historical line rows have valid timestamps rejected by the default parser. Explicit `format="mixed"` is now used in canonical live/training timestamp paths and the naive-baseline audit. Full-data selector comparison: 46,824 contracts before, 65,835 after; 17,159 matching contracts change line. This is a pick/data-selection-affecting correctness fix, not a measured accuracy improvement. Legacy unknown-time/timezone provenance remains unresolved, and rows are preserved.

Sorted row-based TimeSeriesSplit still shared a date at all five boundaries on master box scores. New `chronological_validation.date_grouped_splits` separates whole dates in nine advanced CV paths and the minutes CV path, failing on missing/unsorted/insufficient dates. Frozen production models and their existing OOF artifacts were not changed. New candidates/calibrators require isolated reevaluation under this procedure.

The frozen future holdout now requires known pregame prediction-capture times before and after scoring, uses append-only writes instead of replacing matching snapshots, evaluates only the first pregame capture per exact contract, and resamples date clusters with row-weighted ratios matching reported accuracy. Current 1,777 stored rows have zero invalid-time or duplicate-contract exclusions. Existing saved summaries are preserved until the next grading run.

### Verification and constraints

- `python -m pytest -q tests --disable-warnings`: 238 passed, 330 warnings, 1 subtest passed (14 regression tests added).
- Read-only audit: 108 active Python files parsed, no syntax errors; box-score and physical line keys have zero duplicates. Evidence includes source SHA-256 hashes.
- Five canonical CSV hashes match the first/final audit. No raw history rows were removed; existing uncommitted work was preserved.
- No paid calls, model retrain/promotion, live card regeneration, threshold change, scheduler change, or bankroll change.
- Full coverage, current-priced performance, limitations, and ranked follow-up: `output/PROJECT_AUDIT_2026-11-10.md`.

### Next step / rollback

Train and audit an isolated candidate with corrected inputs, date-grouped folds and its bound calibrator; compare on untouched common contracts versus production, always-UNDER, and paired-price market probabilities. Continue the frozen future holdout. Do not relabel old backtests as verified under the new procedure. Rollback is limited to these source edits; never revert data, artifacts, or earlier user changes to achieve rollback.


## 2026-11-10 ? Isolated-refresh path correction (DEC-076)

User execution exposed hard-coded CLI defaults in feature_pipeline.py that wrote original-project data/model_data_advanced.csv and feature_cols_advanced.json despite execution from the isolated copy. Earlier isolation assurances were incorrect. Corrected defaults to the script directory and made run_training_maintenance pass explicit input/output/features paths. Applied surgical source fixes to both the original and Fresh_Start_NBA_retrain_20261110_095208 copies. Two new regression tests pass; copied scripts parse. No refresh/retrain was launched by the agent, and original generated files were not reverted. Rerun refresh in the corrected isolated copy before training.


## 2026-11-12 ? Unify live publication gates and reject invalid inputs

User requested a deep gate correctness/analytics review using validate-data. Demonstrated P1 defects repaired under that review. `all` previously published through an independent branch omitting the exact-price/source policy. It now calls canonical `predict` after its existing update/train steps with fresh features. Pricing rejects infinite/out-of-range probabilities, infinite American odds and unknown/future snapshots. Cached prediction selection supplies the run timestamp. These changes enforce existing executable/pregame contracts; numerical thresholds and models were not retuned.

Verification `python -m pytest -q tests --disable-warnings` passed 245 tests and 10 subtests, 330 warnings. New regressions in `tests/test_live_gate_entry_points.py`. `python scripts/audit_live_gates.py` independently checked 149 outcomes with zero outcome disagreements and identified two actual-value conflicts. Histories and model input hashes were verified unchanged during audit calculation. Report `output/LIVE_GATE_REVIEW_2026-11-12.md` documents incomplete upstream rejection evidence, publication/first-candidate provenance differences, grading revision risk and proposed future gate evaluation. Existing dirty workspace changes were preserved.


## 2026-11-12 ? Preserve gate decisions and initialize prospective evaluation

User authorized the next steps following the deep gate review. Added `gate_decisions.py` and optional rejection collectors to primary generation and active sanity filtering. Existing callers retain their return contracts. Each successful live publication gets a separate atomic JSON receipt with a unique run ID, effective policy/configuration hash excluding credentials, active model hashes, scoring-source hash, exact market offers, injury report, first primary/sanity failures, final candidates and actual publication rows. A receipt cannot overwrite an existing run ID. Logging errors are visible and nonfatal to the saved card.

The experiment starts 2026-11-13 and uses the final post-sanity candidate population. Freeze metadata was finalized before its start with no captured run receipts. Incumbent, meta46 and volatility1.25 alternatives share pregame offers and retain exact prices. Comparators measure final candidate eligibility before one-player selection, while actual published picks are reported separately. Deduplicate first contracts across runs, exclude policy/model/scoring-source drift, and report date-bootstrap incremental units per offered contract. The chosen 30-date/300-contract review minimum supports a planned review and does not imply a power guarantee. Daily Step 1.8 refreshes evaluation.

Added immutable local grade-source conflict observations and a daily summary. Five conflicts across 1,733 checked history rows include one integer-line tie stored as LOSS. Original grading records remain evidence pending source/convention reconciliation. Removed the positive-floor no-op in sigma edge logic, keeping the existing effective floor, and made fallback-source exclusion explicit in diagnostics. Broader threshold tuning awaits future evidence. Full suite passed 252 tests and 10 subtests, with 330 warnings. Existing unrelated workspace changes were preserved.

Fingerprint scope follows the existing production release patterns plus live sigma/minutes/availability policy and legacy fallback models. A scheduled RefreshTrainingData run changed the auxiliary availability model during verification. Its policy disables live use, so its model bytes are included only when enabled_for_live is true. A regression verifies disabled refresh stability and enabled-model inclusion. Final frozen baseline matches current protected production files.


## 2026-11-12 ? Preserve every version of pipeline artifacts

User requires all picks and data retained across daily runs. Cumulative pick/slip/candidate/tracking stores already append logical keys, but derived datasets, daily paths, model refreshes and reports reuse filenames. Added `artifact_retention.py` using a project-scoped Python audit hook before file mutations. It retains old bytes before write/delete and retains staged replacement bytes plus the destination before rename. Compressed content is addressed by SHA-256, with per-path version records and checkpoint manifests. No expiry or deletion routine was added. Baseline retained 13,141 available files, approximately 1.3 GB compressed, and 12,569 unique objects passed decompression/hash verification.

Normal CLI modules install the hook. Daily/training child processes and both operator PowerShell launchers run scripts through the protected Python entry point. Watcher logs are retained before external redirection. File-preservation errors abort the individual write. Cross-process history append locking protects cooperating appenders. Same-day snapshot precision was raised to microseconds, and conflicting legacy archive names use unique destinations. Existing histories remain the downstream logical stores, while retained versions preserve corrected rows and prior run contents for inspection/restoration.

Verification `python -m pytest -q tests --disable-warnings` passed 256 tests and 10 subtests, 330 warnings. Added write/replace/append/delete/full-disk/concurrent-append regressions. Restored pick history matched its source bytes. Both edited PowerShell files parsed. Guide `output/DATA_RETENTION.md`. Protection applies to this workspace's instrumented processes and cannot reconstruct already-absent files. Sources/data external to protected paths require separate coverage. The archive is local and its disk use grows with changed versions. Prior experiment metadata was retained when updating its scoring-source baseline before the first eligible date.

## 2026-11-13 — DEC-077 — Carry season_games_played into prediction and pick rows

**Context:** While reviewing the 2026-11-12 slate (2-3), the agent found `season_games_played` blank on all 1,702 `picks_history.csv` rows. Jake ordered an immediate fix.

**Root cause:** `generate_predictions` built each pred row with `games_played` but not `season_games_played`. `create_features` computes it correctly (shifted cumcount per player-season), and the cached live feature frame had correct values (e.g. Grayson Allen 11, Trae Young 12 on 2026-11-13). `filter_best_picks` reads it from the pred row, so it was always None. `_apply_postseason_context` then used career games clipped to 20-82, overstating role stability for every player early in the season. That score is the first sort key in `_sort_betslip_candidates` all season; the live card uses it only in postseason. `_adaptive_fp_blend_weights` reads the feature frame, so projections were not affected.

**Decision:** Add `'season_games_played': row.get('season_games_played', np.nan)` to the pred row. Keep the fallback so the morning run cannot break, but print a `!!! WARNING` with the missing count. No schema change: the column already existed and was empty. No leakage: the value counts games strictly before the slate.

**Evidence:** Read-only replay of `_apply_postseason_context` on 2026-10-21..11-13 picks with true season counts from `nba_data.csv`: max `postseason_pick_score` shift 0.017; sort order changed on 2/16 slates (2026-11-06, 2026-11-13), never in the top 6. `picks_history.csv` hash unchanged by the replay. Tests: `tests/test_season_games_propagation.py` (3 new); `python -m pytest -q tests --disable-warnings` gave 259 passed, 10 subtests passed.

**Risk:** Betslip ordering can change slightly. In the playoffs the role score weighs more (0.24-0.26), so the fix matters more there. This change alters the scoring-source hash, which affects the gate experiment that started 2026-11-13. Historical rows stay blank; nothing was backfilled (DEC-070).

## 2026-11-13 — DEC-078 — Rest/back-to-back and rematch audit: no change

**Context:** In the 2026-11-12 slate review, WAS played a back-to-back (at IND 11/11, then DET 11/12), and DET/WAS replayed their 11/09 game. Jake asked to check whether the model accounts for rest days.

**Findings:**
- Already modeled: `create_features` builds `days_rest` (player's previous game, fillna 3, clip 0-7), `is_b2b`, `opp_days_rest`, `opp_is_b2b`, `rest_advantage`, the B2B interaction terms and `{stat}_on_b2b/_rested/_b2b_penalty` (shifted rolling). Production selection kept `ast_rested` (AST, PA, STL, TOV), `pts_rested` (PTS, PR) and `trb_rested` (TRB, STL). Raw `days_rest`/`is_b2b` did not survive selection for any prop.
- Live correctness: the 2026-11-13 cached feature frame computes target-row rest from the latest completed game (e.g. IND played 11/11 → `days_rest`=2 on 11/13). No leakage.
- League effect (`nba_data.csv`, rotation players mp_l10>=24, actual minus shifted L10): back-to-back AST −0.045 (n=6,054), MP −0.30, PTS −0.11; same-opponent-as-previous-game AST −0.106 (n=2,147). Both are tiny and lean toward the UNDER.
- Our graded picks (team schedule join, 1,563 rows): back-to-back 60.1% (n=271) vs 55.9% (n=1,292); rematch 62.3% (n=204) vs 55.8% (n=1,359). AST UNDER on back-to-backs 17-3 (85%) vs 63.6%.

**Decision:** No gate, feature or retrain. A rest penalty or rematch filter would cut a segment that has outperformed. The 11/12 Cade/Trae losses came from a double-overtime game (290 team-minutes), not rest; that risk is addressed by the separate same-game concentration idea, not by rest features.

**Open note:** `is_b2b` comes from the player's previous game, so a player who sat the first night of a back-to-back reads as rested. Recorded as optional in TASKS.

## 2026-11-13 — DEC-079 — Top-end calibration and teammate-out minutes audit: no change

**Context:** The 2026-11-12 slate review flagged (a) picks with `model_side_prob` 0.70–0.75 hitting 53% (n=15) and a model-vs-market gap >0.20 going 0-2, and (b) Grayson Allen's projected minutes staying at 27.7 with Brandon Miller out (actual 34). Jake asked for both to be addressed.

**Calibration findings (`candidates_history.csv`, 162 unique graded contracts, 2026-11-05..11-12, 8 dates):**
- `model_side_prob` average 0.626 vs actual 0.673, so the model is underconfident on average. Buckets: 0.55–0.60 hit 63.9% (61), 0.60–0.65 70.2% (57), 0.65–0.70 73.3% (15), 0.70–0.75 65.2% (23), >0.75 50% (4).
- Brier (lower is better): market no-vig 0.2363, model 0.2257; blends at model weight 0.25/0.5/0.75 scored 0.2321/0.2290/0.2269. Date-bootstrap 95% CI of model-minus-market Brier: [−0.0209, −0.0006]. A 0.70 cap scored 0.2242, a gain driven by 27 rows on 8 dates; not adopted.
- Model-vs-market gap >0.20: 2-2 (n=4). Too few to act on.

**Minutes findings:**
- `minutes_deviation_model.pkl` (71 features) has no teammate-availability inputs. `mp_expected` = `mp_l10` + deviation. It feeds only `_apply_postseason_context` and the minutes sanity check (blocks an UNDER when the line implies <65% of projected minutes). It does not feed stat predictions, so a minutes bump alone would not have changed the Allen pick; a higher `mp_expected` would only make the UNDER sanity check stricter.
- Teammate absences are handled in `apply_injury_adjustments` (PTS ×(1+boost), AST ×(1+0.9·boost), boost ≤0.12). Graded picks: injury-boosted UNDER 70.2% (n=57) vs 60.5% (n=953); boosted AST UNDER 67.9% (n=28), actual minus prediction −0.18 (not undersized).
- Side checks: `mp_location_diff` is computed at training time (`minutes_deviation.py:284`) but is 0 at prediction time (`nba_props.py:4193`); its importance is 0.0. Setting `opp_pace_l10` to 100 for every live row changes minutes predictions by 0.0. Live rows default `pace_factor` to 1.0 because they have no same-day pace, but no production model selects it (only BLK uses `opp_pace_pt`, attached point-in-time).

**Decision:** No recalibration, no minutes change, no injury-boost change. Each change would act on samples of 4–15 rows and cut segments that are outperforming. Revisit calibration at the ~2026-11-30 review with more dates. The `mp_location_diff` mismatch is recorded as an optional item for the next isolated minutes retrain.

## 2026-11-13 — DEC-080 — Re-derive the direction policy for FanDuel: UNDER-only confirmed

**Context:** Jake flagged that the card is 100% UNDER (no OVER pick since April; 194/194 candidates since 11/05 are UNDER), so one high-scoring or overtime night can sink the whole card. DEC-038 (decision item 2) said: never run the UNDER-only strategy against sportsbook lines without re-deriving the direction policy for that book first. DEC-060 (2026-10-26) made FanDuel the only live market without that step. This entry is that re-derivation. DEC-038 is kept as history; its "FanDuel favours OVER" finding came from 304 rows and the 30+ minute subset, and does not hold on the full priced sample below.

**Method (read-only, exploratory):** `data/historical_lines.csv`, bookmaker `fanduel`, both prices present, snapshot at or before commence time (timestamps parsed with `format='mixed'`; plain parsing silently dropped this season's rows), last pregame snapshot per contract, most price-balanced line per player/prop/date, joined to `data/nba_data.csv` actuals (mp>0, pushes dropped). 34,065 contracts.

**Baseline by stat (over hit vs mean no-vig over; ROI per unit at the offered price):**
- 2026-02-23..05-18: OVER hit is below no-vig for all 8 stats. Blind OVER ROI −6.9% (PTS) to −13.2% (STL); blind UNDER −1.1% (AST) to −5.8% (PA).
- 2026-10-26..11-12: OVER hit is below no-vig for all 8 stats. Blind OVER ROI −11.8% (AST) to −23.9% (BLK); blind UNDER −1.5% (TRB) to +9.3% (BLK).

**Model-selected OVERs:** `models/probability_oof_advanced.csv` `p_over_raw` joined at the exact same line (23,983 contracts, 79 dates, spring only). Betting OVER when p_over_raw minus no-vig over exceeded 0/0.03/0.05/0.08/0.10/0.15 returned −9.4%/−10.1%/−10.6%/−11.2%/−10.4%/−9.9% (n 9,819 down to 1,100). A higher claimed edge did not improve results. By stat at edge >0.05, every stat lost except STL (n=24, noise). The same raw rule on the UNDER side returned −5.0% to −3.3%; the live card's further gates are what has turned that into +8.3u on 77 priced picks since 11/01.

**Caveats:** The OOF predates DEC-075's date-grouped folds, and raw (uncalibrated) probabilities were used. Both caveats would tend to flatter the model, which still lost on OVERs.

**Decision:** Keep the card UNDER-only on FanDuel. Enabling OVERs would add negative-EV bets. The concentration risk Jake raised is real, but the remedy is exposure control (per-game cap, daily stake cap), recorded as proposed pick-affecting work for Jake to approve. Also proposed: a startup warning if the live book changes from the one this policy was derived on.

### DEC-080 addendum (2026-11-13) — per-game cap replay: not adopted

Replay on the published FanDuel-era cards (`picks_history.csv`, 2026-10-26..11-12, graded and priced, 77 picks / 10 dates), keeping the top picks per game in live-card sort order (`signal_score`, `estimated_hit_rate`, ...):
- Current card: 47-30 (61.0%), +8.31u.
- Cap 2 per game: 40-26 (60.6%), +6.15u. The 11 removed picks went 7-4 (+2.16u). Date-bootstrap 90% CI of the units change: [−6.62, +2.14]. Worst single day got worse (−1.77u → −2.37u); daily SD 1.87 → 1.52.
- Cap 1 per game: 31-16 (66.0%), +8.88u. The 30 removed picks went 16-14 (−0.57u).

The earlier "75% alone vs 52% when 3 share a game" comparison mixed game composition with pick order; once the cap keeps the best-ranked picks, the removed ones were not losers. Neither cap is supported on 10 dates. No change; rerun at the ~2026-11-30 review. A daily stake cap is a bankroll decision for Jake, not a model change.

## 2026-11-13 — DEC-081 — Fix CodeRabbit findings in the CLV report and gate audit scripts

**Context:** Jake ran `coderabbit review --committed --dir scripts` and `--dir tests` (a whole-repo review exceeds CodeRabbit's 100 MiB request limit because of `models/weekly_candidates/.../probability_oof_advanced.csv`). It returned 5 findings. Each was checked against the code before any change.

**Findings and actions:**
1. `scripts/clv_report.py` bet snapshot (major, valid): `latest_run_times` keeps only the last archived run per slate, so a morning pick was priced at a later same-day rerun's snapshot. Fix: `build_pick_rows` carries `published_at` (naive America/Chicago time, parsed with an explicit format) as an internal `_published_utc` column, and `attach_clv` uses it ahead of the slate run time. Rows without `published_at` (51 of 131 FanDuel picks) keep the old fallback. `_published_utc` is dropped before `clv_picks.csv` is written, so the output schema is unchanged.
2. `scripts/clv_report.py` close line (minor, valid): with no exact line at close, `at_close.iloc[0]` picked an arbitrary alternate, so `line_clv` depended on row order. Fix: use the most price-balanced line (the same main-line rule as the live FanDuel read view), with ties going to the nearest line, then the lower line. Stored data has no primary/alternate flag; that refinement is not done.
3. `scripts/clv_report.py` grading (minor, valid): a missing box-score component made `actual` NaN, which graded OVER as WIN and UNDER as LOSS. Fix: skip (leave ungraded) unless every component is finite.
4. `scripts/audit_live_gates.py` receipt (major, valid edge case): an empty probability bucket produced NaN, and `json.dumps(allow_nan=False)` aborted the receipt. Fix: write `n: 0` with null metrics.
5. `tests/test_artifact_retention.py` subprocess cleanup: already fixed in commit 6224438 (CodeRabbit compared against an older `origin/master`). No change.

**Verification:** 3 new tests in `tests/test_clv_report.py`, which fail on the HEAD version of `clv_report.py` and pass with the fix. `python -m pytest -q tests --disable-warnings`: 262 passed, 10 subtests passed. CodeRabbit re-review of uncommitted `scripts/` changes: 0 findings.

**Impact:** Reporting and audit only; no pick, gate, model or history change. CLV/ROI in `output/clv_summary.json` may move on the next run. Not covered: `nba_props.py` (including the DEC-077 change) was not reviewed by CodeRabbit because of the payload limit.

**DEC-081 follow-up (2026-11-13):** `coderabbit review --agent -t committed --base-commit 3f6a37d` reviewed commit 6224438 only (`nba_props.py`, `tests/test_season_games_propagation.py`, `tests/test_artifact_retention.py`, three output docs) and returned 0 findings. Scoping by base commit keeps unpushed `models/` artifacts out of the payload. This covers the changed lines of `nba_props.py`, not the whole file.

## 2026-11-13 — DEC-082 — CodeRabbit review of unpushed root code: fix grading data-gap DNPs, slate timezone crash, retention gaps

**Context:** Five local commits ahead of `origin/master` change 16 root Python files (about 2,000 lines, including `nba_props.py` and `run_daily.py`). A normal review exceeds CodeRabbit's 100 MiB limit because those commits also carry `models/` and `data/` artifacts. The review ran in a scratch git worktree: `origin/master` plus only those 16 files, reviewed as uncommitted. CodeRabbit returned 8 findings, each checked against the code.

**Fixed (5):**
1. **Grading, data integrity (`run_daily.py`, picks Step 1.5 and `_grade_candidate_rows`).** A date with no box scores that was more than 3 days old was graded DNP for every row, permanently (later runs only grade blank results). It ran even when the gap was a data or API failure. **Confirmed impact:** all 9 `picks_history.csv` rows for 2026-04-11 are DNP (graded 2026-04-15 and 2026-11-05), while `data/nba_data.csv` has no 2026-04-11 rows (it skips from 04-10 to 04-12) and `historical_lines.csv` has 2,205 lines for that date. Games were played; the outcomes were lost. Fix: leave those rows ungraded and log a `!!!` warning asking for a backfill. Players missing from a date that does have box scores are still graded DNP. `tests/test_candidate_history.py` encoded the old rule; it now asserts the rows stay ungraded. The 9 stored rows are **not** modified (DEC-070); TASKS has the backfill item for Jake.
2. **Morning-run crash (`nba_props._resolve_active_slate_date`).** `predict` passes a UTC-aware `as_of`; with no exact-today FanDuel lines cached, comparing it with naive slate dates raised `TypeError: Cannot compare tz-naive and tz-aware timestamps` before the live refresh. Reproduced. Fix: convert an aware `as_of` to the local wall-clock date before comparing.
3. **Alternate-market price match (`nba_props`, price gate).** `player_assists_alternate` was not mapped to AST, so an alternate pick could not find its prices. Fix: strip `_alternate` before mapping. Dormant (`live_fetch_alternate_markets` False; no `data/fanduel_alternate_lines.csv`).
4. **Retention, path truncate (`artifact_retention.py`).** A path-based `os.truncate` emptied a protected file without an `open` event, so no version was kept. Fix: handle the `os.truncate` audit event. A descriptor-based truncate needs an earlier writable `open`, which was already preserved. Not used anywhere in project code today.
5. **Retention, byte paths.** `Path(source)` raised `TypeError` for a bytes source in `os.replace`, blocking the write. Fix: `os.fsdecode`.

**Recorded, not changed (3):**
- Alternate-line selection keeps main-line edge, support flags and sanity result after moving the line. Valid but dormant; logged as a blocker before enabling alternates. Fixing it means re-running gate logic, which needs its own plan.
- Promotion of an earlier-rejected candidate keeps that run's price, EV and failure text, so "Published (bet)" ROI in candidate reporting uses the rejected price. Valid, reporting only. A fix needs a schema decision (new columns) or a cross-file lookup; left for Jake.
- `dir_fd`-relative removal and rename. Not possible on this platform (`os.unlink`/`os.rename` do not support `dir_fd` on Windows), and unused in the project.

**Verification:** New `tests/test_slate_and_alternate_fixes.py` (2) and `test_path_truncate_and_byte_path_replace_are_preserved` fail on the HEAD versions and pass with the fixes. `python -m pytest -q tests --disable-warnings`: 265 passed, 10 subtests passed. CodeRabbit `review -t uncommitted` over all working-tree changes (including `tests/test_clv_report.py` from DEC-081): 0 findings. The scratch worktree was removed.

**Pick impact:** No change to selection, gates or models. Grading now leaves missing-data dates pending instead of DNP. The timezone fix only changes behavior where the old code crashed.


## 2026-11-15 — DEC-083 — Read-only Command Center dashboard (`scripts/build_dashboard.py`)

**Context:** Jake asked for a modern dashboard of all metrics, records and analytics. The existing outputs were `output/dashboard_latest.html` (pipeline-generated, pick-centric) and the Streamlit `vegas_gui.py`. Neither shows equity, gate effectiveness, CLV, model evidence, pipeline health and the decision log together, and neither flags data-integrity problems.

**Decision:** Add a standalone generator, `scripts/build_dashboard.py` with assets in `scripts/dashboard_assets/` (`template.html`, `style.css`, `charts.js`, `app.js`), launched by `run_dashboard.bat`. It writes one self-contained HTML file, `output/dashboard_pro.html`, with no network dependency (custom SVG charts, no external libraries). It is a derived view: it can always be regenerated from the sources, so it overwrites only itself and keeps no timestamped copy.

**Safety contract:**
- Reads `output/`, `data/`, `models/`, `logs/` and git metadata; writes exactly one file, atomically. A test hashes a synthetic project tree before and after a build and asserts nothing else changes (DEC-070).
- Not wired into `run_daily.py` or the scheduler, so the morning run cannot be affected. Hooking it in is an optional follow-up in TASKS.md.
- Every section runs in an isolation wrapper. A missing or malformed input degrades that section, is listed in a warning banner and printed to stderr, and never produces a silent number. Missing `picks_history.csv` is fatal.
- Reporting rules: flat 1-unit P&L at the logged FanDuel price only; a pick with no logged price (and no CLV-report bet price for the same pick) gets no units rather than a guessed price; DNP is void; PUSH is 0 units; the 2026-27 season, all-live and everything-incl-backfill scopes are separate, and the backfilled March era is labelled as not a forward record (DEC-036). Hit rates carry Wilson 95% intervals and small samples are tagged.
- CLV-report outcomes appear on pending picks only as clearly labelled **provisional** results and never enter a record or unit total.

**Integrity findings surfaced (reported, not changed):** (1) `betslips_history.csv` CSV line 33 is an orphan row with no date and no legs (`result=LOSS`, `actual_payout=0.0`). The dashboard excludes it from slip numbers and flags it; removal is Jake's call (DEC-070). (2) 104 of 286 slips repeat identical legs under a second slip label, so slips are reported unique (headline) and raw. (3) The 2026-04-11 all-DNP date (DEC-082) is flagged. (4) The CLV report grades 2026-11-14 picks from final box scores before `picks_history.csv` does (Step 1.5 runs the next morning); on every pick settled in both files the grades agree.

**Verification:** `tests/test_build_dashboard.py` (29 tests): helpers, parsers, pinned record/unit/net-dollar numbers on a synthetic project, orphan and duplicate slips, all-DNP flag, provisional results, strict-JSON and `</script>` escaping, read-only guarantee, loud degradation, fatal inputs. Three deliberate mutations (orphan rows kept, run end not closed at a log gap, unpriced picks given guessed units) each fail a test; sources restored byte-identical. Independent recomputation outside the builder matched the dashboard exactly: season 63-41, +8.0004u over 104 priced bets, slips 181 unique (77-88-3, +$3,319.94). Full suite `python -m pytest -q tests`: 294 passed, 10 subtests passed. Visual check in the in-app browser at desktop and 375px width, dark and light themes, all 10 views with no horizontal overflow and no console errors.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change. Files added: `scripts/build_dashboard.py`, `scripts/dashboard_assets/*`, `run_dashboard.bat`, `tests/test_build_dashboard.py`, `output/dashboard_pro.html` (generated). README gained a short entry.

**Known limits:** Price-less historical eras (backfill, PrizePicks) show hit rates but no units. Slip dollars in the PrizePicks era are dominated by fixed payout multipliers and the backfilled era is reconstructed, so they are not forward expectations. Closing-line value uses the nearest T-5/T-60 capture (63 exact-close picks), so it is directional only.


## 2026-11-15 — DEC-084 — Dashboard gets circular dials, pie/donut charts and real NBA team logos

**Context:** Jake asked the Command Center (DEC-083) for circular dials, pie graphs and real NBA team logos.

**Decision:**
- **Charts (no library, same SVG toolkit):** semicircle dials with zones, needle and a break-even tick (hit rate, ROI, edge vs break-even, green days, meta-classifier AUC for all nine props, beat-the-close, market direction, published-candidate ROI, slip ROI, Odds API credits, model-vs-market on the card); concentric progress rings (lane hit rates with break-even ticks, settlement, holdout progress, file freshness); pies and donuts with hover lift, centre labels and a legend (result mix, lane share, team share with logos, price/probability/EV bands, candidate flow and stop reasons, close quality, slip results and types, line sources, box-score seasons).
- **Logos:** the browser loads each official logo from the NBA's public CDN (`https://cdn.nba.com/logos/nba/<teamId>/global/L/logo.svg`) when the page is opened. Nothing is downloaded into the project. Requests carry `referrerpolicy="no-referrer"`; the only data sent is the team ID. If a logo cannot load (offline, blocked, CDN change) that team shows a team-coloured abbreviation badge, so the page degrades rather than breaks. Logos appear on pick cards, tonight's slate, the team board (teams we picked and opponents we picked against), pick explorer, tables and the team pie legend.
- **Team metadata:** `TEAMS` in `scripts/build_dashboard.py` maps all 30 abbreviations to official team IDs, names and fallback colours. The IDs were checked against `TEAM_ID`/`team` in `data/nba_data.csv` (30 of 30 match) and the 30 abbreviations used in `picks_history.csv` all resolve. An abbreviation outside the table is listed in the page's warning banner and gets a text badge.
- **Supersedes the "no network" wording in DEC-083 and the README:** the page is still a single file with no server and no libraries and works offline, but logo images need internet.

**Bugs found and fixed while verifying (all in the new view code):** (1) logos rendered 0–4 px wide inside a grid box (cyclic percentage width); the logo box is now absolutely positioned. (2) Charts mounted while their view was `display:none` stayed blank until a resize, which showed at phone width: views are now shown before they render, and resize redraws use a timer instead of `requestAnimationFrame` (paused in hidden/emulated tabs). (3) The older hit-rate ring caption rendered dark-on-dark.

**Verification:** `tests/test_build_dashboard.py` is now 38 tests (9 new): team table completeness and official IDs, team and opponent breakdowns, tonight's slate, unknown-abbreviation reporting, provisional results in the slate, an asset contract (the only permitted external host is `cdn.nba.com`; logo requests send no referrer and have a text fallback) and the show-before-render regression. Four deliberate breakages (referrer leak, unknown teams ignored, opponent breakdown built from the wrong column, render-before-show) each fail a test; sources restored byte-identical. Full suite `python -m pytest -q tests`: 303 passed, 10 subtests passed. In the in-app browser at desktop and 375 px: all 10 views draw every chart with no blank charts, no overflow, no console errors, and forcing every logo to fail swaps in the abbreviation badges.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change. Open option (TASKS.md): embedding the 30 logos into the HTML for fully offline use would mean downloading about 30 SVGs (roughly 8 KB each) from the NBA CDN; not done without Jake's go-ahead.


## 2026-11-15 — DEC-085 — Dashboard logo embedding built; the 30 logo files are not downloaded yet; DEC-084 CSS collision fixed

**Context:** Jake approved embedding the team logos so the dashboard works fully offline (the open option in DEC-084). The download itself (30 SVGs from `https://cdn.nba.com/logos/nba/<teamId>/global/L/logo.svg`) was denied by the session's permission classifier, so it was **not** run and no other route to the same files was tried.

**Decision:**
- **Where logos live:** committed files `scripts/dashboard_assets/logos/<ABBR>.svg` (file name = the abbreviation in `TEAMS`, e.g. `CLE.svg`). `load_logos()` validates each file and inlines it into the page as a base64 `data:` URI; the builder never touches the network. `build_payload(..., logo_dir=...)` lets tests point it elsewhere.
- **Validation (`validate_logo_svg`, fail loudly):** at most 64 KB, well-formed XML with an `<svg>` root, no DOCTYPE/ENTITY, no `script`/`foreignObject`/`iframe`/`object`/`embed`/`image`/`animate`/`set`, no `on*` attributes, no `href` other than a `#fragment`, no `url()` or `@import` to anything but a `#fragment`, no `javascript:`. The files are third-party and only ever drawn as CSS backgrounds (browsers run no script there), so this is defence in depth. A rejected file or an unknown file name is listed in the page's warning banner and that team keeps the fallback.
- **Front end:** each embedded logo is stored once, in one generated `<style>` rule per team (`.tlogo.e-ABC i{background-image:url(data:…)}`), not repeated in every image tag. Only a plain base64 SVG data URI for a three-letter key is ever injected. A team with no embedded logo still gets the DEC-084 behaviour (NBA CDN image, then a team-coloured badge offline), so a partial set never breaks the page.
- **Loud, not silent:** while any team has no embedded logo the builder warns `N of 30 team logos are not embedded and load from the NBA CDN when online (text badge offline): …`. Today that reads 30 of 30. The warning disappears once all 30 files are present.

**Bug fixed (introduced in DEC-084):** the logo box class was `.tl`, the same name as the decision-log timeline (`.tl`, `#dc-list`). Every logo inherited the timeline's 18 px left padding, 8 px margin and 2 px left border, and the timeline inherited the logo's 26 px square, so the 84-entry decision log rendered inside a 26×26 px box. The logo class is now `.tlogo`; a regression test pins that each class is defined once and the helper emits only `.tlogo`. The DEC-084 view sweep missed this because it checked logo size and chart drawing but not the Decisions list.

**Verification:** `tests/test_build_dashboard.py` is now 48 tests (10 new): the validator accepts clean SVG (including `url( #g )` spacing) and rejects 12 hostile variants plus malformed, wrong-root, empty and oversized input; `load_logos` embeds byte for byte and reports bad or unknown files; the payload names exactly the missing teams; a full set of 30 embeds with no warning and survives rendering; the asset contract for the generated stylesheet; the class-collision regression; and a check that every committed logo file passes validation. Eight deliberate breakages (allow script, `on*`, external href, external `url()`, DOCTYPE, no size cap, no missing-logo warning, skip validation) each fail a test; source restored byte-identical. Full suite `python -m pytest -q`: 313 passed, 26 subtests passed. In the in-app browser, with 30 SYNTHETIC placeholder logos (never committed), all 10 views had square logos at the requested size, zero CDN requests and no console errors, and the Decisions timeline measured 818×630 px with 84 entries; with 29 embedded, the 30th used the CDN image and the banner named it. The real logo look is therefore not yet seen.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change; `output/dashboard_pro.html` was regenerated. Remaining step (TASKS.md): get the 30 official SVGs into `scripts/dashboard_assets/logos/` and rebuild.


## 2026-11-15 — DEC-086 — Team logos embedded; dashboard now works fully offline

**Context:** DEC-085 built the embedding but the session's permission classifier denied the download twice (including after Jake said to retry), so no download was run from the session and no other route to the same files was tried. Jake then downloaded the 30 official SVGs himself into `scripts/dashboard_assets/logos/<ABBR>.svg`.

**Decision:**
- **Files accepted:** all 30 (3 KB–31 KB each, 325 KB total) pass `validate_logo_svg`; an inventory of every element and attribute found only plain vector shapes, `<style>` class fills, `<defs>` and one `<clipPath>`, with no script, link, image or external reference. Each logo is drawn as its own CSS background image, so the style classes inside different logos cannot collide. `output/dashboard_pro.html` was rebuilt: 1.3 MB, 0 warnings.
- **Offline proof:** in the in-app browser the page made no request to `cdn.nba.com` (the network log shows only the page itself), all 10 views drew every logo square at its requested size, and nothing overflowed at 375 px.
- **Legibility fix:** dark marks (Spurs, Nets, Jazz) were hard to read on the dark theme. Logos now carry a thin light halo in the dark theme; the light theme keeps the soft dark shadow. Checked in both themes.
- **Provenance:** `scripts/dashboard_assets/logos/README.md` records the source URL pattern and download date. The CDN image path stays in the code only as a fallback for a missing or rejected file; the banner reports it.

**Verification:** `tests/test_build_dashboard.py` is 48 tests; the former "committed logos are valid" test now requires all 30 files to be present and valid. Full suite `python -m pytest -q`: 313 passed, 56 subtests passed. `picks_history.csv`, `betslips_history.csv` and `nba_data.csv` modification times are unchanged.

**Impact:** Reporting only. Supersedes the "logo images need internet" caveat in DEC-084. No pick, gate, model, schema, history, scheduler or `run_daily.py` change.


## 2026-11-15 — DEC-087 — Dashboard readiness review: daily refresh, trades, injuries, other dashboards

**Context:** Jake asked whether the dashboard's metrics will update daily from results, whether it can handle injuries and trades, and said it should be the only dashboard still active. Answers below were checked against the code, the scheduler and the data, not assumed.

**Findings (verified):**
- **Daily refresh is NOT automatic yet.** Nothing calls `scripts/build_dashboard.py` except `run_dashboard.bat`; the page is a snapshot of the files at build time. The live schedule is the Task Scheduler job `FreshStartNBA-DailySafe` at **07:30** (not 09:00 and not `run_morning.bat`; `CLAUDE.md` is stale on both): `scripts/run_scheduled_action.ps1 -Action DailySafe` → `scripts/project_operator.ps1`, which runs `run_daily.py --skip-weekly-retrain` and then `scripts/capture_candidate_future_holdout.py` (a file the dashboard reads). `PicksNow` and `PredictCached` regenerate picks on demand and also leave the page stale.
- **Where a daily rebuild should hook in (not done, needs Jake's go-ahead because it is the live scheduler path):** at the end of `DailySafe` after the holdout capture (and after `PicksNow`), as a non-fatal step that cannot change the run's result. It should call the builder directly, not through `Invoke-PythonStep`: that helper wraps every `.py` in `artifact_retention.py --run`, which keeps every replaced file as a compressed version with no expiry, pointless for a regenerable 1.3 MB page.
- **Trades:** history is safe. Every pick keeps the team it was made for (8 players in `picks_history.csv` have more than one team, e.g. Jaylen Brown BOS→PHI, LeBron James LAL→PHI). **Bug found and fixed:** `picks_history.csv` is not in date order, and the Players table took each player's last row in file order, so Kelly Oubre Jr. (PHI→IND) showed PHI. It now takes the latest team by `game_date`. A team abbreviation outside the 30 in `TEAMS` is listed in the warning banner and gets a text badge.
- **Injuries:** the page does not fetch or compute injury status; the pipeline does (Layer 3, `data/live_injury_status.csv`, official rosters). The page reports outcomes: a player who does not play grades DNP and is void (no units). Not shown yet: `injury_adj` (71 of 1,726 picks) and `teammates_out` (names, e.g. Joel Embiid out on 2026-11-15 picks), and the live injury file. A player ruled out after the page was built is **not** flagged.
- **Other dashboards:** `output/dashboard_latest.html` is still generated by `nba_props.py predict` (Step 9, with dated copies under `output/archive/`); the Streamlit "Vegas Intelligence Console" (`vegas_gui.py`, `run_gui.bat`) embeds that file and also has run-a-command buttons, which the read-only dashboard does not. All left as they are.

**Changes made (dashboard only):** (1) trade-safe team attribution (above); (2) the page now shows how old it is in the header and a red "Out of date" banner once it is 26 hours old, so a missed daily rebuild cannot go unnoticed.

**Verification:** `tests/test_build_dashboard.py` is 51 tests (3 new): latest team by date with out-of-order file rows, a trade does not rewrite pick-time team history, and the staleness banner contract. Reverting the sort fails both trade tests; restored byte-identical. Full suite `python -m pytest -q`: 316 passed. In the in-app browser, pages stamped 20 minutes, 30 hours and 3 days old showed no banner, a banner and a banner respectively, with the age in the header.

**Note:** a manual `run_daily.py` (started 09:15) was still running while this was verified; it is not part of this change. The page was rebuilt during that run and should be rebuilt after it finishes.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change. Open decisions for Jake are in TASKS.md (schedule the rebuild, add an availability panel, retire the other dashboards).


## 2026-11-15 — DEC-088 — Dashboard now rebuilds at the end of the scheduled daily run

**Context:** Jake approved the proposal in DEC-087 to add the dashboard rebuild to the 07:30 `FreshStartNBA-DailySafe` job. This touches the live scheduler path (`scripts/project_operator.ps1`), so it was built to be incapable of affecting picks.

**Decision:**
- **Where:** `DailySafe` now runs `Safe daily pipeline` and `Grade and capture frozen candidate future holdout` inside a `try`, and calls the new `Invoke-DashboardRebuild` in the `finally`. The two existing steps, their labels, order and protections are unchanged. Other actions (`PicksNow`, `PredictCached`, manual) are not hooked; `run_dashboard.bat` still rebuilds on demand.
- **Why a `finally`:** the page should be rebuilt even when the pipeline reports an error, so its System Health view shows the failure. The function cannot throw, so the original error still propagates unchanged and a passing run cannot become a failing one.
- **Cannot block or break the run:** the builder is not in `Assert-RequiredFiles` or the Preflight `py_compile` list (a dashboard bug can never stop picks); the function has a catch-all, a 180 s limit (a hung builder is stopped), and writes `[Dashboard rebuild] OK`, or `FAILED` / `SKIPPED` / `NOT refreshed` with the builder's own error text, to `logs/scheduled_dailysafe.log`.
- **Not routed through `Invoke-PythonStep`:** that helper wraps `.py` steps in `artifact_retention.py --run`, which keeps every replaced file as a compressed version with no expiry; for a regenerable 1.3 MB page that is pure growth. The rebuild calls `python scripts\build_dashboard.py` directly.
- **Builder hardening:** `main()` now makes stdout/stderr tolerant of characters outside cp1252 (Task Scheduler redirects output), so a non-ASCII name in a warning cannot crash the run after the page is written.

**Verification:** `scripts/project_operator.ps1` parses with 0 errors and keeps consistent CRLF line endings. `tests/test_dashboard_rebuild_step.py` (10 tests) pins the structure and runs the REAL function against stand-in builders: success relays output and logs OK; a failing builder is logged with its error and not raised; a hung builder is stopped at the limit; a missing builder is skipped loudly; a pipeline error survives the `finally` while the page is still rebuilt; a failed rebuild cannot turn a passing run into a failure. Six deliberate breakages of the script (not in a `finally`, throws on failure, retention-wrapped, no time limit, exit code ignored, builder added to Preflight) each fail a test; the script was restored byte-identical. One extra test covers the cp1252 console fix (fails without it). The real function was run against the real project: exit 0 in 6 s, `dashboard_pro.html` rewritten. Full suite `python -m pytest -q`: 327 passed. The real `DailySafe` was not run (it calls paid APIs).

**Not yet verified:** the first scheduled run. Tomorrow's 07:30 log should end with `[Dashboard rebuild] OK`, and the page header should read `Built … (under 1 hour ago)`.

**Impact:** No pick, gate, model, schema, history or `run_daily.py` change. The only live-path change is the `try/finally` and one non-throwing function in `scripts/project_operator.ps1`.


## 2026-11-15 — DEC-089 — Dashboard gets an optional, toggleable wallpaper

**Context:** Jake supplied an image (AI-generated artwork of NBA players beside sports cars at a mansion at sunset, 1792×1008 JPEG, 353 KB) and asked for it as a transparent wallpaper for the dashboard, still clearly visible.

**Decision:**
- **Where it lives:** `scripts/dashboard_assets/wallpaper.jpg`, copied byte-identical from the supplied file. It is optional: the builder also accepts `wallpaper.jpeg|png|webp`, and with no file the dashboard is exactly as before. To change it, replace the file and rebuild.
- **Embedding (`load_wallpaper`):** inlined byte for byte as a `data:` URI (one offline file; no new host, no image library, no download). The MIME type comes from the file's magic bytes, not its extension. An empty, over-1.5 MB or non-image file is not embedded and is named in the warning banner. The page grows from about 1.33 MB to 1.8 MB.
- **Look:** a fixed layer under the existing glow and grid, so it stays put while you scroll. Dark theme 64% and light theme 46% opacity. While it is on, panels become slightly more transparent (dark `.54`, light `.68`) and stay blurred so the image shows through but numbers and charts stay readable; the header and warning banners get their own frosted backing because they sit directly on the page background. Tuned by eye on Overview, Today and the pick explorer at 1440 px, in both themes and at 375 px.
- **Control:** a wallpaper button beside the theme button turns it on or off; the choice is remembered in the browser (`nba.wall`). Every override lives behind `[data-wall="on"]`, so off restores the original look exactly. The front end only ever injects a plain base64 JPEG/PNG/WebP data URI.

**Verification:** `tests/test_build_dashboard.py` is 60 tests (8 new): no file is silent; JPEG/PNG/WebP embed byte for byte with the right MIME; HTML or empty or oversized files are rejected whatever their extension; the payload warns loudly about a bad file and not about an absent one; the wallpaper survives rendering and adds no external host or `<img>`; the committed file validates; the CSS/JS contract (strict data-URI pattern, panel overrides only behind `data-wall="on"`). Four deliberate breakages (no magic-byte check, no size cap, unscoped panel override, loose front-end pattern) each fail a test; sources restored byte-identical. One existing test that searched the data block for the text `NaN` was changed to look for a bare `NaN` value, because base64 text can contain those letters by chance. Full suite `python -m pytest -q`: 335 passed. In the in-app browser: toggle off/on, persistence across reload, light and dark themes, and 375 px with no overflow.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change; `output/dashboard_pro.html` was rebuilt (1.8 MB, 0 warnings).


## 2026-11-15 — DEC-090 — Side-menu icons become animated 3D icons on desktop (phone layout unchanged)

**Context:** Jake asked for more appealing, eye-catching, 3D animated side-menu icons, then specified: leave the phone version as it is and animate only the desktop version opened in a browser.

**Decision:**
- **Real depth, no library:** each of the 10 menu icons (Overview, Today, Results, Picks, Gates, Market, Models, Slips, Health, Memory) is a new solid, colour-coded icon built as a stack of identical SVG layers at different depths: 8 dark extrusion slices, a gradient face (white detail) and a gloss highlight, inside a CSS perspective box. Because the slices are real layers in 3D space, tilting shows true side depth. All gradients live in one hidden SVG; nothing is downloaded or embedded as an image.
- **Motion:** every icon floats and sways (staggered so they never move in sync) over a soft ground shadow; hovering or keyboard focus scales it and tilts it toward the pointer; the current section's icon spins once every 8 seconds and glows. `prefers-reduced-motion` gives a still, tilted icon with no pointer tilt.
- **Desktop only:** the 3D markup and the original flat icon are both rendered and CSS chooses: at 860 px and below (the existing phone layout) the 3D icon is hidden and the original flat stroke icons are shown with the original padding, size and colours, and no animation runs. Above 860 px the flat icon is hidden.
- **Pre-existing bug found:** `app.js` marks the current section with `aria-current="true"` while the old CSS matched `aria-current="page"`, so the active-item highlight and accent bar never appeared. To keep the phone exactly as it was, the highlight is now applied on desktop only (inside `@media (min-width: 861px)`, keyed to `"true"`); the old `"page"` rules are left in place and still match nothing on the phone.
- Desktop menu buttons got slightly tighter padding (6 px) so ten 28 px icons still fit a normal screen without scrolling the menu.

**Verification:** `tests/test_build_dashboard.py` is 63 tests (3 new): every menu section has a complete 3D icon (a missing one would otherwise render blank), depth/motion/reduced-motion contracts, and the phone guarantees (3D hidden, flat icon rule and the original phone padding/active rules unchanged, the active highlight only inside the desktop media block). Five deliberate breakages (phone shows 3D, a section loses its icon, reduced-motion rule removed, highlight leaks onto the phone, original phone padding altered) each fail a test; sources restored byte-identical. In the in-app browser: all 10 icons inspected magnified; hover tilt sets and clears its tilt; the active icon highlights and spins and follows navigation; at 375 px the top bar is the original 21 px flat icons with no animations running and no overflow. Full suite `python -m pytest -q`: 338 passed.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change; `output/dashboard_pro.html` was rebuilt (1.8 MB, 0 warnings).


## 2026-11-15 - DEC-091 - Diagnose the CLV alert and expose its measured value and cohorts

Jake requested that the five screenshot findings be addressed one at a time. Work started with the first, closing-line value. The overview alert was hard-coded as a warning regardless of results. It now warns when measured average closing-price EV is negative and reports informational status for positive, zero, or unavailable measurements. Positive averages are described without asserting proven edge.

The alert displays average EV, exact-line sample size, price movement, settled ROI sample size, and coverage. ROI includes changed-line closes, so its cohort is labelled explicitly. The generic -110 vig baseline was removed from this alert because the actual offered prices vary. The existing report and market view remain the source of the underlying metrics.

Independent recomputation from output/clv_picks.csv agrees with output/clv_summary.json. There are 132 picks, 73 usable closes, 63 exact-line picks across 13 dates, and 71 settled picks with usable closes. Exact-line average EV is -0.0510142, beat-close rate 0.1904762, mean probability movement +0.0062899. Settled ROI is +0.0729410. There are 59 picks without a later snapshot and 10 changed closing lines. Usable snapshots range from 8 to 808.2 minutes before tip (median 24.7).

The negative EV and incomplete/early closing captures remain open evidence issues in TASKS.md. This reporting correction establishes neither improved prediction performance nor positive betting edge.

Verification ran python -m pytest tests/test_build_dashboard.py tests/test_clv_report.py -q, with 76 passed and 56 subtests passed. Four new regression tests cover negative EV despite positive movement/ROI, positive/zero EV, missing/nonfinite measurements, and changed-line-only samples. The dashboard was regenerated from existing inputs.

## 2026-11-15 — DEC-092 — Atomic availability-model writes and a training-aware production-bundle guard

**Problem:** A manual full `run_daily.py` (started 09:15) logged `[FAIL] Weekly maintenance mutated the production model bundle: availability_minutes_model.pkl`. The separate `FreshStartNBA-TrainingRefresh` job (09:30) retrained that file at 09:33:23 while Step 8.5 was running. Nothing in `run_daily` wrote it, and the previous version was retained. Two real defects were behind it: (1) `scripts/train_availability_minutes_model.py` wrote the pickle in place (`open("wb")`), so a concurrent `predict` could load a partial file; (2) the guard could not tell an expected TrainingRefresh update from an unexplained mutation.

**Decision (per Jake: production must be safe when jobs overlap unexpectedly):**
- New `atomic_io.py`. Every write goes to a unique same-directory temp file (`.<name>.<random>.tmp`), is flushed and fsynced, then swapped in with `os.replace`. On Windows the swap is retried for up to 60 s while a reader holds the file open, then fails loudly with the previous version intact. The temp file is removed on every failure path.
- The trainer now writes all seven outputs atomically (model pickle, policy, summary and four CSVs). Before swapping in the model and policy, it writes an append-only receipt to `models/training_receipts/` naming their exact SHA-256 hashes. Because the receipt comes first, a guard can never see a new file without its receipt. Pickle protocol and CSV bytes are unchanged.
- `run_daily` guard: `TRAINING_REFRESH_OWNED` = {`availability_minutes_model.pkl`, `availability_minutes_policy.json`}. A change to one of these is logged as an expected `[WARN]`, not an error, only if all of the following hold:
  - the file still exists;
  - its current hash equals a hash in a receipt written since maintenance started (15 min slack);
  - that receipt was not written by this run's own child steps. Step 8.5 sets `NBA_PIPELINE_CONTEXT=run_daily_weekly_<stamp>_<pid>`; children inherit it, and receipts carrying it are excluded.
- Anything else still fails exactly as before, including an owned file changed without a receipt, a receipt with a mismatched hash, a deleted file, or any other production file. Calls that do not pass the window keep the old strict behavior. The hash set ignores `*.tmp` files, so an in-flight temp file cannot cause a false failure.
- `_production_model_hashes` now resolves `MODEL_DIR` at call time (previously a default argument bound at import).

**Not changed:** pick logic, gates, features, model training, model contents, schemas, the scheduler, and `run_training_maintenance.py`, whose own guard globs only top-level patterns, so the receipt folder is outside it.

**Verification:**
- `tests/test_atomic_writes_and_training_guard.py`: 20 new tests. They cover a real open-handle swap on Windows, and a two-process test in which 40 alternating 2 MB atomic writes were read concurrently with no partial or mixed read. Guard cases: accept, own-context, hash mismatch, non-owned file, deletion, stale receipt, mixed change, legacy call, temp file, corrupt receipt, and context propagation.
- Seven deliberate breakages were all caught: non-atomic write, no retry, no owned check, no context exclusion, no hash match, temp files hashed, context not set.
- Real end-to-end trainer run with all outputs redirected to the scratchpad (rc 0): the receipt hashes matched the written files; the CSVs were byte-identical in size to production's; the model's predictions on 5,000 rows differed from the production model by 0.0 (same training cutoff 2026-11-14).
- Full suite: 362 passed.

**Interim rule:** avoid starting manual full runs between 09:15 and 09:40 until a real overlap has been observed passing (`[WARN] ... updated by the TrainingRefresh job ... expected` with no `[FAIL]`), or until the first daily TrainingRefresh after this change has written a receipt and the next Sunday full run passes the guard.

## 2026-11-15 - DEC-093 - Remove misdated April 11 records with user authorization

This corrects the factual diagnosis in DEC-082 and its backfill task. The official NBA schedule confirms zero games on April 11. The 2,205 historical line rows establish a capture/publication date. They do not establish that games occurred on that date. No April 11 box scores need backfilling. Source https://www.nba.com/games?date=2026-04-11

All nine players have April 12 final box scores. Their relevant statistics were independently checked against ESPN summary events 401811041, 401811044, 401811047 and 401811055 and agree with data/nba_data.csv. Assigning the archived lines to April 12 would produce 5 wins and 4 losses. That date assignment remains an inference because the old projection archives omitted event IDs and start times. Jake was given both choices and requested that the records be disposed of.

Removed nine April 11 picks and the 17 April 11 slips whose legs map to those nine contracts. Picks history changed from 1,726 to 1,717 rows and slips history from 286 to 269. Every surviving cell matches its original value. No reconstructed outcomes were added. Complete pre-change history bytes are retained under .retained_versions/objects using the SHA-256 hashes in output/slate_date_audit_2026-04-11.json. That receipt records all removed rows, source checks, user instruction and applied disposition.

Moved all 20 files from output/archive/2026-04-11 into .retained_versions/quarantine/2026-04-11 and verified each file hash. This removes their source from both Step 1.5 archive backfill and recover_lost_history_rows.py. Raw captured line observations are preserved. The box-score dataset hash matches its pre-change hash.

Verification covers exact before/after cell comparisons, zero remaining April 11 history rows, identical original April 12 rows, retained history checksum verification, archive recovery discovery, existing dashboard/slate/parser tests, and dashboard rebuild. The resulting page has 1,717 picks, zero build warnings and no April 11 integrity flag. Current-season record remains 70-43 because this correction concerns spring records. STATUS.md and TASKS.md record completion and supersede the original backfill task.


## 2026-11-15 - DEC-094 - Investigate the orphan betslip

Jake requested investigation of the orphan row flagged by the dashboard. CSV line 33 in output/betslips_history.csv has exactly two populated values, result LOSS and actual_payout 0.0. There is no date, slip type, wager or leg identity with which to reconstruct a bet.

Read every committed version of that history file. The orphan appears in eb1e973 (April 3) and every subsequent committed version. The prior available version, 9a7aaa3 (March 25), has only three rows and no orphan. The exact creating operation cannot be established within that interval. A scan of all active output/archive betslip CSVs found no legless row to recover or reimport.

Current log_betslips builds dated rows from selected subsets with at least two legs and initializes grades blank. Step 1.6 skips dates absent from the box-score date set and skips zero-leg rows. Those normal paths do not establish a current cause for this inherited malformed record. The Step 1.6 all-time graded-row count includes every nonblank result, so it can still count this inherited LOSS. The dashboard explicitly excludes it.

Current raw history has 269 rows, 133 WIN, 129 LOSS and 5 PUSH. Excluding the orphan would give 268 rows, 133 WIN, 128 LOSS and 5 PUSH. The raw W/L win rate changes from 50.763% to 50.958%. This does not address duplicate slip labels, which remain a separate task.

Investigation only. Saved history was not edited in this step. The correction remains removal of this one invalid row with its original retained, under the existing task. Verification ran python -m pytest tests/test_build_dashboard.py::PayloadTests -q, with 14 passed, including orphan exclusion from dashboard counts and its integrity flag. STATUS.md and TASKS.md record these findings.


## 2026-11-15 - DEC-095 - Remove the orphan betslip after user authorization

Following DEC-094, Jake instructed "ok then proceed". Removed exactly CSV line 33, whose only populated fields were result LOSS and actual_payout 0.0, from output/betslips_history.csv. No valid date, wager, leg or archived counterpart existed for reconstruction.

The operation used artifact_lock and an atomic temporary-file replacement. Complete original bytes were retained under .retained_versions/objects and verified by decompression and SHA-256. Every surviving cell and column was compared to the original. The audit receipt is output/orphan_betslip_removal_2026-11-15.json and includes the removed row, original object path, before/after file hashes, counts and user authorization.

Active history changed from 269 to 268 rows. It now contains 133 WIN, 128 LOSS, 5 PUSH and 2 pending records. Duplicate slip labels remain a separate investigation. The earlier archive scan found no source for this malformed row. The dashboard was rebuilt with zero warnings, and its orphan count is zero. Verification used exact data comparisons and the existing dashboard payload tests. STATUS.md and TASKS.md record completion.


## 2026-11-15 - DEC-096 - Apply verified corrections to the five grade conflicts

Jake requested execution of the grade reconciliation correction. Verified three final games using ESPN summaries 401810916, 401909944 and 401909958. The March 25 Clingan 14 points and November 7 Wembanyama 2 blocks were additionally corroborated by the NBA official scorer reports. NBA live JSON endpoints returned 403, and the November 9 NBA PDF was unavailable, so the November 9 final count was verified from the ESPN box score. All three external counts agree with the current local box-score dataset.

Applied exactly five cell changes across five rows. Donovan Clingan, March 25, PTS OVER 14.0, actual 14.0, changes LOSS to PUSH in picks_history.csv. Victor Wembanyama, November 7, BLK UNDER 3.5, changes actual 1 to 2 in picks_history.csv and candidates_history.csv. November 9 BLK UNDER 3.5 changes actual 9 to 8 in both files. Those four stat corrections keep their existing outcomes and candidate profit values. The earlier dashboard wording about a W/L flip referred to the Clingan LOSS-to-PUSH correction.

History files were locked, complete original bytes retained and verified by decompression and SHA-256, and replacements staged and verified before atomic publication. The audit receipt output/grade_corrections_2026-11-15.json stores before/after cell values, source evidence and hashes, correction timestamp and source file hashes. Row counts, column order and every other cell were verified against the originals. The initial grading timestamps remain evidence of when the original grades were written; the audit records the correction time.

Refreshed scripts/reconcile_grade_evidence.py. It checked 1,814 settled history rows and reports zero current conflicts and zero outcome conflicts. Immutable earlier conflict observations remain retained in output/grade_reconciliation. Current grading already explicitly handles exact-line ties as PUSH. Wembanyama's old values were preserved initial observations that needed reconciliation after final-stat changes, and the corrective operation is audited explicitly.

Verification includes exact five-cell comparison, external-final-stat checks, retained original verification, existing reconciliation/dashboard tests, zero-conflict recomputation, and dashboard rebuild. STATUS.md and TASKS.md record completion.


## 2026-11-15 - DEC-097 - Verify the screenshot's last pipeline error

Jake requested proceeding with the last screenshot issue. The run starting 09:15:20 failed its Step 8.5 production-bundle guard at 09:33:24 because the separate 09:30 TrainingRefresh changed availability_minutes_model.pkl. DEC-092 already implements the correction in the active code, using atomic model publication and matching training receipts for the two legitimately owned files. Unexplained changes, missing receipts, stale receipts, own-context writes and non-owned artifacts continue to fail the guard.

Repeated python -m pytest tests/test_atomic_writes_and_training_guard.py -q, with 20 passed. Ran scripts/project_operator.ps1 -Action Preflight, exit 0, with syntax and dependency checks passing and 56 protected production artifacts hashed.

The five WARNING entries in the historical run are two luck-layer messages, one no-betslips message and two pipeline failure-summary entries. Luck MIN_GP is 15 and current 2026-27 data has at most 14 appearances per player, so neutralizing that layer follows its existing contract. The no-slip message follows the eligible card. These warnings provide operating status.

No new pipeline code change was necessary in this step. The failed historical log remains intact and the dashboard continues to report that run as failed. No subsequent production TrainingRefresh receipt is available yet. The existing DEC-092 task retains explicit verification of a real receipt and a real overlapping full run. Test success is recorded separately from that pending production confirmation. STATUS.md and TASKS.md now record the screenshot investigation.


## 2026-11-16 — DEC-098 — Dashboard renamed "Jake's NBA Dashboard"; first scheduled rebuild verified; push blocked by a 138 MB file

**Context:** Jake asked to retitle the tool "Jake's NBA Dashboard". Earlier in the same session Jake asked to push the commits.

**Decision (rename):** the name now shows in the browser tab (`<title>`), the page header (gradient on "Jake's NBA", plain "Dashboard") and the menu logo tooltip, plus the README heading. The old name no longer appears anywhere in the visible text. Code comments and docstrings, and every earlier entry in this file, still say "Command Center": those are history and internal labels, left as they are.

**Verified (closes the open DEC-088 check):** the 2026-11-16 07:30 `FreshStartNBA-DailySafe` run logged `[Dashboard rebuild] [dashboard] wrote …\output\dashboard_pro.html (1816 KB)` and `[Dashboard rebuild] OK`, then `COMPLETE DailySafe` at 07:52:03; the page was stamped 07:51:58. Note: `logs/scheduled_dailysafe.log` mixes two encodings (the wrapper's START/COMPLETE lines are ASCII; the teed pipeline output is UTF-16), so an ordinary text search misses the teed lines; read them as UTF-16.

**Push check (nothing was pushed):**
- `origin` (https://github.com/hacksumn/NBA-Prop-Maker) is **public** and has only `master` (26c9d3b); `fix/dec-078-082-review-fixes` has no upstream, so a push creates a new remote branch with 14 commits (about 179 MB compressed). Two of those commits (DEC-091..095) were made by another session and would be published too.
- **Blocker:** `output/backups/before_append_only_20261105_041553/historical_lines.csv` is 138.2 MB (added in commit 1b4012c "Update 11/5/26"). GitHub rejects any file over 100 MB, so the whole push would fail. Five candidate CSVs of 64–92 MB (`models/weekly_candidates/*/probability_oof_advanced.csv`) would only draw warnings.
- A scan of the added lines found no API-key, token or password patterns; no secret-like file names are tracked.
- Fixing the blocker means rewriting history (Git LFS migration, or removing the file from history) or keeping the branch local. That is Jake's decision (DEC-070); see TASKS.md.

**Verification:** `tests/test_build_dashboard.py` gained a test pinning the title in the template, the header, the logo tooltip and the rendered page (it fails when the old header text is restored). Full suite `python -m pytest -q`: 363 passed. In the in-app browser the tab title, header, tooltip and visible text were checked on the rebuilt page.

**Impact:** Reporting only. No pick, gate, model, schema, history, scheduler or `run_daily.py` change; `output/dashboard_pro.html` was rebuilt.
