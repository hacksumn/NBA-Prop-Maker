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
