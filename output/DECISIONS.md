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
