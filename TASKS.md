# NBA Prop Predictor — Task Tracker

**Last Updated:** 2026-08-23 | **Owner:** Jake

---

## Phase 1: Immediate Production Risk Mitigation

### 1.1 Always-UNDER Gate (High Priority)
**Status:** Not started  
**Objective:** Disable or de-rank props that lose to always-UNDER on holdout.

- [ ] Query production holdout results; identify all props with win rate < 52%.
- [ ] Props to disable: PTS (50.5%), and others <52%.
- [ ] Update `nba_props.py` to filter based on always-UNDER win rate.
- [ ] Log when a prop is gated out; verify in `output/picks_history.csv`.
- [ ] Test: run_daily.bat should produce no picks for disabled props.

**Done When:** Live card excludes sub-52% props, with clear logging.

---

### 1.2 Production Holdout Win Rate Audit (High Priority)
**Status:** Partially done  
**Objective:** Confirm that each active prop beats always-UNDER on the true chronological holdout.

- [ ] Re-run holdout evaluation for all 6 props (PTS, TRB, AST, STL, BLK, TOV).
- [ ] Use the canonical chronological split: train on rows before T−1, holdout on T−1 to T+0 only.
- [ ] Compare each prop model output vs. always-UNDER decision on same rows.
- [ ] Record: prop, holdout sample size, model win rate, always-UNDER win rate, delta.
- [ ] Flag any prop that: (a) loses to always-UNDER, (b) has wide confidence intervals, (c) has <200 holdout rows.

**Done When:** Audit table saved to `output/production_holdout_audit.csv` with full transparency.

---

## Phase 2: Proof of Market Edge (High Priority)

### 2.1 Build Paired Forward Panel (Forward T−60)
**Status:** Not started  
**Objective:** Capture PrizePicks + sportsbook T−60 line pairs, run model, and track win rate.

**Scope:**
- Start: 2026-09-01 (first full day of season).
- Duration: 100+ days (into Q1 2027).
- Format: CSV with columns:
  ```
  game_date, player_id, player_name, prop, line, model_confidence, 
  model_selection (OVER/UNDER), prizepicks_line, sportsbook_no_vig_prob,
  sportsbook_selection, outcome, model_won, always_under_won
  ```

**Implementation:**
1. Add live scraper hook to `nba_props.py predict` to capture PrizePicks line at T−60.
2. Add sportsbook line scraper to fetch T−60 no-vig equivalent (Odds API).
3. Log both to daily CSV append; never overwrite.
4. Include `run_daily.bat` integration to ensure capture happens at correct time.

**Success Criteria:**
- Paired data for 100+ unique game-days.
- Model win rate vs. PrizePicks lines (not sportsbook).
- Win rate breakdown by prop, by month, by confidence bucket.
- Isolation of model contribution vs. always-UNDER bias.

**Done When:** Panel runs for 100+ days with <5% data loss; win rate analysis complete.

---

### 2.2 STL Availability Ablation (Medium Priority)
**Status:** Not started  
**Objective:** Understand why availability (AUC=0.886) didn't help STL prop selection despite high prediction quality.

**Approach:**
1. **Baseline:** STL prop regression using only traditional features (rolling stats, matchup).
2. **Treatment 1:** Add availability features (predicted minutes, injury probability).
3. **Treatment 2:** Add availability-derived usage redistribution.
4. Compare holdout win rates across all three.

**Candidates to Test:**
- Predicted player minutes as interaction term.
- Bench replacement usage bump (availability-driven usage shift).
- Injury flag as binary feature (player is out vs. playing).
- Bench depth signal (backup's recent performance).

**Expected Output:**
- Table: treatment, holdout sample size, holdout win rate, vs. always-UNDER.
- Recommendation: retain feature? (if win rate ≥ baseline + 1%), drop it?, or restructure calibration?

**Done When:** Ablation results saved to `output/stl_availability_ablation.csv`; decision documented in DECISIONS.md.

---

## Phase 3: Production Integrity (High Priority)

### 3.1 Stale Live-State Audit (High Priority)
**Status:** Not started  
**Objective:** Verify that every feature input to prop predictions is computed fresh each morning, not cached from yesterday.

**Checks:**
1. `nba_data.csv`: last box score date = today − 1 (yesterday). Not older.
2. `feature_pipeline.py`: all rolling stats (L5, L10, L20) are shifted backward by 1 game minimum.
3. Injury feed: injury status as of 8:55 AM today (before 9:15 AM pick generation).
4. Team defense ratings: computed from all games up to T−1, not including T+0 in-progress games.
5. Minutes model: using yesterday's roster, not today's (pre-game).

**For Each Feature:**
- [ ] Verify its source data (which CSV, which day).
- [ ] Confirm shift logic (if rolling, by how many games).
- [ ] Test: inject an artificial future game into nba_data.csv; confirm it does NOT appear in that day's projections.
- [ ] Log every data input's age in `output/pipeline_log.txt`.

**Done When:** All features pass freshness check; stale data is impossible (caught before prediction).

---

### 3.2 Remove Duplicate Training/Inference Paths (Medium Priority)
**Status:** Not started  
**Objective:** Consolidate all prop training logic into the canonical pipeline; eliminate legacy ad hoc scripts.

**Inventory:**
- [ ] List all Python files that train or predict props: `nba_props.py`, `train_advanced_models.py`, others?
- [ ] For each, document: what props does it handle? What features? What's the output format?
- [ ] Identify duplicates: two files training the same prop? Conflicting feature definitions?

**Consolidation:**
- [ ] Decide: canonical location for prop training (likely `nba_props.py` or a new `train_props.py`).
- [ ] Migrate all logic to canonical location.
- [ ] Delete legacy files; commit message lists what was removed.
- [ ] Update `run_daily.py` to call only the canonical trainer.
- [ ] Verify: same predictions for a sample game before/after consolidation.

**Done When:** One canonical prop-training script; old scripts deleted; predictions bit-identical.

---

### 3.3 Feature Path Cleanup (Low Priority)
**Status:** Not started  
**Objective:** Document the canonical feature set; remove legacy/unused feature columns.

**Steps:**
1. **Document canonical features:**
   - Run feature parity check across all 73,531 training rows.
   - Output: `feature_canonical_set.json` with column names, formulas, and data freshness.
   - Include 99 live defense features + 2 live-defense-only features.

2. **Remove unused columns:**
   - Grep for any feature columns that are no longer used in prop predictions.
   - Delete from all CSVs (nba_data.csv, etc.).
   - Commit message lists what was deleted.

3. **Update documentation:**
   - Update ARCHITECTURE.md to list only current features.
   - Delete any stale feature description sections.

**Done When:** `feature_canonical_set.json` exists and is audited; no dead code or unused columns in CSVs.

---

## Phase 4: Future Improvements (Offseason)

### 4.1 Defense Feature Rearchitecture (Low Priority)
**Status:** Not started  
**Rationale:** Defense features mostly failed selection. Either the feature engineering is weak, or the model architecture can't leverage them. This experiment will clarify.

**Options:**
1. **Test interaction terms:** defense * player_efficiency, defense * usage_rate.
2. **Test non-linear:** bucketed opponent defense (ELITE/GOOD/AVERAGE/BAD) instead of continuous.
3. **Test decay:** recent games (L3) vs. older games (L10); does opponent defense have recency value?
4. **Test ensemble:** separate small models for each opponent (Celtics defense, Suns defense, etc.).

**Gate:** Only pursue if Phase 2 (paired forward panel) shows <50% win rate on PrizePicks. Otherwise, it's premature optimization.

---

### 4.2 Live Injury & Roster Adapter (Low Priority)
**Status:** Not started  
**Rationale:** Availability model is good, but live injury/transaction feed is incomplete.

**Scope:**
- Integrate ESPN or NBA API injury reports (real-time).
- Detect roster movements (trades, call-ups, signings) same-day.
- Auto-update player rosters in memory before prop generation.

**Gate:** Defer until availability features actually get selected and improve props.

---

## Phase 5: Validation & Testing

### 5.1 Test Suite Expansion (Low Priority)
**Status:** 103 tests passing  
**Objective:** Reach 110+ tests; add coverage for:

- [ ] Always-UNDER gate logic (verify disabled props produce no picks).
- [ ] Stale live-state detection (verify injection attacks fail).
- [ ] Paired forward panel log format (CSV schema validation).
- [ ] Feature freshness (confirm all rolling stats are shift-lagged).

**Done When:** 110+ tests pass; all new logic has test coverage.

---

## Milestones

| Milestone | Target Date | Criteria |
|-----------|------------|----------|
| **Always-UNDER Gate Live** | 2026-09-01 | Props below 52% disabled in run_daily.bat output |
| **Paired Forward Panel Running** | 2026-09-01 | Daily captures of PrizePicks + sportsbook T−60 lines |
| **Production Holdout Audit** | 2026-09-01 | Audit table published; each active prop confirmed >52% |
| **Stale Live-State Audit Done** | 2026-09-08 | All features verified fresh; injection test passes |
| **STL Ablation Complete** | 2026-09-15 | Ablation results and feature retention decision documented |
| **Duplicate Path Cleanup Done** | 2026-09-22 | Canonical trainer confirmed; legacy files deleted |
| **Forward Test Conclusion** | 2026-12-15 | 100+ days of paired PrizePicks data; win rate and ROI reported |

---

## Open Decisions

1. **Which props should we disable immediately?** (See 1.1, Always-UNDER Gate)
   - Decision pending: holdout audit results (Task 1.2).
   - Candidate: PTS (50.5%), TOV (56.3%), BLK (33% on real lines, but bias-driven).

2. **Should defense features be retired or rearchitected?** (Offseason)
   - Decision pending: Phase 2 (paired forward panel) results.
   - If edge stays marginal, pivot to simpler model (rolling + blowout).

3. **What's the minimum edge threshold to keep a prop live?** 
   - Current: >50% vs. always-UNDER.
   - Better: >52% (2% margin for hold/volatility).
   - Proposal: encode in config, not hard-coded.

---

## References

- **STATUS.md** — System health and diagnosis.
- **ARCHITECTURE.md** — Feature pipeline and model design.
- **DECISIONS.md** — Historical decisions and rationale (to be created).
- **output/production_holdout_audit.csv** — Prop-level holdout results (to be created).
- **output/training_live_selected_feature_parity_summary.json** — Feature consistency check (exists).
- **Paired forward panel** — `output/paired_forward_panel_YYYYMM.csv` (to be created weekly).

---

## Contact

Owner: Jake  
Last Review: 2026-08-23  
Next Review: 2026-09-01 (Phase 1 status check)
