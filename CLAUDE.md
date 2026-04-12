# CLAUDE.md

## Mission
This repository builds and improves an NBA prediction system with strict anti-leakage discipline.

Primary goals, in order:
1. Prevent leakage and preserve causal integrity.
2. Keep training/evaluation reproducible.
3. Improve true forward performance, not cosmetic backtest metrics.
4. Preserve project continuity across long Claude Code sessions.

## Canonical Project Memory
Before doing meaningful work, read these files first:
- README.md
- ARCHITECTURE.md
- STATUS.md
- TASKS.md
- DECISIONS.md

Treat those files as canonical memory.
Do not rely on old chat context when they disagree with prior conversation history.

## Non-Negotiable Rules
- Never introduce feature leakage, target leakage, or time leakage.
- Never use future information in features.
- Never silently change schemas, file names, interfaces, or column names.
- Never claim model improvement without before/after evidence.
- Never optimize for backtest aesthetics over deployable forward realism.
- Never mark work complete without verification.
- Never delete decision history from DECISIONS.md.

## Required Workflow
For any non-trivial task:
1. Read the canonical memory files listed above.
2. Summarize current state in bullets.
3. Propose a short plan.
4. Execute in small, reversible steps.
5. Verify with commands/tests/checks.
6. Update STATUS.md, TASKS.md, and DECISIONS.md when relevant.

## Planning Rules
Use plan-first behavior when:
- changing feature engineering
- changing data schemas
- changing training/validation/test split logic
- changing model architecture
- changing odds merge logic
- changing grading logic
- doing multi-file refactors

A good plan must include:
- objective
- affected files
- risks
- validation
- definition of done

## Data Integrity Rules
- Every feature must be computable from information available strictly before prediction time.
- Any rolling stat must be shifted appropriately.
- Any join using odds, injury, team, player, or opponent data must preserve time ordering.
- Chronological evaluation is the default.
- Random split results are non-canonical unless explicitly labeled exploratory.
- Every training/evaluation run should record dataset snapshot, config, commit hash, timestamp, and output artifact location.

## Evaluation Rules
Always distinguish:
- training metrics
- validation metrics
- holdout / forward metrics
- business-facing metrics

Do not claim a new model is better unless the evaluation is time-respecting and reproducible.

## Coding Rules
- Prefer explicit code over clever abstractions.
- Keep functions focused.
- Add docstrings to public functions/classes.
- Add comments only when the reasoning is non-obvious.
- Fail loudly on invalid inputs.
- Preserve deterministic behavior where practical.
- Avoid hidden state and ad hoc manual file edits.

## Verification Rules
After meaningful changes, report:
1. what changed
2. why it changed
3. exact commands/checks run
4. pass/fail result
5. remaining risks or open questions
6. which project-memory files were updated

## Anti-Drift Rules
When the session gets long or fuzzy:
- re-read STATUS.md and DECISIONS.md
- re-ground on current scope
- prefer a fresh session over continuing with vague assumptions
- use the compact reinjection hook output as a reminder, not as the only truth
- treat repo files as source of truth

## Definition of Done
Work is only done when:
- implementation is complete
- validation has run
- relevant project-memory docs are updated
- next-step risks or blockers are documented

---

# NBA Project Reference

*Project-specific context that must survive session compaction.
Full details live in ARCHITECTURE.md, STATUS.md, TASKS.md, DECISIONS.md.*

## What This Is
Fully automated NBA player prop prediction system owned by Jake.
Runs at ~9 am daily via Windows Task Scheduler.
**Real money is bet based on this system's output every day.**

## The Daily Pipeline (run_daily.py)
Steps run in order; each must succeed before the next begins.

| Step | What It Does |
|------|--------------|
| 1 | Fetch last night's box scores → `data/nba_data.csv` |
| 1.5 | Grade picks in `output/picks_history.csv` (has live NBA API fallback) |
| 1.6 | Grade betslips in `output/betslips_history.csv` |
| 2 | Scrape PrizePicks lines → `data/historical_lines.csv` |
| 3–7 | Run Layers 1–5 (PBP, PPP, Usage/Injury, Luck, Blowout) |
| 8 | Merge all layers → `data/player_projections_today.csv` |
| 9 | `nba_props.py predict` → picks, betslips, dashboard |

## Key Files — Data Contract (never change column structure silently)
- `data/nba_data.csv` — master box score dataset
- `data/historical_lines.csv` — all PrizePicks lines; now includes `source` column
- `output/picks_history.csv` — all historical picks; now includes `line_source` column
- `output/betslips_history.csv` — all betslips with grading

## Critical AI Rules for This Project
1. **Debug before delivering** — verify syntax and logic before presenting code.
2. **Read the actual file first** — never assume column names or function signatures.
3. **Patch surgically** — change only what needs to change; no "cleanup" scope creep.
4. **Never break the morning run** — `run_morning.bat` runs automatically; test risky changes in isolation.
5. **Money is on the line** — flag any change that could affect pick rankings or betslip construction.
6. **Log everything** — every pipeline step must explain why it produced zero output, not just report null.

## Current Performance (April 2026)
Individual pick win rate: 53–57% | Best: AST (72%), TRB (62%), PA (57%)
PTS model disabled (49%). STL/BLK excluded. 2-pick slips most efficient format.

## Line Source Tracking
`picks_history.csv` now includes `line_source` ('prizepicks' or 'odds_api').
Picks with `line_source='odds_api'` were generated when PrizePicks was blocked —
verify those lines on PrizePicks before betting. The pipeline logs a `!!!` warning block
when this occurs.

## Volume Fill Picks
When the primary filter produces < 10 candidates, a relaxed second pass adds picks
tagged `pick_source='volume_fill'` (confidence capped at 63%). These fill the card
but sort below primary picks.