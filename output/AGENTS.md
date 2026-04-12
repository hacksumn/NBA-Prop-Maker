# AGENTS.md

## Mission
This repository builds and improves an NBA prediction system with strict anti-leakage discipline.
Primary goals, in order:
1. Preserve data integrity and prevent leakage.
2. Produce reproducible training/evaluation runs.
3. Improve real out-of-sample performance, not just backtest metrics.
4. Keep the codebase understandable and easy to resume after long sessions.

## Source of Truth
Before making any changes, read these files first:
- README.md
- ARCHITECTURE.md
- STATUS.md
- TASKS.md
- DECISIONS.md

If chat instructions conflict with those files, pause and explicitly call out the conflict.

## Non-Negotiable Constraints
- Never introduce feature leakage, target leakage, or time leakage.
- Never use future information in training features.
- Never silently change data schemas.
- Never silently rename files, columns, or interfaces relied on elsewhere.
- Never optimize for backtest appearance at the expense of forward realism.
- Never assume a missing requirement; document the assumption or ask.
- Never delete or overwrite decision history in DECISIONS.md.
- Never mark work complete without verification.

## Working Style
For non-trivial tasks:
1. Read repo context files.
2. Summarize current state in bullets.
3. Produce a short plan.
4. Execute in small, reversible steps.
5. Run verification.
6. Update STATUS.md, TASKS.md, and DECISIONS.md if relevant.

## Planning Rules
Use plan-first behavior when:
- changing model architecture
- changing feature engineering
- changing data pipeline behavior
- altering training/evaluation split logic
- touching bankroll, betting, or grading logic
- making multi-file refactors

A good plan should include:
- goal
- files likely affected
- risks
- validation steps
- definition of done

## Coding Standards
- Prefer simple, explicit code over clever abstractions.
- Keep functions focused and single-purpose.
- Add docstrings to public functions/classes.
- Add comments only where reasoning is non-obvious.
- Fail loudly on invalid inputs.
- Preserve deterministic behavior where possible.
- Prefer typed interfaces where practical.

## Data and Modeling Rules
- All time-based features must be computable from information available strictly before prediction time.
- Training, validation, and test splits must respect chronology.
- Any rolling stats must be shifted appropriately.
- Any opponent/team/player features must be checked for leakage.
- Record the exact dataset version or data snapshot used.
- Record all feature additions/removals in DECISIONS.md when they affect model behavior.

## Evaluation Rules
- Prefer walk-forward or other time-respecting evaluation.
- Report both predictive metrics and business-facing metrics where applicable.
- Distinguish clearly between:
  - training performance
  - validation performance
  - true holdout / forward performance
- Do not claim improvement without before/after evidence.

## Verification Requirements
For each meaningful code change:
- run the relevant tests, lint, or validation scripts
- report exactly what was run
- report pass/fail
- report unresolved issues

If tests do not exist, add the smallest useful verification possible.

## File Update Rules
Update STATUS.md when:
- the active objective changes
- something starts/stops working
- a blocker appears/disappears
- a run completes
- a bug is diagnosed

Update TASKS.md when:
- priorities change
- new work is discovered
- tasks are finished
- work is blocked

Update DECISIONS.md when:
- choosing one design over another
- changing model assumptions
- changing schema/contracts
- accepting a tradeoff
- rejecting an alternative after review

## Communication Format
When responding after work:
1. What changed
2. Why it changed
3. What was verified
4. Risks / open questions
5. Which repo docs were updated

## Anti-Drift Rules
When context gets long:
- re-read STATUS.md and DECISIONS.md before continuing
- do not rely on memory of earlier chat messages
- treat repo files as canonical state
- prefer a fresh task boundary over continuing with fuzzy assumptions

## Definition of Done
Work is only done when:
- implementation is complete
- relevant verification has run
- repo documentation is updated
- next step or follow-up risk is documented