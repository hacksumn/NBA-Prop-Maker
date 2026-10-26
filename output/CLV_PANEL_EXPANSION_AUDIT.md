# CLV Panel Acquisition Audit

## Outcome

- Completed **528/528** evidence-safe snapshots; **528** contain odds and **0** are genuine empty-market observations.
- Used **42,129** Odds API credits; **13,436** remain.
- Preserved **314,162** paid line/price rows in the master store with **0** physical-key duplicates.
- Paired over/under prices: **314,054 / 314,162 (99.97%)**.
- Box-score join coverage at unique player/prop/date grain: **94.81%**.

## Integrity checks

- Paid rows with wrong local game date: **0**.
- Actual API snapshots after requested time: **0**.
- Actual API snapshots at/after tip: **0**.
- API snapshot lag: median **263s**, maximum **264s**.

## Coverage implications

- Historical props were returned for eight markets. `player_turnovers` was never returned and should not be retried blindly.
- Empty event snapshots are retained as market-withdrawal/unavailability labels, not treated as ingest failures.
- The panel supports decision/T-60/T-5 CLV where the archived decision timestamp exists and T-60/T-5 late movement for the complete 187-game window.
- Remaining credits should be spent only after feature materialization identifies a specific missing comparison.
