# CLV Panel Acquisition Audit

## Outcome

- Completed **474/474** evidence-safe snapshots; **469** contain odds and **5** are genuine empty-market observations.
- Used **35,880** Odds API credits; **64,120** remain.
- Preserved **253,725** paid line/price rows in the master store with **0** physical-key duplicates.
- Paired over/under prices: **253,680 / 253,725 (99.98%)**.
- Box-score join coverage at unique player/prop/date grain: **95.28%**.

## Integrity checks

- Paid rows with wrong local game date: **0**.
- Actual API snapshots after requested time: **0**.
- Actual API snapshots at/after tip: **0**.
- API snapshot lag: median **262s**, maximum **299s**.

## Coverage implications

- Historical props were returned for eight markets. `player_turnovers` was never returned and should not be retried blindly.
- Empty event snapshots are retained as market-withdrawal/unavailability labels, not treated as ingest failures.
- The panel supports decision/T-60/T-5 CLV where the archived decision timestamp exists and T-60/T-5 late movement for the complete 187-game window.
- Remaining credits should be spent only after feature materialization identifies a specific missing comparison.
