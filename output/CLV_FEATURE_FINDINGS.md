# CLV Feature Findings

## What the paid panel resolves

The PrizePicks-minus-consensus gap survives the larger panel. At T-60, UNDER win
rate rises from **49.5%** when PrizePicks is
below consensus to **67.1%** when it is at
least one unit higher (`n=10,943`). This expands the
original 973-key diagnostic to a much larger, price-bearing panel.

The selected card has closing consensus coverage on **159**
rows. Direction-adjusted line advantage versus sportsbook consensus averages
**+0.671** units and is positive on
**71.7%**. This is primarily
a cross-market PrizePicks shading advantage—not proof the model predicts market
movement.

## What remains unresolved

The average sportsbook closing no-vig probability on the selected side is only
**50.2%**, and exceeds 50% on
**47.8%** of covered
picks. Sportsbook prices therefore do not independently confirm the selected
direction.

Archived PrizePicks references are not true closes: their median age at T-5 is
**534 minutes**, with only
**3** observations captured in
the final hour. Odds API credits cannot repair this because PrizePicks is not an
Odds API bookmaker. Next-season collection must snapshot the PrizePicks board
near T-60 and T-5 if same-book CLV is required.

Late sportsbook line movement is weak rather than monotonic: UNDER rates are
51.5% after a downward move,
54.0% unchanged, and
53.6% after an upward move.

The larger sportsbook sample corrects DEC-040's market-wide claim. At 30+
minutes, T-5 consensus UNDER is only
**51.9%**
for STL (`n=621`)
and **58.7%**
for BLK (`n=965`).
STL is efficiently priced at the book; BLK retains only a modest skew. The much
higher PrizePicks UNDER rates are therefore primarily book-specific shading.

## Productive next model work

1. Add the pre-tip PP-minus-consensus gap, book dispersion, and no-vig consensus
   probability as source-specific candidate features.
2. Evaluate them chronologically against same-date random UNDER and always-UNDER
   controls; do not tune on this complete live window and then quote it as holdout.
3. Keep STL/BLK, AST/TRB, and composite regimes separate.
4. Begin scheduled PrizePicks T-60/T-5 capture next season; do not spend more
   Odds API credits trying to manufacture a missing PrizePicks close.
