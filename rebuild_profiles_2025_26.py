"""
ONE-OFF: rebuild data/player_profiles.csv from the 2025-26 season.

Why: the 2026-10-25 test run overwrote player_profiles.csv with an empty file (no player
has the minimum games yet this season). run_daily.py now refuses to do that again, but the
emptied file needs to be rebuilt once. This produces the same kind of file the pipeline had
all offseason (final 2025-26 per-game profiles). Once players reach the minimum games in
2026-27, the normal daily run replaces it automatically.

Usage (from the Fresh_Start_NBA folder):
    python rebuild_profiles_2025_26.py
It only writes if the current file has no data rows (add --force to overwrite anyway).
"""
import sys
from pathlib import Path

import pandas as pd

from usage_injury_model import build_player_profiles

HERE = Path(__file__).parent.resolve()
PATH = HERE / "data" / "player_profiles.csv"
SEASON = "2025-26"
MIN_EXPECTED_ROWS = 300  # last season's file had ~500 rows; fewer means something is wrong


def current_rows() -> int:
    try:
        return len(pd.read_csv(PATH))
    except Exception:
        return 0


def main():
    force = "--force" in sys.argv
    n = current_rows()
    if n > 0 and not force:
        print(f"{PATH.name} already has {n} rows - nothing to do (use --force to overwrite).")
        return
    df = build_player_profiles(season=SEASON)
    if len(df) < MIN_EXPECTED_ROWS:
        sys.exit(f"Only {len(df)} players returned for {SEASON}; not writing. Try again in a minute.")
    tmp = PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(PATH)
    print(f"Wrote {len(df)} players ({df['team_abbr'].nunique()} teams) to {PATH}")


if __name__ == "__main__":
    main()
