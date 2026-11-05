"""Restore picks/betslips that later same-day reruns deleted from the history files (DEC-070).

Before DEC-070, every predict run replaced all of a slate's rows in
output/picks_history.csv and output/betslips_history.csv with only that run's
output, so picks and slips published by an earlier run (often the morning card
that was bet) disappeared. Every run also saved a timestamped copy under
output/archive/<date>/{picks,betslips}_<date>_<HHMMSS>.csv; those copies were
never touched, so they are the recovery source.

The script is additive only: it walks each slate's run copies in time order and
appends rows whose key is not already in the history. Existing rows are never
modified. Restored rows carry history_source=archive_recovered and published_at
from the run copy's timestamp, so they can be filtered or removed by the
operator. Default is a dry run; pass --apply to write.

Usage:
    python scripts/recover_lost_history_rows.py            # dry run
    python scripts/recover_lost_history_rows.py --apply
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nba_props import (  # noqa: E402
    PICKS_HISTORY_KEY,
    _append_only_history,
    _betslip_history_key,
    _history_key_series,
)

ARCHIVE = ROOT / "output" / "archive"
RUN_FILE = re.compile(r"^(?P<stem>picks|betslips)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hms>\d{6})\.csv$")


def _run_copies(stem: str):
    """Yield (slate_date, published_at, path) for timestamped run copies, oldest first."""
    found = []
    for path in ARCHIVE.glob(f"*/{stem}_*.csv"):
        match = RUN_FILE.match(path.name)
        if not match or path.parent.name != match["date"]:
            continue
        hms = match["hms"]
        published_at = f"{match['date']} {hms[:2]}:{hms[2:4]}:{hms[4:]}"
        found.append((match["date"], published_at, path))
    return sorted(found, key=lambda item: (item[0], item[2].name))


def _recover(stem: str, history_path: Path, key, apply: bool) -> int:
    existing = (
        pd.read_csv(history_path, dtype=str, keep_default_na=False, low_memory=False)
        if history_path.exists() else pd.DataFrame()
    )
    known = set(_history_key_series(existing, key)) if len(existing) else set()
    pending = []
    for slate_date, published_at, path in _run_copies(stem):
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame["game_date"] = slate_date  # run copies are filed under their slate date
        frame["published_at"] = published_at
        frame["history_source"] = "archive_recovered"
        frame["recovered_from"] = str(path.relative_to(ROOT)).replace("\\", "/")
        keys = _history_key_series(frame, key)
        fresh = frame[~keys.isin(known) & ~keys.duplicated()]
        if len(fresh):
            known.update(_history_key_series(fresh, key))
            pending.append(fresh)
    if not pending:
        print(f"  {history_path.name}: nothing to recover")
        return 0
    recovered = pd.concat(pending, ignore_index=True)
    by_date = recovered.groupby("game_date").size().to_dict()
    print(f"  {history_path.name}: {len(recovered)} row(s) recoverable across {len(by_date)} slate(s)")
    for date, n in sorted(by_date.items()):
        print(f"    {date}: {n}")
    if apply:
        before = len(existing)
        combined, added, already = _append_only_history(history_path, recovered, key)
        print(f"    applied: +{added} rows ({before:,} -> {len(combined):,}); {already} already present")
        return added
    return len(recovered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the recovered rows (default: dry run)")
    args = parser.parse_args()
    print("APPLY" if args.apply else "DRY RUN (pass --apply to write)")
    _recover("picks", ROOT / "output" / "picks_history.csv", PICKS_HISTORY_KEY, args.apply)
    # The leading "game_date" column inserted by save_picks is restored above;
    # betslip run copies already carry game_date and Pick_N_* leg columns.
    _recover("betslips", ROOT / "output" / "betslips_history.csv", _betslip_history_key, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
