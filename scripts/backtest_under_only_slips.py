"""Replay the current under-only betslip policy against graded picks history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_props import backtest_under_only_slips


def main() -> None:
    detail_df, summary = backtest_under_only_slips()
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "under_only_slip_backtest_detail.csv"
    summary_path = output_dir / "under_only_slip_backtest_summary.json"

    if len(detail_df) > 0:
        detail_df.to_csv(detail_path, index=False)
        print(f"Detail written -> {detail_path}")
    else:
        print("No detail rows generated")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written -> {summary_path}")


if __name__ == "__main__":
    main()
