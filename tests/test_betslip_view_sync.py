import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_daily import _sync_betslip_views_from_history


class BetslipViewSyncTests(unittest.TestCase):
    def test_sync_updates_daily_archive_and_empty_latest(self):
        slips = pd.DataFrame(
            [
                {"game_date": "2026-04-15", "slip_type": "2-pick Power Play", "result": "WIN", "actual_payout": "40.0"},
                {"game_date": "2026-04-15", "slip_type": "3-pick Power Play", "result": "LOSS", "actual_payout": "0.0"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_root = root / "archive"
            latest_path = root / "betslips_latest.csv"
            pd.DataFrame(columns=slips.columns).to_csv(latest_path, index=False)

            synced_dates, latest_synced = _sync_betslip_views_from_history(
                slips,
                latest_path=latest_path,
                archive_root=archive_root,
            )

            self.assertEqual(synced_dates, ["2026-04-15"])
            self.assertTrue(latest_synced)

            archive_path = archive_root / "2026-04-15" / "betslips_2026-04-15.csv"
            self.assertTrue(archive_path.exists())

            archived = pd.read_csv(archive_path, dtype=str)
            latest = pd.read_csv(latest_path, dtype=str)
            self.assertEqual(archived["result"].tolist(), ["WIN", "LOSS"])
            self.assertEqual(latest["game_date"].tolist(), ["2026-04-15", "2026-04-15"])

    def test_sync_does_not_overwrite_live_latest_for_another_date(self):
        slips = pd.DataFrame(
            [
                {"game_date": "2026-04-15", "slip_type": "2-pick Power Play", "result": "WIN", "actual_payout": "40.0"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_root = root / "archive"
            latest_path = root / "betslips_latest.csv"
            pd.DataFrame(
                [{"game_date": "2026-04-16", "slip_type": "2-pick Power Play", "result": "", "actual_payout": ""}]
            ).to_csv(latest_path, index=False)

            synced_dates, latest_synced = _sync_betslip_views_from_history(
                slips,
                latest_path=latest_path,
                archive_root=archive_root,
            )

            self.assertEqual(synced_dates, ["2026-04-15"])
            self.assertFalse(latest_synced)

            latest = pd.read_csv(latest_path, dtype=str)
            self.assertEqual(latest["game_date"].tolist(), ["2026-04-16"])


if __name__ == "__main__":
    unittest.main()
