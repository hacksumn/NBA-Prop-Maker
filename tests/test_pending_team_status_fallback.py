import unittest

import pandas as pd

from run_daily import _apply_pending_team_status_fallback


class PendingTeamStatusFallbackTests(unittest.TestCase):
    def test_pending_team_without_player_entry_gets_explicit_pending_status(self):
        frame = pd.DataFrame(
            [
                {
                    "player": "Kawhi Leonard",
                    "live_team_status_pending": True,
                    "live_injury_bucket": pd.NA,
                    "live_injury_status": pd.NA,
                    "live_injury_comment": pd.NA,
                    "live_injury_source": pd.NA,
                },
                {
                    "player": "Norman Powell",
                    "live_team_status_pending": True,
                    "live_injury_bucket": "questionable",
                    "live_injury_status": "Questionable",
                    "live_injury_comment": "ESPN listing",
                    "live_injury_source": "espn",
                },
                {
                    "player": "LeBron James",
                    "live_team_status_pending": False,
                    "live_injury_bucket": "",
                    "live_injury_status": "",
                    "live_injury_comment": "",
                    "live_injury_source": "",
                },
            ]
        )

        updated, marked_count = _apply_pending_team_status_fallback(frame)

        self.assertEqual(marked_count, 1)
        self.assertEqual(updated.loc[0, "live_injury_bucket"], "team_pending")
        self.assertEqual(updated.loc[0, "live_injury_status"], "Team Status Pending")
        self.assertEqual(updated.loc[0, "live_injury_source"], "team_pending")
        self.assertIn("Official NBA injury report pending", updated.loc[0, "live_injury_comment"])
        self.assertEqual(updated.loc[1, "live_injury_bucket"], "questionable")
        self.assertEqual(updated.loc[1, "live_injury_source"], "espn")
        self.assertEqual(updated.loc[2, "live_injury_bucket"], "")

    def test_no_unresolved_pending_rows_returns_zero(self):
        frame = pd.DataFrame(
            [
                {
                    "player": "Victor Wembanyama",
                    "live_team_status_pending": True,
                    "live_injury_bucket": "probable",
                    "live_injury_status": "Probable",
                    "live_injury_comment": "NBA official injury report",
                    "live_injury_source": "nba_official",
                },
                {
                    "player": "Shai Gilgeous-Alexander",
                    "live_team_status_pending": False,
                    "live_injury_bucket": "",
                    "live_injury_status": "",
                    "live_injury_comment": "",
                    "live_injury_source": "",
                },
            ]
        )

        updated, marked_count = _apply_pending_team_status_fallback(frame)

        self.assertEqual(marked_count, 0)
        self.assertEqual(updated.to_dict("records"), frame.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
