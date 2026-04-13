import unittest
from unittest.mock import patch

from injury_feed import fetch_injury_data


class InjuryFeedMergeTests(unittest.TestCase):
    def test_official_status_overrides_espn_and_pending_flag_survives(self):
        official = {
            "LAC": {
                "out": ["kawhi leonard"],
                "doubtful": [],
                "questionable": [],
                "probable": [],
                "day_to_day": [],
                "available": [],
                "status_map": {
                    "kawhi leonard": {
                        "bucket": "out",
                        "status": "Out",
                        "comment": "NBA official injury report",
                        "source": "nba_official",
                    }
                },
                "not_yet_submitted": False,
                "report_label": "1:00 p.m ET report",
                "report_url": "https://official.nba.com/mock.pdf",
            },
            "PHX": {
                "out": [],
                "doubtful": [],
                "questionable": [],
                "probable": [],
                "day_to_day": [],
                "available": [],
                "status_map": {},
                "not_yet_submitted": True,
                "report_label": "1:00 p.m ET report",
                "report_url": "https://official.nba.com/mock.pdf",
            },
        }
        espn = {
            "LAC": {
                "out": [],
                "doubtful": [],
                "questionable": ["norman powell"],
                "probable": ["kawhi leonard"],
                "day_to_day": [],
                "available": [],
                "status_map": {
                    "kawhi leonard": {
                        "bucket": "probable",
                        "status": "Probable",
                        "comment": "ESPN status",
                        "source": "espn",
                    },
                    "norman powell": {
                        "bucket": "questionable",
                        "status": "Questionable",
                        "comment": "ESPN status",
                        "source": "espn",
                    },
                },
                "not_yet_submitted": False,
                "report_label": "",
                "report_url": "",
            }
        }

        with patch("injury_feed.fetch_official_nba_injury_data", return_value=official), patch(
            "injury_feed._fetch_espn_injury_data", return_value=espn
        ):
            merged = fetch_injury_data()

        self.assertIn("LAC", merged)
        self.assertEqual(merged["LAC"]["status_map"]["kawhi leonard"]["bucket"], "out")
        self.assertEqual(merged["LAC"]["status_map"]["kawhi leonard"]["source"], "nba_official")
        self.assertEqual(merged["LAC"]["status_map"]["norman powell"]["bucket"], "questionable")
        self.assertEqual(merged["LAC"]["status_map"]["norman powell"]["source"], "espn")
        self.assertTrue(merged["PHX"]["not_yet_submitted"])


if __name__ == "__main__":
    unittest.main()
