import unittest

from scripts.audit_naive_under_baseline import BOX_SCORE_COLUMNS, PROP_ALIASES, PROP_FORMULAS


class NaiveUnderAuditContractTests(unittest.TestCase):
    def test_all_nine_odds_markets_have_box_score_contracts(self):
        expected = {
            "player_points": "PTS",
            "player_rebounds": "TRB",
            "player_assists": "AST",
            "player_steals": "STL",
            "player_blocks": "BLK",
            "player_turnovers": "TOV",
            "player_points_assists": "PA",
            "player_points_rebounds": "PR",
            "player_points_rebounds_assists": "PRA",
        }
        for market, code in expected.items():
            self.assertEqual(PROP_ALIASES.get(market), code)
            self.assertIn(code, PROP_FORMULAS)

    def test_turnovers_are_loaded_from_box_scores(self):
        self.assertIn("tov", BOX_SCORE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
