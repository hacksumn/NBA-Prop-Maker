"""Regression: season_games_played must reach pick rows and role scoring.

Before the fix, generate_predictions copied games_played but not
season_games_played into pred_df, so every pick row had it blank and
_apply_postseason_context silently used career games capped at 82.
"""
import contextlib
import inspect
import io
import unittest

import pandas as pd

import nba_props
from nba_props import _apply_postseason_context, filter_best_picks


QUALITY_POLICY = {
    "ast": {
        "allowed": True, "allow_under": True, "allow_over": True,
        "min_prob": 0.50, "min_residual_edge": 0.05, "min_edge": 0.05,
        "min_edge_under": 0.05, "min_edge_over": 0.05, "rmse": 1.8,
        "shrunk_acc": 0.54, "meta_gate_mode": "off", "priority_under_lane": False,
    }
}


def _pred_row(**extra):
    row = {
        "player": "Residual Under", "team": "OKC", "games_played": 203,
        "ast_line": 6.5, "ast_pred": 6.6, "ast_market_edge_pred": -0.71,
        "ast_p_over_cal": 0.41, "ast_projection_type": "standard",
        "ast_is_promo": False, "ast_sigma": 1.2,
    }
    row.update(extra)
    return row


class SeasonGamesPropagationTests(unittest.TestCase):
    def test_generate_predictions_copies_season_games_into_pred_rows(self):
        source = inspect.getsource(nba_props.generate_predictions)
        self.assertIn("'season_games_played': row.get('season_games_played'", source)

    def test_filter_best_picks_carries_season_games_played(self):
        pred_df = pd.DataFrame([_pred_row(season_games_played=11)])
        picks = filter_best_picks(pred_df, quality_policy=QUALITY_POLICY, max_picks=5)
        self.assertEqual(len(picks), 1)
        self.assertEqual(int(picks.loc[0, "season_games_played"]), 11)

    def test_role_score_uses_actual_season_games_not_career_fallback(self):
        base = {
            "player": "P", "prop": "AST", "direction": "UNDER",
            "games_played": 203, "mp_predicted": 30.0, "mp_l10": 30.0,
            "estimated_hit_rate": 62.0, "meta_prob": 55.0,
        }
        real = pd.DataFrame([dict(base, season_games_played=11)])
        missing = pd.DataFrame([dict(base, season_games_played=float("nan"))])

        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            real_out = _apply_postseason_context(real, slate_date="2026-11-13")
        self.assertNotIn("season_games_played missing", quiet.getvalue())

        loud = io.StringIO()
        with contextlib.redirect_stdout(loud):
            missing_out = _apply_postseason_context(missing, slate_date="2026-11-13")
        self.assertIn("season_games_played missing for 1/1", loud.getvalue())

        # 11 season games must score as a thinner role than the 82-game fallback.
        self.assertLess(
            float(real_out.loc[0, "postseason_role_score"]),
            float(missing_out.loc[0, "postseason_role_score"]),
        )


if __name__ == "__main__":
    unittest.main()
