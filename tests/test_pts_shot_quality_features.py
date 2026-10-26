import unittest

import pandas as pd

from feature_pipeline import FeaturePipeline
from train_advanced_models import _build_market_feature_block, _build_meta_feature_block


class PtsShotQualityFeatureTests(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        rows = []
        dates = pd.date_range("2026-01-01", periods=8, freq="2D")
        pts_vals = [18, 22, 16, 24, 21, 27, 19, 25]
        fga_vals = [14, 17, 13, 18, 16, 20, 15, 19]
        fta_vals = [4, 5, 3, 6, 5, 7, 4, 6]
        fg_pct_vals = [0.46, 0.51, 0.42, 0.56, 0.50, 0.58, 0.47, 0.54]
        for idx, game_date in enumerate(dates):
            rows.append(
                {
                    "player": "Shot Maker",
                    "team": "AAA",
                    "opp": f"OPP{idx % 3}",
                    "game_date": game_date,
                    "is_home": idx % 2,
                    "pts": pts_vals[idx],
                    "trb": 5 + (idx % 3),
                    "ast": 4 + (idx % 2),
                    "mp": 30 + idx,
                    "fga": fga_vals[idx],
                    "fta": fta_vals[idx],
                    "orb": 2 + (idx % 2),
                    "3pa": 5 + (idx % 2),
                    "tov": 2 + (idx % 2),
                    "stl": 1,
                    "blk": 0,
                    "fg_pct": fg_pct_vals[idx],
                    "3p_pct": 0.34 + (idx * 0.01),
                    "ft_pct": 0.78 + (idx * 0.01),
                }
            )
        return pd.DataFrame(rows)

    def test_feature_pipeline_builds_pts_shot_quality_columns(self):
        pipeline = FeaturePipeline(ewma_span=5, split_window=15, min_games=3)
        featured = pipeline.fit_transform(self._sample_frame())

        required_cols = [
            "fg_pct_l5",
            "fg_pct_l10",
            "fg_pct_l20",
            "fg_pct_luck",
            "ts_pct_l10",
            "ts_pct_trend",
            "points_per_fga_l10",
            "scoring_opps_l10",
            "shot_volume_quality_l10",
            "usage_fga_interaction_l10",
        ]
        for col in required_cols:
            self.assertIn(col, featured.columns)

        latest = featured.iloc[-1]
        self.assertTrue(pd.notna(latest["fg_pct_l10"]))
        self.assertTrue(pd.notna(latest["points_per_fga_l10"]))
        self.assertTrue(pd.notna(latest["shot_volume_quality_l10"]))

    def test_market_feature_block_exposes_pts_efficiency_inputs(self):
        pipeline = FeaturePipeline(ewma_span=5, split_window=15, min_games=3)
        featured = pipeline.fit_transform(self._sample_frame())
        candidate_rows = featured.tail(2).reset_index(drop=True)
        lines = pd.Series([21.5, 22.0], index=candidate_rows.index)

        market = _build_market_feature_block(candidate_rows, "pts", lines)

        for col in [
            "market_fg_pct_l10",
            "market_fg_pct_delta",
            "market_ts_pct_l10",
            "market_points_per_fga_l10",
            "market_scoring_opps_l10",
            "market_shot_volume_quality_l10",
            "market_usage_fga_interaction_l10",
        ]:
            self.assertIn(col, market.columns)
            self.assertTrue(market[col].notna().all())

    def test_meta_feature_block_uses_probability_and_sigma_inputs(self):
        pipeline = FeaturePipeline(ewma_span=5, split_window=15, min_games=3)
        featured = pipeline.fit_transform(self._sample_frame())
        candidate_rows = featured.tail(2).reset_index(drop=True)
        lines = pd.Series([21.5, 22.0], index=candidate_rows.index)

        market = _build_market_feature_block(candidate_rows, "pts", lines)
        meta = _build_meta_feature_block(
            market,
            raw_edge=pd.Series([-1.1, -0.8], index=market.index),
            market_edge_pred=pd.Series([-1.4, -0.5], index=market.index),
            p_over_raw=pd.Series([0.41, 0.46], index=market.index),
            sigma=pd.Series([3.2, 2.7], index=market.index),
        )

        for col in [
            "signal_combined_edge",
            "signal_abs_combined_edge",
            "signal_market_support_ratio",
            "signal_direction_prob",
            "signal_direction_prob_margin",
            "signal_sigma",
            "signal_edge_to_sigma",
            "signal_market_edge_to_sigma",
        ]:
            self.assertIn(col, meta.columns)
            self.assertTrue(meta[col].notna().all())

        self.assertLess(meta.loc[0, "signal_combined_edge"], 0.0)
        self.assertGreater(meta.loc[0, "signal_direction_prob"], 0.5)


if __name__ == "__main__":
    unittest.main()
