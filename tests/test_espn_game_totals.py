"""ESPN linked teams, honest attachment coverage, and disabled environment effects."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import nba_props as nba


CORE = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba'


def response(body, error=None):
    out = Mock()
    out.json.return_value = body
    out.raise_for_status.side_effect = error
    return out


class FetchGameTotalsTests(unittest.TestCase):
    def fetch(self, home, away, odds=None, team_error=False):
        odds = odds or {'overUnder': 230.5, 'spread': -4.5,
                        'homeTeamOdds': {'favorite': False, 'moneyLine': 150},
                        'awayTeamOdds': {'moneyLine': -170}}
        routes = {
            f'{CORE}/events?dates=20261120&limit=20': {'items': [{'$ref': f'{CORE}/events/1'}]},
            f'{CORE}/events/1': {'id': '1', 'competitions': [{'id': '2', 'competitors': [
                {'homeAway': 'home', 'team': home}, {'homeAway': 'away', 'team': away}]}]},
            f'{CORE}/teams/1': {'abbreviation': 'GS'},
            f'{CORE}/teams/2': {'abbreviation': 'NY'},
            f'{CORE}/events/1/competitions/2/odds': {'items': [odds]},
            f'{CORE}/odds/100': {'overUnder': 241.5, 'spread': -3.5},
        }

        def get(url, **kwargs):
            if team_error and url == f'{CORE}/teams/1':
                raise nba.requests.Timeout('team reference timed out')
            return response(routes[url])

        with patch.object(nba.requests, 'get', side_effect=get) as fetch, patch.object(nba._time, 'sleep'):
            result = nba.fetch_game_totals('2026-11-20')
        return result, fetch

    def test_linked_teams_resolve_and_normalize_and_home_spread_has_correct_sign(self):
        result, fetch = self.fetch({'$ref': f'{CORE}/teams/1'.replace('https:', 'http:')},
                                   {'$ref': f'{CORE}/teams/2'})
        self.assertEqual(result.home_team.tolist(), ['GSW'])
        self.assertEqual(result.away_team.tolist(), ['NYK'])
        self.assertEqual(result.spread.tolist(), [4.5])
        self.assertEqual(result.total.tolist(), [230.5])
        self.assertTrue(all(call.args[0].startswith('https:') for call in fetch.call_args_list))
        self.assertEqual(result.home_ml.tolist(), [150])

    def test_embedded_teams_do_not_need_reference_calls(self):
        result, fetch = self.fetch({'abbreviation': 'BOS'}, {'abbreviation': 'SA'})
        self.assertEqual(result.away_team.tolist(), ['SAS'])
        self.assertEqual(fetch.call_count, 3)

    def test_explicit_home_point_spread_takes_precedence(self):
        result, _ = self.fetch({'abbreviation': 'LAL'}, {'abbreviation': 'BOS'}, odds={
            'overUnder': 220, 'spread': -6.5,
            'homeTeamOdds': {'current': {'pointSpread': {'american': '+6.5'}}}})
        self.assertEqual(result.spread.tolist(), [6.5])

    def test_odds_can_also_be_a_reference(self):
        result, _ = self.fetch({'abbreviation': 'LAL'}, {'abbreviation': 'BOS'},
                               odds={'$ref': f'{CORE}/odds/100'})
        self.assertEqual(result.total.tolist(), [241.5])

    def test_unknown_or_failed_teams_never_emit_blank_matchups(self):
        for home, error in [({'abbreviation': 'UNKNOWN'}, False),
                            ({'$ref': f'{CORE}/teams/1'}, True)]:
            with self.subTest(home=home), contextlib.redirect_stdout(io.StringIO()) as log:
                result, _ = self.fetch(home, {'abbreviation': 'BOS'}, team_error=error)
                self.assertTrue(result.empty)
                self.assertIn('skipped', log.getvalue())

    def test_http_error_is_reported_and_returns_empty(self):
        with patch.object(nba.requests, 'get', return_value=response({}, nba.requests.HTTPError('503'))), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertTrue(nba.fetch_game_totals('2026-11-20').empty)
        self.assertIn('Game totals unavailable', log.getvalue())


class GameTotalContextTests(unittest.TestCase):
    def test_context_matches_teams_without_changing_any_score(self):
        predictions = pd.DataFrame({'team': ['LAL', 'LAL', 'BOS', 'GSW'],
                                    'pts_pred': [20.123] * 4, 'ast_pred': [5.678] * 4,
                                    'pra_pred': [30.456] * 4, 'pa_pred': [25.89] * 4,
                                    'pts_p_over_cal': [.55] * 4, 'pts_meta_prob': [.6] * 4})
        totals = pd.DataFrame([{'home_team': 'LAL', 'away_team': 'BOS', 'total': 260, 'spread': 4.5}])
        with contextlib.redirect_stdout(io.StringIO()) as log:
            out = nba._attach_game_total_context(predictions, totals)
        pd.testing.assert_frame_equal(predictions, out[predictions.columns])
        self.assertEqual(out.game_total.iloc[:3].tolist(), [260] * 3)
        self.assertTrue(pd.isna(out.game_total.iloc[3]))
        self.assertEqual(out.team_spread.tolist(), [4.5, 4.5, -4.5, 0])
        self.assertTrue(out.game_total_scale.eq(1).all())
        self.assertIn('matched 3/4 player rows across 1 games', log.getvalue())
        self.assertIn('scaling OFF', log.getvalue())
        self.assertNotIn('Game totals applied', log.getvalue())

    def test_empty_invalid_and_unmatched_totals_report_zero_coverage(self):
        predictions = pd.DataFrame({'team': ['GSW'], 'pts_pred': [20.123]})
        for rows in [[], [{'home_team': '', 'away_team': '', 'total': 230}],
                     [{'home_team': 'LAL', 'away_team': 'BOS', 'total': 'bad'}],
                     [{'home_team': 'LAL', 'away_team': 'BOS', 'total': float('inf')}],
                     [{'home_team': 'LAL', 'away_team': 'BOS', 'total': None}],
                     [{'home_team': 'LAL', 'away_team': 'BOS', 'total': 225}]]:
            with self.subTest(rows=rows), contextlib.redirect_stdout(io.StringIO()) as log:
                out = nba._attach_game_total_context(predictions, pd.DataFrame(rows))
            self.assertTrue(out.game_total.isna().all())
            self.assertTrue(out.game_total_scale.eq(1).all())
            self.assertIn('matched 0/1', log.getvalue())
            self.assertIn('no totals attached', log.getvalue())

    def test_generation_keeps_scoring_inputs_neutral_and_fetches_the_active_slate(self):
        latest = pd.DataFrame([{'player': 'Test', 'team': 'LAL', 'games_played': 50,
                                'game_date': '2026-11-19', 'mp_l10': 30}])
        totals = pd.DataFrame([{'home_team': 'LAL', 'away_team': 'BOS', 'total': 260, 'spread': -5}])

        def score(pred, *args):
            # A repaired total must not change the inputs seen by edge/meta models.
            self.assertTrue(pred.game_total.isna().all())
            self.assertTrue(pred.team_spread.eq(0).all())
            self.assertTrue(pred.game_total_scale.eq(1).all())
            return pred.assign(pts_meta_prob=.57, pts_p_over_cal=.48)

        with TemporaryDirectory() as tmp, patch.dict(nba.CONFIG, {'models_dir': Path(tmp), 'data_dir': Path(tmp)}), \
                patch.object(nba, '_load_advanced_models', return_value=({'pts': object()}, [], {}, {}, {}, {}, {})), \
                patch.object(nba, '_load_probability_sidecars', return_value={}), \
                patch.object(nba, '_load_probability_calibrators', return_value={}), \
                patch.object(nba, '_prepare_latest_prediction_frame', return_value=latest), \
                patch.object(nba, '_predict', return_value=np.array([21.2])), \
                patch.object(nba, '_attach_market_model_scores', side_effect=score), \
                patch.object(nba, 'fetch_game_totals', return_value=totals) as fetch, \
                contextlib.redirect_stdout(io.StringIO()):
            out = nba.generate_predictions(latest, apply_injury_boosts=False,
                                           apply_fp_blend=False, slate_date='2026-11-20')
            fetch.assert_called_once_with('2026-11-20')
            neutral = nba.generate_predictions(latest, apply_injury_boosts=False,
                                               apply_fp_blend=False, apply_game_totals=False,
                                               slate_date='2026-11-20')
            self.assertEqual(fetch.call_count, 1)
        scores = ['pts_pred', 'pra_pred', 'pa_pred', 'pts_meta_prob', 'pts_p_over_cal']
        pd.testing.assert_frame_equal(out[scores], neutral[scores])
        self.assertEqual(out.game_total.tolist(), [260])
        self.assertEqual(out.game_total_scale.tolist(), [1.0])


if __name__ == '__main__':
    unittest.main()
