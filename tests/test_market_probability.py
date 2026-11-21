"""Market-offset model causality, fixed-price anchoring and isolated live scoring."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import logit

import market_probability as mp
import nba_props as nba
import run_daily as daily
from scripts.train_market_probability import build_dataset, evaluate, losses, summarize


def training():
    return pd.DataFrame({'game_date': ['2026-11-19'] * 100, 'prop': ['pts'] * 100,
                         'line': [20.5] * 100, 'l10': [21] * 100, 'l5': [22] * 100,
                         'mp_l10': [30] * 100, 'games_prior': [20] * 100,
                         'nv_over': [.5] * 100, 'over_hit': [1] * 65 + [0] * 35})


def boxes():
    return pd.DataFrame([{'player': 'Test', 'game_date': day.strftime('%Y-%m-%d'),
                          'mp': 30, 'pts': i + 1, 'trb': 5, 'ast': 4, 'stl': 1, 'blk': 1}
                         for i, day in enumerate(pd.date_range('2026-11-01', periods=19))])


def quote(**overrides):
    return pd.DataFrame([{'game_date': '2026-11-20', 'player': 'Test', 'prop': 'player_points',
                          'line': 20.5, 'over_price': -110, 'under_price': -110,
                          'snapshot_ts': '2026-11-20T14:00:00Z', 'commence_time': '2026-11-21T01:00:00Z',
                          **overrides}])


NOW = pd.Timestamp('2026-11-20T15:00:00Z')


class MarketOffsetTests(unittest.TestCase):
    def test_rolling_features_exclude_current_and_future_outcomes(self):
        offers = pd.DataFrame({'player': ['Test', 'Test'], 'game_date': ['2026-11-12', '2026-11-20'],
                               'prop': ['pts', 'pts'], 'line': [12, 20]})
        box = boxes()
        before = mp.attach_history(offers, box)
        self.assertAlmostEqual(before.loc[0, 'l10'], 6.5)  # Nov 2..11, never Nov 12
        self.assertEqual(before.history_through.tolist(), ['2026-11-11', '2026-11-19'])
        changed = box.copy()
        changed.loc[changed.game_date.ge('2026-11-12'), 'pts'] = 999
        after = mp.attach_history(offers.iloc[:1], changed)
        pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])
        night_offer = offers.iloc[1:].assign(feature_cutoff_date='2026-11-18')
        night = mp.attach_history(night_offer, box)
        self.assertEqual(night.history_through.iloc[0], '2026-11-17')

    def test_duplicate_actuals_fail_instead_of_choosing_an_outcome(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            mp.prepare_boxes(pd.concat([boxes(), boxes().iloc[:1]]))

    def test_zero_correction_is_market_and_prices_remain_the_anchor(self):
        frame = training().iloc[:2].copy()
        frame.nv_over = [.4, .6]
        model = mp.fit(training())
        model['coefficients'] = [0.] * (len(mp.PROPS) * len(mp.FEATURES))
        np.testing.assert_allclose(mp.predict(model, frame), frame.nv_over, rtol=0, atol=1e-15)
        model['coefficients'][0] = .15
        result = mp.predict(model, frame)
        np.testing.assert_allclose(logit(result) - logit(frame.nv_over), [.15, .15], atol=1e-12)

    def test_residual_learns_and_is_bounded_and_missing_history_falls_back(self):
        model = mp.fit(training())
        frame = training().iloc[:3].copy()
        result = mp.predict(model, frame)
        self.assertTrue((result > .5).all())
        self.assertTrue((losses(frame.over_hit, result) < losses(frame.over_hit, frame.nv_over)).all())
        model['coefficients'][0] = 999
        result = mp.predict(model, frame)
        np.testing.assert_allclose(logit(result) - logit(frame.nv_over), mp.MAX_CORRECTION)
        frame.loc[0, 'l10'] = np.nan
        frame.loc[1, 'nv_over'] = np.nan
        out = mp.predict(model, frame)
        self.assertEqual(out[0], .5)
        self.assertTrue(np.isnan(out[1]))

    def test_invalid_price_pairs_and_model_schemas_fail_safely(self):
        q, _, _ = mp.paired_market(pd.Series([-110, 50, -110, np.inf]), pd.Series([-110, -110, np.nan, -110]))
        self.assertEqual(q[0], .5)
        self.assertTrue(q.iloc[1:].isna().all())
        model = mp.fit(training())
        model['features'] = ['unknown']
        with self.assertRaisesRegex(ValueError, 'contract'):
            mp.predict(model, training())

    def test_live_snapshot_date_training_cutoff_and_pair_validation(self):
        model = mp.fit(training())
        model['evaluation_status'] = 'shadow_only'
        out = mp.score_live(model, quote(), boxes(), '2026-11-20', NOW)
        self.assertEqual(len(out), 1)
        self.assertFalse(out.published.any())
        self.assertTrue(out.tracking_only.all())
        for change in [{'snapshot_ts': '2026-11-20T16:00:00Z'}, {'commence_time': None},
                       {'under_price': None}, {'game_date': '2026-11-21'},
                       {'snapshot_ts': '2026-11-18T14:00:00Z'}]:
            with self.subTest(change=change):
                self.assertTrue(mp.score_live(model, quote(**change), boxes(), '2026-11-20', NOW).empty)
        model['trained_through'] = '2026-11-20'
        with self.assertRaisesRegex(ValueError, 'before'):
            mp.score_live(model, quote(), boxes(), '2026-11-20', NOW)

    def test_bootstrap_and_losses_compare_the_same_rows(self):
        frame = training().iloc[:4].copy()
        frame['game_date'] = ['2026-11-17', '2026-11-18', '2026-11-19', '2026-11-20']
        frame['anchored_over'] = [.6, .6, .6, np.nan]
        frame['dec_under'] = 1.9
        result = summarize(frame)
        self.assertEqual(result['rows'], 3)
        self.assertAlmostEqual(result['difference'], -np.log(.6) + np.log(.5))

    def test_walk_forward_never_uses_held_date_or_later_outcomes(self):
        train = pd.concat([training().iloc[:40].assign(
            game_date=day.strftime('%Y-%m-%d'), over_hit=[0, 1] * 20)
            for day in pd.date_range('2026-04-01', periods=35)], ignore_index=True)
        future = pd.concat([training().iloc[:40].assign(
            game_date=day.strftime('%Y-%m-%d'), over_hit=[0, 1] * 20)
            for day in pd.date_range('2026-11-01', periods=5)], ignore_index=True)
        frame = pd.concat([train, future], ignore_index=True).assign(dec_under=1.9)
        _, _, first, summary = evaluate(frame)
        changed = frame.copy()
        changed.loc[changed.game_date.ge('2026-11-01'), 'over_hit'] = 1
        _, _, second, _ = evaluate(changed)
        np.testing.assert_array_equal(first.loc[first.game_date.eq('2026-11-01'), 'anchored_over'],
                                      second.loc[second.game_date.eq('2026-11-01'), 'anchored_over'])
        self.assertTrue((first.trained_through < first.game_date).all())
        self.assertFalse(summary['promotion_eligible'])

    def test_dataset_uses_first_pretip_foundation_and_available_morning_forward_quote(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data').mkdir()
            old_boxes = boxes().assign(game_date=pd.date_range('2026-02-12', periods=19).strftime('%Y-%m-%d'))
            all_boxes = pd.concat([old_boxes, boxes()], ignore_index=True)
            all_boxes.to_csv(root/'data/nba_data.csv', index=False)
            rows = []
            for day, tip, stamps in [
                ('2026-03-02', '2026-03-03T01:00:00Z', ['2026-03-02T18:00:00Z', '2026-03-02T23:00:00Z']),
                ('2026-11-19', '2026-11-20T01:00:00Z', ['2026-11-19T15:00:00Z', '2026-11-19T20:00:00Z'])]:
                for i, stamp in enumerate(stamps):
                    rows.append({'game_date': day, 'player': 'Test', 'prop': 'player_points',
                                 'line': 20.5 + i, 'snapshot_ts': stamp, 'commence_time': tip,
                                 'over_price': -110, 'under_price': -110, 'event_id': day,
                                 'source': 'odds_api_fanduel'})
            pd.DataFrame(rows).to_csv(root/'data/historical_lines.csv', index=False)
            out, _, _ = build_dataset(root)
            self.assertEqual(out.line.tolist(), [20.5, 20.5])
            self.assertTrue((pd.to_datetime(out.snapshot_ts, utc=True) < pd.to_datetime(out.commence_time, utc=True)).all())
            self.assertTrue((out.history_through < out.game_date).all())


class MarketShadowIntegrationTests(unittest.TestCase):
    def test_logging_preserves_incumbent_and_first_contract_and_grades_isolated_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = mp.fit(training())
            model['evaluation_status'] = 'shadow_only'
            (root / 'market_probability_shadow.json').write_text(json.dumps(model), encoding='utf-8')
            boxes().to_csv(root / 'nba_data.csv', index=False)
            pred = pd.DataFrame({'player': ['Test'], 'pts_line': [20.5], 'pts_p_over_cal': [.7]})
            original = pred.copy()
            with patch.dict(nba.CONFIG, {'output_dir': root, 'data_dir': root}), contextlib.redirect_stdout(io.StringIO()):
                out = nba._log_market_probability_shadow(pred, quote(), '2026-11-20', now_utc=NOW, run_id='run')
                self.assertEqual(len(out), 1)
                self.assertEqual(out.production_cal.tolist(), [.7])
                path = root / 'market_probability_history.csv'
                first = path.read_bytes()
                nba._log_market_probability_shadow(pred, quote(line=22.5), '2026-11-20', now_utc=NOW)
                self.assertEqual(path.read_bytes(), first)
            pd.testing.assert_frame_equal(pred, original)
            self.assertFalse((root / 'picks_history.csv').exists())
            hist = pd.read_csv(path, keep_default_na=False)
            hist['game_date'] = (pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            hist.to_csv(path, index=False)
            game = boxes().iloc[-1:].copy()
            game['game_date'] = hist.game_date.iloc[0]
            game['pts'] = 10
            game.to_csv(root / 'nba_data.csv', index=False)
            with patch.object(daily, 'OUTPUT_DIR', root), patch.object(daily, 'GAME_LOG_CSV', root/'nba_data.csv'), \
                    patch.object(daily, '_hydrate_live_box_scores', side_effect=lambda b, *a, **k: (b, {hist.game_date.iloc[0]})), \
                    self.assertLogs(daily.logger, level='INFO') as logs:
                daily.step1_9_grade_shadow_lanes([], history_name='market_probability_history.csv')
            self.assertEqual(pd.read_csv(path).result.tolist(), ['WIN'])
            self.assertIn('market-offset vs market: 1 comparable rows', '\n'.join(logs.output))

    def test_missing_or_corrupt_shadow_artifact_never_changes_live_inputs(self):
        with TemporaryDirectory() as tmp, patch.dict(nba.CONFIG, {'output_dir': Path(tmp)}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(nba._log_market_probability_shadow(pd.DataFrame(), quote(), '2026-11-20').empty)
            (Path(tmp) / 'market_probability_shadow.json').write_text('{}', encoding='utf-8')
            self.assertTrue(nba._log_market_probability_shadow(pd.DataFrame(), quote(), '2026-11-20').empty)


if __name__ == '__main__':
    unittest.main()
