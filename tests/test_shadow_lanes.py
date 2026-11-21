"""Prospective lane boundaries, immutable pre-tip offers and isolated grading."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import nba_props as nba
import run_daily as daily


NOW = pd.Timestamp('2026-11-20T15:00:00Z')


def prediction(name='Test', prop='pts', meta=.6, line=20.5):
    return {'player': name, 'team': 'BOS', f'{prop}_meta_prob': meta,
            f'{prop}_line': line, f'{prop}_pred': 22, f'{prop}_p_over_cal': .8}


def market(name='Test', prop='player_points', line=20.5, **overrides):
    return {'player': name, 'prop': prop, 'line': line, 'game_date': '2026-11-20',
            'over_price': -110, 'under_price': -110, 'event_id': 'test-event',
            'commence_time': '2026-11-21T01:00:00Z', 'snapshot_ts': '2026-11-20T14:00:00Z',
            **overrides}


class ShadowLoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cfg = patch.dict(nba.CONFIG, {'output_dir': self.out})
        self.cfg.start()
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.cfg.stop()
        self.tmp.cleanup()

    def log(self, predictions, lines):
        return nba._log_shadow_lanes(pd.DataFrame(predictions), pd.DataFrame(lines),
                                     '2026-11-20', now_utc=NOW, run_id='test-run')

    def test_fixed_meta_boundaries_fraction_units_and_under_direction(self):
        predictions, lines, expected = [], [], []
        for prop, metas in [('pts', [.4999, .5, .6, .7, .7001, 60, None]),
                            ('pra', [.7, .7001, 1, 1.001, float('nan')])]:
            for idx, meta in enumerate(metas):
                name = f'{prop}{idx}'
                predictions.append(prediction(name, prop, meta))
                lines.append(market(name, 'player_points' if prop == 'pts' else 'player_points_rebounds_assists'))
                if (prop == 'pts' and idx in [1, 2, 3]) or (prop == 'pra' and idx in [1, 2]):
                    expected.append(name)
        out = self.log(predictions, lines)
        self.assertCountEqual(out.player.tolist(), expected)
        self.assertTrue(out.direction.eq('UNDER').all())  # p_over=.8 still observes UNDER
        self.assertTrue(out.tracking_only.all())
        self.assertFalse(out.published.any())
        self.assertTrue(out.run_id.eq('test-run').all())
        self.assertTrue(out.side_price.eq(-110).all())
        self.assertEqual(set(self.out.glob('*.csv')), {self.out/'shadow_lanes_history.csv'})
        self.assertFalse(out.price_gate_pass.any())  # collect even rejected low-EV offers

    def test_first_capture_survives_price_meta_and_line_changes(self):
        self.log([prediction()], [market()])
        path = self.out/'shadow_lanes_history.csv'
        original = path.read_bytes()
        self.log([prediction(meta=.65, line=21.5)], [market(line=21.5, under_price=150)])
        self.assertEqual(path.read_bytes(), original)

    def test_only_paired_known_pretip_quotes_from_the_slate_are_captured(self):
        for overrides in [
            {'commence_time': '2026-11-20T14:00:00Z'}, {'commence_time': None},
            {'snapshot_ts': None}, {'snapshot_ts': '2026-11-20T16:00:00Z'},
            {'under_price': None}, {'over_price': None}, {'game_date': '2026-11-21'},
            {'commence_time': '2026-11-22T01:00:00Z'},
        ]:
            with self.subTest(overrides=overrides):
                out = self.log([prediction()], [market(**overrides)])
                self.assertTrue(out.empty)
                self.assertFalse((self.out/'shadow_lanes_history.csv').exists())

    def test_logging_failure_never_blocks_the_card(self):
        with patch.object(nba, '_append_only_history', side_effect=OSError('disk full')):
            self.assertTrue(self.log([prediction()], [market()]).empty)

    def test_logging_does_not_modify_any_input_or_live_policy(self):
        predictions, lines = pd.DataFrame([prediction()]), pd.DataFrame([market()])
        before_p, before_l, config = predictions.copy(), lines.copy(), nba.CONFIG.copy()
        nba._log_shadow_lanes(predictions, lines, '2026-11-20', now_utc=NOW)
        pd.testing.assert_frame_equal(predictions, before_p)
        pd.testing.assert_frame_equal(lines, before_l)
        self.assertEqual(config, nba.CONFIG)
        # These hypotheses remain disabled in the active publication policy.
        policy = nba._load_target_quality_policy()
        self.assertFalse(policy['pts']['allow_under'])
        self.assertFalse(policy['pra']['allow_under'])


class ShadowGradingTests(unittest.TestCase):
    def test_daily_grades_combos_payouts_and_missing_dates_without_touching_live_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            today = pd.Timestamp.now(tz='America/Chicago').normalize()
            yesterday = (today - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            gap = (today - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
            rows = []
            for name, prop, line, date in [('Winner', 'PRA', 30.5, yesterday),
                                           ('Loser', 'PTS', 20.5, yesterday),
                                           ('Push', 'PTS', 20, yesterday),
                                           ('Absent', 'PTS', 20, yesterday),
                                           ('Gap', 'PTS', 20, gap)]:
                rows.append({'player': name, 'prop': prop, 'direction': 'UNDER', 'line': line,
                             'game_date': date, 'shadow_lane': 'pra_under_meta_gt_070' if prop == 'PRA' else 'pts_under_meta_050_070',
                             'result': '', 'actual': '', 'profit_units': '', 'graded_at': '',
                             'side_price': 150, 'side_decimal_odds': 2.5, 'published': False})
            path = root/'shadow_lanes_history.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            pd.DataFrame([{'player': 'Winner', 'game_date': yesterday, 'pts': 20, 'trb': 5, 'ast': 4},
                          {'player': 'Loser', 'game_date': yesterday, 'pts': 21, 'trb': 0, 'ast': 0},
                          {'player': 'Push', 'game_date': yesterday, 'pts': 20, 'trb': 0, 'ast': 0}]).to_csv(root/'box.csv', index=False)
            for name in ['picks_history.csv', 'betslips_history.csv', 'picks_latest.csv']:
                (root/name).write_text('unchanged\n', encoding='utf-8')
            errors = []
            with patch.object(daily, 'OUTPUT_DIR', root), patch.object(daily, 'GAME_LOG_CSV', root/'box.csv'), \
                    patch.object(daily, '_hydrate_live_box_scores', side_effect=lambda box, *args, **kwargs: (box, {yesterday})):
                daily.step1_9_grade_shadow_lanes(errors)
                graded = pd.read_csv(path, keep_default_na=False)
                self.assertEqual(graded.result.tolist(), ['WIN', 'LOSS', 'PUSH', 'DNP', ''])
                self.assertEqual(graded.profit_units.tolist(), ['1.5', '-1', '0', '', ''])
                original = path.read_bytes()
                daily.step1_9_grade_shadow_lanes(errors)
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(errors, [])
            for name in ['picks_history.csv', 'betslips_history.csv', 'picks_latest.csv']:
                self.assertEqual((root/name).read_text(encoding='utf-8'), 'unchanged\n')

    def test_grading_errors_are_nonfatal(self):
        with TemporaryDirectory() as tmp, patch.object(daily, 'OUTPUT_DIR', Path(tmp)):
            (Path(tmp)/'shadow_lanes_history.csv').write_text('bad schema\n', encoding='utf-8')
            errors = []
            daily.step1_9_grade_shadow_lanes(errors)
            self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
