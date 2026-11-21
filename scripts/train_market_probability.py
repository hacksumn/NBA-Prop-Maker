"""Build timed FanDuel offers and evaluate a market-offset model, without promotion.

python scripts/train_market_probability.py --output-dir output/market_probability_<run>
One immutable run directory; no existing model/config/history is ever overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import market_probability as mp

TRAIN_END = '2026-05-31'
FORWARD_START = '2026-10-21'
END = '2026-11-19'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def archived_runs(root, end=END):
    """First morning run per date, same convention as DEC-104; no after-tip use."""
    files = {}
    for day in sorted((root / 'output/archive').glob('20??-??-??')):
        if not FORWARD_START <= day.name <= end:
            continue
        runs = []
        for path in day.glob('predictions_*_*.csv'):
            match = re.search(r'_(\d{6})(?:_\d+)?\.csv$', path.name)
            if match and '060000' <= match[1] <= '120000':
                runs.append((match[1], path))
        if runs:
            stamp, path = sorted(runs)[0]
            files[day.name] = (pd.Timestamp(f'{day.name} {stamp[:2]}:{stamp[2:4]}:{stamp[4:]}',
                                            tz='America/Chicago').tz_convert('UTC'), path)
    return files


def build_dataset(root, end=END):
    """Select first pre-tip foundation quotes and timed morning forward offers."""
    paths = [root / 'data/historical_lines.csv', root / 'data/nba_data.csv']
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    archives = archived_runs(root, end=end)
    chunks = []
    columns = ['game_date', 'player', 'prop', 'line', 'snapshot_ts', 'commence_time',
               'over_price', 'under_price', 'event_id', 'source']
    for chunk in pd.read_csv(paths[0], chunksize=300000, usecols=columns, low_memory=False):
        selected = chunk.source.eq('odds_api_fanduel') & chunk.game_date.between('2026-02-01', end)
        chunks.append(chunk[selected])
    offers = pd.concat(chunks, ignore_index=True)
    counts = {'raw_fanduel_rows': len(offers)}
    offers['prop'] = offers.prop.map({v: k for k, v in mp.ODDS_PROPS.items()})
    offers['player_norm'] = offers.player.map(mp.normalize_name)
    offers['line'] = pd.to_numeric(offers.line, errors='coerce')
    offers['quote_ts'] = pd.to_datetime(offers.snapshot_ts, errors='coerce', format='mixed', utc=True)
    offers['tip_ts'] = pd.to_datetime(offers.commence_time, errors='coerce', format='mixed', utc=True)
    # Prior-season foundation: fixed 10am Chicago. Forward: actual archived run.
    captures = {day: pd.Timestamp(day + ' 10:00:00', tz='America/Chicago').tz_convert('UTC')
                for day in offers.game_date.unique()}
    captures.update({day: stamp for day, (stamp, _) in archives.items()})
    offers['capture_ts'] = pd.to_datetime(offers.game_date.map(captures), utc=True)
    # Earlier archives are mostly late-day captures. Use their FIRST pre-tip
    # offer, not a fabricated morning quote. This training/evaluation timing
    # difference is reported and does not relax the forward morning boundary.
    prior_season = offers.game_date.le(TRAIN_END)
    offers.loc[prior_season, 'capture_ts'] = offers.loc[prior_season, 'quote_ts']
    offers['nv_over'], offers['dec_over'], offers['dec_under'] = mp.paired_market(offers.over_price, offers.under_price)
    valid = (offers.prop.notna() & offers.line.ge(0) & offers.nv_over.notna()
             & offers.quote_ts.le(offers.capture_ts) & offers.tip_ts.gt(offers.capture_ts)
             & offers.capture_ts.sub(offers.quote_ts).le(pd.Timedelta(hours=24))
             & offers.tip_ts.sub(offers.quote_ts).le(pd.Timedelta(hours=24))
             & offers.tip_ts.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d').eq(offers.game_date))
    offers = offers[valid].copy()
    counts['valid_timed_price_rows'] = len(offers)
    # One main offer per player/prop/date, never treat repeated snapshots as games.
    offers = offers.sort_values(['quote_ts', 'line'], kind='stable')
    key = ['game_date', 'player_norm', 'prop']
    offers = pd.concat([offers[offers.game_date.le(TRAIN_END)].drop_duplicates(key, keep='first'),
                        offers[offers.game_date.gt(TRAIN_END)].drop_duplicates(key, keep='last')])
    counts['unique_selected_offers'] = len(offers)
    box = pd.read_csv(paths[1], usecols=['player', 'game_date', 'mp', 'pts', 'trb', 'ast', 'stl', 'blk'], low_memory=False)
    actuals = mp.prepare_boxes(box).melt(id_vars=['game_date', 'player_norm'], value_vars=mp.PROPS,
                                       var_name='prop', value_name='actual')
    offers = offers.merge(actuals, on=['game_date', 'player_norm', 'prop'], how='left', validate='many_to_one')
    counts['missing_or_nonparticipant_actuals'] = int(offers.actual.isna().sum())
    counts['pushes'] = int(offers.actual.eq(offers.line).sum())
    offers = offers[offers.actual.notna() & offers.actual.ne(offers.line)].copy()
    offers['over_hit'] = offers.actual.gt(offers.line).astype(int)
    offers['feature_cutoff_date'] = offers.capture_ts.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d')
    offers = mp.attach_history(offers, box)
    if (offers.history_through.dropna() >= offers.loc[offers.history_through.notna(), 'feature_cutoff_date']).any():
        raise AssertionError('Rolling history is not strictly before the capture date')
    counts['graded_nonpush_offers'] = len(offers)
    counts['valid_residual_features'] = int(mp.design(offers)[1].sum())
    # Attach already-recorded production probabilities ONLY for comparison, not X.
    comparisons = []
    for day, (_, path) in archives.items():
        hashes[str(path.relative_to(root))] = digest(path)
        pred = pd.read_csv(path, low_memory=False)
        for prop in mp.PROPS:
            if not {f'{prop}_line', f'{prop}_p_over_cal'} <= set(pred):
                continue
            comparisons.append(pd.DataFrame({'game_date': day, 'player_norm': pred.player.map(mp.normalize_name),
                                              'prop': prop, 'line': pred[f'{prop}_line'],
                                              'production_cal': pred[f'{prop}_p_over_cal']}))
    if comparisons:
        comparison = pd.concat(comparisons).dropna(subset=['line'])
        if comparison.duplicated(['game_date', 'player_norm', 'prop', 'line']).any():
            raise ValueError('Ambiguous archived predictions; cannot select a production probability')
        offers = offers.merge(comparison, on=['game_date', 'player_norm', 'prop', 'line'], how='left', validate='many_to_one')
    if any(digest(root / path) != sha for path, sha in hashes.items()):
        raise RuntimeError('Inputs changed during build; rerun against stable source files')
    return offers.sort_values(['game_date', 'player_norm', 'prop']).reset_index(drop=True), counts, hashes


def losses(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def summarize(frame, prediction='anchored_over'):
    """Comparable-row predictive metrics and unselected flat one-unit UNDER replay."""
    valid = frame[prediction].between(0, 1) & frame.nv_over.between(0, 1)
    rows = frame[valid].copy()
    if rows.empty:
        return {'rows': 0}
    y, p, q = rows.over_hit.to_numpy(), rows[prediction].to_numpy(), rows.nv_over.to_numpy()
    delta = losses(y, p) - losses(y, q)
    daily = pd.DataFrame({'day': rows.game_date, 'delta': delta, 'n': 1}).groupby('day').sum()
    draws = np.random.default_rng(109).integers(0, len(daily), (10000, len(daily)))
    sums = daily.to_numpy()[draws].sum(axis=1)
    ci = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist()
    # Predeclared diagnostic, not a new live-selection rule or threshold search.
    selected = (1 - p) * rows.dec_under.to_numpy() - 1 >= .01
    units = np.where(y[selected] == 0, rows.dec_under.to_numpy()[selected] - 1, -1)
    return {'rows': len(rows), 'dates': int(rows.game_date.nunique()),
            'market_log_loss': float(losses(y, q).mean()), 'model_log_loss': float(losses(y, p).mean()),
            'difference': float(delta.mean()), 'date_cluster_difference_95': ci,
            'market_brier': float(np.mean((y - q) ** 2)), 'model_brier': float(np.mean((y - p) ** 2)),
            'market_calibration_bias': float(np.mean(q - y)),
            'model_calibration_bias': float(np.mean(p - y)),
            'mean_probability_change': float(np.abs(p - q).mean()),
            'under_ev_001': {'picks': int(selected.sum()), 'profit_units': float(units.sum()),
                             'roi': float(units.mean()) if len(units) else None,
                             'hit_rate': float((y[selected] == 0).mean()) if selected.any() else None}}


def evaluate(frame):
    """Frozen prior-season fit plus expanding whole-date evaluation of this season."""
    train = frame[frame.game_date.le(TRAIN_END)]
    forward = frame[frame.game_date.ge(FORWARD_START)].copy()
    if train.game_date.nunique() < 30 or len(train) < 1000 or forward.game_date.nunique() < 5:
        raise ValueError('Insufficient prior-season training or forward-date coverage')
    frozen = mp.fit(train)
    frozen_rows = forward.copy()
    frozen_rows['anchored_over'] = mp.predict(frozen, frozen_rows)
    walks = []
    for day, rows in forward.groupby('game_date', sort=True):
        past = frame[frame.game_date.lt(day)]
        model = mp.fit(past)
        if model['trained_through'] >= day:
            raise AssertionError('Whole-date chronology violated')
        rows = rows.copy()
        rows['anchored_over'] = mp.predict(model, rows)
        rows['trained_through'] = model['trained_through']
        walks.append(rows)
    walked = pd.concat(walks, ignore_index=True)
    summary = {'specification': {'version': mp.VERSION, 'train_end': TRAIN_END,
                                  'forward_start': FORWARD_START, 'data_end': str(frame.game_date.max()),
                                  'penalty': mp.PENALTY, 'max_logit_correction': mp.MAX_CORRECTION,
                                  'features': list(mp.FEATURES), 'props': list(mp.PROPS)},
               'foundation_training_rows': len(train), 'foundation_training_dates': int(train.game_date.nunique()),
               'frozen_prior_season': summarize(frozen_rows), 'walk_forward': summarize(walked),
               'walk_forward_by_prop': {p: summarize(g) for p, g in walked.groupby('prop')},
               'production_cal_comparable': summarize(walked, 'production_cal') if 'production_cal' in walked else {},
               'anchored_on_production_comparable': summarize(walked[walked.production_cal.between(0, 1)])
                   if 'production_cal' in walked else {},
               'limitations': [
                   'November outcomes were already inspected in DEC-104; this is chronological retrospective evidence, not untouched prospective proof.',
                   'Foundation uses first recorded pre-tip quotes (many late-day); forward archived slates use actual first morning runs, other dates use 10am. Capture timing differs.',
                   'Main-line no-vig price is a baseline probability, not an executable return; profit uses offered side odds.',
                   'Related props and player-game outcomes overlap; date-cluster intervals account for nights, not independent replication.',
                   'ROI diagnostics do not replay incumbent meta gates, game/player exposure or parlays.',
                   'Historical master-box revisions can differ from what was known live; prospective scoring is still required.',
               ]}
    # No subgroup cherry-picking: require overall improvement in BOTH tests plus
    # a distinct prospective sample. Historical evidence alone cannot promote.
    historical_pass = all(summary[key]['difference'] < 0 and
                          summary[key]['date_cluster_difference_95'][1] < 0
                          for key in ['frozen_prior_season', 'walk_forward'])
    summary['historical_evidence_pass'] = historical_pass
    summary['promotion_eligible'] = False
    summary['promotion_reason'] = 'Prospective validation required; no live promotion performed'
    final = mp.fit(frame)
    final['evaluation_status'] = 'shadow_only'
    return final, frozen_rows, walked, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-end', default=END, help='Last completed date (YYYY-MM-DD); never the current slate')
    parser.add_argument('--install-shadow', action='store_true',
                        help='Install only output/market_probability_shadow.json; never replace live models/config')
    args = parser.parse_args()
    end = pd.Timestamp(args.data_end).strftime('%Y-%m-%d')
    if end >= pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d'):
        raise ValueError('Training data-end must precede today; partial/current slates cannot train the candidate')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_hashes = {str(p.relative_to(args.root)): digest(p) for p in (args.root / 'models').glob('*') if p.is_file()}
    frame, counts, hashes = build_dataset(args.root, end=end)
    print('Dataset', json.dumps(counts), flush=True)
    frame.to_csv(args.output_dir / 'priced_dataset.csv', index=False)
    candidate, frozen, walked, summary = evaluate(frame)
    summary.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes,
                    'dataset_sha256': digest(args.output_dir / 'priced_dataset.csv'), 'build_counts': counts,
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
                    'source_code_sha256': {p: digest(ROOT / p) for p in ['market_probability.py', 'scripts/train_market_probability.py']}})
    if model_hashes != {str(p.relative_to(args.root)): digest(p) for p in (args.root / 'models').glob('*') if p.is_file()}:
        raise RuntimeError('Production model files changed during training; investigate before interpreting this run')
    summary['production_model_files_unchanged'] = True
    summary['production_model_sha256'] = model_hashes
    candidate['training_dataset_sha256'] = summary['dataset_sha256']
    frozen.to_csv(args.output_dir / 'frozen_forward.csv', index=False)
    walked.to_csv(args.output_dir / 'walk_forward.csv', index=False)
    for name, payload in [('candidate.json', candidate), ('evaluation.json', summary)]:
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if args.install_shadow:
        from atomic_io import atomic_write_text
        shadow = {**candidate, 'evaluation_artifact': str((args.output_dir / 'evaluation.json').resolve()),
                  'evaluation_sha256': digest(args.output_dir / 'evaluation.json')}
        atomic_write_text(args.root / 'output/market_probability_shadow.json',
                          json.dumps(shadow, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: summary[key] for key in ['frozen_prior_season', 'walk_forward', 'promotion_eligible']}, indent=2))


if __name__ == '__main__':
    main()
