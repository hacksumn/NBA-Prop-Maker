"""Market-offset probabilities: learn a small correction to paired FanDuel prices.

No production sidecar, calibrator, pick policy or stat projection is replaced here.
Fitting and inference share the same strictly historical rolling-feature builder.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

VERSION = 'market_offset_v1'
PROPS = ('pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk')
ODDS_PROPS = dict(zip(PROPS, ('player_points', 'player_rebounds', 'player_assists',
                            'player_points_rebounds_assists', 'player_points_rebounds',
                            'player_points_assists', 'player_steals', 'player_blocks')))
FEATURES = ('intercept', 'line_level', 'l10_gap', 'momentum', 'minutes')
PENALTY = 0.02  # Fixed before evaluation; no search over holdout outcomes.
MAX_CORRECTION = 0.50  # Logit units, not a probability/ROI guarantee.


def normalize_name(name):
    """Use one normalizer for priced offers, box histories and live rows."""
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()


def paired_market(over, under):
    """Return no-vig over probability and side decimal odds; invalid pairs stay NaN."""
    over, under = pd.to_numeric(over, errors='coerce'), pd.to_numeric(under, errors='coerce')

    def decimal(price):
        valid = np.isfinite(price) & (price.abs() >= 100)
        return pd.Series(np.where(price < 0, 1 + 100 / price.abs(), 1 + price / 100),
                         index=price.index).where(valid)

    do, du = decimal(over), decimal(under)
    q = (1 / do) / (1 / do + 1 / du)
    return q.where(do.notna() & du.notna()), do, du


def prepare_boxes(box):
    """Validate one played player-game; outcomes are never used as same-game features."""
    frame = box.copy()
    frame['game_date'] = pd.to_datetime(frame.game_date, errors='raise').dt.strftime('%Y-%m-%d')
    frame['player_norm'] = frame.player.map(normalize_name)
    for column in ['mp', 'pts', 'trb', 'ast', 'stl', 'blk']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame[frame.mp.gt(0)].copy()
    if frame.duplicated(['game_date', 'player_norm']).any():
        raise ValueError('Ambiguous player/date box scores; do not silently choose an actual')
    frame['pra'] = frame.pts + frame.trb + frame.ast
    frame['pr'] = frame.pts + frame.trb
    frame['pa'] = frame.pts + frame.ast
    return frame.sort_values(['player_norm', 'game_date']).reset_index(drop=True)


def attach_history(offers, box):
    """As-of join prior played games only, including for a future/DNP offer date.

    Build unshifted rolling values dated at their completed game, then select
    strictly before the offer date. Current-game outcomes cannot enter X.
    """
    frame = offers.copy().reset_index(drop=True)
    frame['player_norm'] = frame.player.map(normalize_name)
    frame['game_date'] = pd.to_datetime(frame.game_date, errors='raise').dt.strftime('%Y-%m-%d')
    history = prepare_boxes(box)
    grouped = history.groupby('player_norm', sort=False)
    history['history_through'] = history.game_date
    history['games_prior'] = grouped.cumcount() + 1
    history['mp_l10'] = grouped.mp.transform(lambda s: s.rolling(10, min_periods=10).mean())
    columns = ['history_through', 'games_prior', 'mp_l10']
    for prop in PROPS:
        for window in (5, 10):
            col = f'{prop}_l{window}'
            history[col] = grouped[prop].transform(lambda s: s.rolling(window, min_periods=window).mean())
            columns.append(col)
    # Keep as-of keys numeric and sorted; equality is deliberately excluded.
    # A next-day offer captured tonight must not see tonight's completed/partial
    # game rows. Date-only box scores cannot prove an intraday completion time.
    frame['_day'] = pd.to_datetime(frame.get('feature_cutoff_date', frame.game_date))
    frame['_order'] = np.arange(len(frame))
    history['_day'] = pd.to_datetime(history.game_date)
    frame = frame.drop(columns=[c for c in columns if c in frame])
    merged = pd.merge_asof(frame.sort_values('_day'), history[['_day', 'player_norm'] + columns].sort_values('_day'),
                           on='_day', by='player_norm', direction='backward', allow_exact_matches=False)
    merged['l10'] = [r.get(f'{r.prop}_l10', np.nan) for _, r in merged.iterrows()]
    merged['l5'] = [r.get(f'{r.prop}_l5', np.nan) for _, r in merged.iterrows()]
    return merged.sort_values('_order').drop(columns=['_day', '_order'] +
                                              [f'{p}_l{w}' for p in PROPS for w in (5, 10)]).reset_index(drop=True)


def design(frame):
    """Fixed, bounded per-prop residual basis; identical for fitting and live scoring."""
    line = pd.to_numeric(frame.line, errors='coerce').to_numpy(float)
    l10 = pd.to_numeric(frame.l10, errors='coerce').to_numpy(float)
    l5 = pd.to_numeric(frame.l5, errors='coerce').to_numpy(float)
    minutes = pd.to_numeric(frame.mp_l10, errors='coerce').to_numpy(float)
    scale = np.sqrt(np.maximum(line, 1))
    base = np.column_stack([np.ones(len(frame)), np.log1p(np.maximum(line, 0)) / np.log(51),
                            (l10 - line) / scale, (l5 - l10) / scale, minutes / 36])
    valid = np.isfinite(base).all(axis=1) & (line >= 0) & frame.prop.isin(PROPS).to_numpy()
    valid &= pd.to_numeric(frame.games_prior, errors='coerce').ge(10).to_numpy()
    base = np.clip(np.nan_to_num(base, nan=0, posinf=0, neginf=0), -3, 3)
    X = np.concatenate([base * frame.prop.eq(prop).to_numpy()[:, None] for prop in PROPS], axis=1)
    return X, valid


def fit(frame, penalty=PENALTY):
    """Fit Bernoulli residual likelihood with market logit fixed at coefficient one."""
    X, valid = design(frame)
    q = pd.to_numeric(frame.nv_over, errors='coerce').to_numpy(float)
    y = pd.to_numeric(frame.over_hit, errors='coerce').to_numpy(float)
    valid &= np.isfinite(q) & (q > 0) & (q < 1) & np.isin(y, [0, 1])
    if valid.sum() < 50 or len(np.unique(y[valid])) != 2:
        raise ValueError('Need at least 50 valid binary priced rows with both outcomes')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Residual penalty must be positive and finite')
    X, q, y = X[valid], q[valid], y[valid]
    offset = logit(q)

    def objective(beta):
        z = offset + X @ beta
        loss = np.mean(np.logaddexp(0, z) - y * z) + penalty * np.dot(beta, beta) / 2
        grad = X.T @ (expit(z) - y) / len(y) + penalty * beta
        return loss, grad

    fitted = minimize(objective, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f'Market residual fitting failed: {fitted.message}')
    selected = frame.loc[valid]
    return {'version': VERSION, 'props': list(PROPS), 'features': list(FEATURES),
            'coefficients': fitted.x.tolist(), 'penalty': penalty, 'max_logit_correction': MAX_CORRECTION,
            'trained_through': str(selected.game_date.max()), 'training_rows': len(selected),
            'training_dates': int(selected.game_date.nunique())}


def predict(model, frame):
    """Unusable history falls back to the exact market; invalid prices stay NaN."""
    if (model.get('version') != VERSION or model.get('props') != list(PROPS)
            or model.get('features') != list(FEATURES)):
        raise ValueError('Market model schema/feature contract mismatch')
    beta = np.asarray(model['coefficients'], dtype=float)
    bound = float(model['max_logit_correction'])
    if beta.shape != (len(PROPS) * len(FEATURES),) or not np.isfinite(beta).all() or bound != MAX_CORRECTION:
        raise ValueError('Invalid market model coefficients/correction bound')
    X, valid = design(frame)
    q = pd.to_numeric(frame.nv_over, errors='coerce').to_numpy(float)
    priced = np.isfinite(q) & (q > 0) & (q < 1)
    correction = np.clip(X @ beta, -bound, bound)
    result = q.copy()
    result[~priced] = np.nan
    active = valid & priced
    result[active] = expit(logit(q[active]) + correction[active])
    return result


def score_live(model, lines, box, slate_date, now):
    """Score valid pre-tip FanDuel main offers, without changing any production score."""
    slate = pd.Timestamp(slate_date).strftime('%Y-%m-%d')
    if model.get('evaluation_status') != 'shadow_only' or model.get('trained_through', '') >= slate:
        raise ValueError('Shadow model must be explicitly isolated and trained before the offer date')
    frame = lines.copy()
    frame['prop'] = frame.prop.map({v: k for k, v in ODDS_PROPS.items()})
    frame['line'] = pd.to_numeric(frame.line, errors='coerce')
    frame['game_date'] = pd.to_datetime(frame.game_date, errors='coerce').dt.strftime('%Y-%m-%d')
    quote = pd.to_datetime(frame.snapshot_ts, utc=True, errors='coerce', format='mixed')
    tip = pd.to_datetime(frame.commence_time, utc=True, errors='coerce', format='mixed')
    frame['nv_over'], frame['dec_over'], frame['dec_under'] = paired_market(frame.over_price, frame.under_price)
    valid = (frame.prop.notna() & frame.game_date.eq(slate) & frame.line.ge(0) & frame.nv_over.notna()
             & tip.gt(now) & quote.le(now) & (now - quote).le(pd.Timedelta(hours=24))
             & tip.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d').eq(slate))
    frame = frame[valid].copy()
    if frame.empty:
        return frame
    frame['_quote'] = quote.loc[frame.index]
    frame['player_norm'] = frame.player.map(normalize_name)
    frame = frame.sort_values('_quote').drop_duplicates(['game_date', 'player_norm', 'prop'], keep='last')
    frame['feature_cutoff_date'] = min(slate, pd.Timestamp(now).tz_convert('America/Chicago').strftime('%Y-%m-%d'))
    frame = attach_history(frame.drop(columns='_quote'), box)
    frame['anchored_over'] = predict(model, frame)
    frame['history_fallback_to_market'] = ~design(frame)[1]
    frame['direction'] = 'UNDER'
    frame['shadow_lane'] = VERSION
    frame['tracking_only'] = True
    frame['published'] = False
    frame['side_price'] = frame.under_price
    frame['side_decimal_odds'] = frame.dec_under
    frame['model_side_prob'] = 1 - frame.anchored_over
    frame['no_vig_side_prob'] = 1 - frame.nv_over
    frame['ev_per_unit'] = frame.model_side_prob * frame.dec_under - 1
    frame['trained_through'] = model['trained_through']
    return frame
