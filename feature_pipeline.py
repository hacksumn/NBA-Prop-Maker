"""
Integrated Feature Pipeline
============================
Combines all feature engineering steps into a single modular pipeline.
Outputs model-ready data with all advanced features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
import warnings
warnings.filterwarnings('ignore')


TEAM_GAME_LOG_PATH = Path(__file__).resolve().parent / "data" / "nba_team_game_logs.csv"


def _season_from_game_date(date_value) -> str:
    """Infer NBA season string from a game date."""
    ts = pd.to_datetime(date_value, errors='coerce')
    if pd.isna(ts):
        return ""
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _ensure_nba_season_columns(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    """Attach season labels when the source frame only has game dates."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    if 'season' not in out.columns:
        out['season'] = out[date_col].apply(_season_from_game_date) if date_col in out.columns else ""
    else:
        if date_col in out.columns:
            inferred = out[date_col].apply(_season_from_game_date)
            out['season'] = out['season'].replace('', np.nan).fillna(inferred)
        out['season'] = out['season'].astype(str)
    return out


def _canonical_game_id(values: pd.Series) -> pd.Series:
    """Normalize NBA game identifiers without depending on CSV dtype inference."""
    numeric = pd.to_numeric(values, errors='coerce')
    return numeric.astype('Int64').astype('string')


def _build_team_allowed_context_source(df: pd.DataFrame) -> pd.DataFrame:
    """Build exact team-game opponent inputs, preferring NBA team game logs.

    Player box rows are a safe fallback for synthetic/unit-test frames only. If
    the local team-log source overlaps the requested games, it must cover every
    requested game; partial coverage fails closed instead of mixing definitions.
    """
    # Prediction-time feature frames contain one synthetic row for the upcoming
    # game.  It deliberately has no box-score outcome.  Only completed games
    # belong in the allowed-stat history or in the exact-log coverage contract.
    # These fields identify a completed player box.  Do not require player ORB:
    # the 2025-26 player feed has ORB missing, while the authoritative team log
    # used below has complete OREB and is exactly why that source exists.
    completed_stat_cols = [
        col for col in ['pts', 'trb', 'ast', 'fga', 'fta', 'tov']
        if col in df.columns
    ]
    if completed_stat_cols:
        completed_mask = df[completed_stat_cols].notna().all(axis=1)
    else:
        completed_mask = pd.Series(True, index=df.index)
    completed = df.loc[completed_mask].copy()

    if 'GAME_ID' in completed.columns and TEAM_GAME_LOG_PATH.exists():
        logs = pd.read_csv(TEAM_GAME_LOG_PATH, low_memory=False)
        required = {
            'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'PTS',
            'REB', 'AST', 'FGA', 'FTA', 'OREB', 'TOV', 'season',
        }
        missing = sorted(required - set(logs.columns))
        if missing:
            raise ValueError(f"NBA team-game log is missing required columns: {missing}")
        requested_ids = set(_canonical_game_id(completed['GAME_ID']).dropna())
        logs = logs.copy()
        logs['_game_id_key'] = _canonical_game_id(logs['GAME_ID'])
        available_ids = set(logs['_game_id_key'].dropna())
        overlap = requested_ids & available_ids
        if overlap:
            missing_ids = requested_ids - available_ids
            if missing_ids:
                raise ValueError(
                    "NBA team-game log partially covers the feature frame: "
                    f"{len(missing_ids)} of {len(requested_ids)} games are missing"
                )
            logs = logs.loc[logs['_game_id_key'].isin(requested_ids)].copy()
            if logs.duplicated(['_game_id_key', 'TEAM_ABBREVIATION']).any():
                raise ValueError("NBA team-game log has duplicate team-game keys")
            if not logs.groupby('_game_id_key').size().eq(2).all():
                raise ValueError("NBA team-game log must contain exactly two teams per game")
            numeric_cols = ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'OREB', 'TOV']
            logs[numeric_cols] = logs[numeric_cols].apply(pd.to_numeric, errors='coerce')
            if logs[numeric_cols].isna().any().any():
                nulls = logs[numeric_cols].isna().sum()
                raise ValueError(
                    "NBA team-game log has required stat nulls: "
                    f"{nulls[nulls.gt(0)].to_dict()}"
                )
            matchup_opp = logs['MATCHUP'].astype(str).str.extract(
                r'(?:vs\.?|@)\s*([A-Z]{3})\s*$', expand=False
            )
            if matchup_opp.isna().any():
                raise ValueError("Could not parse opponent from NBA team-game MATCHUP")
            team_agg = pd.DataFrame({
                'season': logs['season'].astype(str),
                'opp': matchup_opp,
                'game_date': pd.to_datetime(logs['GAME_DATE'], errors='coerce').dt.normalize(),
                'pts_allowed': logs['PTS'],
                'trb_allowed': logs['REB'],
                'ast_allowed': logs['AST'],
                'fga_allowed': logs['FGA'],
                'fta_allowed': logs['FTA'],
                'orb_allowed': logs['OREB'],
                'tov_allowed': logs['TOV'],
            })
            if team_agg['game_date'].isna().any():
                raise ValueError("NBA team-game log has invalid GAME_DATE values")
            return team_agg.sort_values(['season', 'opp', 'game_date']).reset_index(drop=True)

    required_player_cols = {
        'season', 'opp', 'game_date', 'pts', 'trb', 'ast', 'fga', 'fta', 'orb', 'tov',
    }
    missing = sorted(required_player_cols - set(completed.columns))
    if missing:
        raise ValueError(
            "Cannot build opponent context without exact team logs or complete player stats; "
            f"missing columns: {missing}"
        )
    stat_cols = ['pts', 'trb', 'ast', 'fga', 'fta', 'orb', 'tov']
    fallback_completed = df.loc[df[stat_cols].notna().all(axis=1)].copy()
    if fallback_completed.empty and not df.empty:
        raise ValueError("Player-box fallback has no completed rows with all required stats")
    if fallback_completed[stat_cols].isna().any().any():
        nulls = fallback_completed[stat_cols].isna().sum()
        raise ValueError(
            "Player-box fallback has required stat nulls and cannot estimate possessions: "
            f"{nulls[nulls.gt(0)].to_dict()}"
        )
    team_agg = (
        fallback_completed.groupby(['season', 'opp', 'game_date'], as_index=False)[stat_cols]
        .sum(min_count=1)
        .rename(columns={
            'pts': 'pts_allowed', 'trb': 'trb_allowed', 'ast': 'ast_allowed',
            'fga': 'fga_allowed', 'fta': 'fta_allowed', 'orb': 'orb_allowed',
            'tov': 'tov_allowed',
        })
    )
    return team_agg


def _repair_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct missing opponent/home fields from GAME_ID and matchup text."""
    out = df.copy()
    if not {'GAME_ID', 'team'}.issubset(out.columns):
        return out
    out['team'] = out['team'].astype(str).str.strip().str.upper()
    supplied_opp = out['opp'].astype('string').str.strip().str.upper()
    supplied_home = pd.to_numeric(out.get('is_home', pd.Series(np.nan, index=out.index)), errors='coerce')
    if not supplied_opp.isna().any() and not supplied_home.isna().any():
        out['opp'] = supplied_opp
        out['is_home'] = supplied_home.astype(int)
        return out
    pairs = out[['GAME_ID', 'team']].drop_duplicates().groupby('GAME_ID')['team'].agg(list)
    invalid = pairs[pairs.map(len).ne(2)]
    if not invalid.empty:
        raise ValueError(f"Cannot repair matchup columns: {len(invalid)} games do not have two teams")
    opponent = {}
    for game_id, teams in pairs.items():
        opponent[(game_id, teams[0])] = teams[1]
        opponent[(game_id, teams[1])] = teams[0]
    derived_opp = pd.Series(
        [opponent[(game_id, team)] for game_id, team in zip(out['GAME_ID'], out['team'])],
        index=out.index,
        dtype='string',
    )
    disagreement = supplied_opp.notna() & supplied_opp.ne(derived_opp)
    if disagreement.any():
        raise ValueError(f"opp disagrees with GAME_ID pairing on {int(disagreement.sum())} rows")
    out['opp'] = supplied_opp.fillna(derived_opp)

    matchup = out.get('matchup', pd.Series('', index=out.index)).astype(str)
    home_marker = matchup.str.contains(r'\bvs\.?', case=False, regex=True)
    away_marker = matchup.str.contains('@', regex=False)
    derived_home = home_marker.astype(int)
    disagreement = supplied_home.notna() & (home_marker | away_marker) & supplied_home.ne(derived_home)
    if disagreement.any():
        raise ValueError(f"is_home disagrees with matchup text on {int(disagreement.sum())} rows")
    derived_home = supplied_home.fillna(derived_home)
    if derived_home.isna().any() or (~(home_marker | away_marker) & supplied_home.isna()).any():
        raise ValueError("Cannot repair is_home without matchup text or a supplied value")
    out['is_home'] = derived_home.astype(int)
    return out


class FeaturePipeline:
    """
    Complete feature engineering pipeline for NBA player props.
    
    Stages:
    1. Base rolling features (L5, L10, L20)
    2. EWMA-weighted features
    3. Opponent context (defense, pace)
    4. Situational splits (home/away, defense tier, B2B)
    5. Role/usage trends
    6. Vegas intelligence
    """
    
    def __init__(self, 
                 ewma_span: int = 5,
                 split_window: int = 15,
                 min_games: int = 10):
        """
        Initialize the pipeline.
        
        Args:
            ewma_span: Span for EWMA calculations
            split_window: Window for situational splits
            min_games: Minimum games required for valid predictions
        """
        self.ewma_span = ewma_span
        self.split_window = split_window
        self.min_games = min_games
        self.feature_cols = []
        
    def fit_transform(self, df: pd.DataFrame, 
                      vegas_lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the complete feature pipeline.
        
        Args:
            df: Raw player game data
            vegas_lines: Optional Vegas lines data
            
        Returns:
            DataFrame with all features added
        """
        print("\n" + "="*70)
        print("INTEGRATED FEATURE PIPELINE")
        print("="*70)
        
        initial_cols = len(df.columns)
        
        # Validate required columns
        self._validate_input(df)

        # Ensure datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = _ensure_nba_season_columns(df)
        df = _repair_matchup_columns(df)
        df = df.sort_values(['player', 'game_date']).reset_index(drop=True)

        # Alias team column so team scoring trends work
        if 'team' in df.columns and 'TEAM_ABBREVIATION' not in df.columns:
            df['TEAM_ABBREVIATION'] = df['team']

        # Compute days_rest and is_b2b_second from game dates
        df['days_rest'] = (
            df.groupby('player')['game_date']
            .diff()
            .dt.days
            .fillna(3)
            .clip(0, 14)
        )
        df['is_b2b_second'] = (df['days_rest'] == 1).astype(int)
        
        # Stage 1: Base rolling features
        print("\n[Stage 1/6] Base rolling features...")
        df = self._create_base_rolling(df)
        
        # Stage 2: EWMA features
        print("\n[Stage 2/6] EWMA-weighted features...")
        df = self._create_ewma_features(df)
        
        # Stage 3: Opponent context
        print("\n[Stage 3/6] Opponent context features...")
        df = self._create_opponent_context(df)
        
        # Stage 4: Situational splits
        print("\n[Stage 4/6] Situational split features...")
        df = self._create_situational_splits(df)
        
        # Stage 5: Role/usage trends
        print("\n[Stage 5/6] Role/usage trends...")
        df = self._create_role_usage_features(df)

        # Stage 5b: Shot quality / scoring efficiency
        print("\n[Stage 5b/6] Shot quality and efficiency...")
        df = self._create_shot_quality_features(df)

        # Stage 6: Vegas intelligence
        print("\n[Stage 6/6] Vegas intelligence features...")
        df = self._create_vegas_features(df, vegas_lines)
        
        # Create composite/combination features
        print("\n[Bonus] Creating composite features...")
        df = self._create_composite_features(df)
        
        # Track feature columns
        self._catalog_features(df, initial_cols)
        
        # Summary
        new_cols = len(df.columns) - initial_cols
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"  Initial columns: {initial_cols}")
        print(f"  New columns: {new_cols}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Feature columns: {len(self.feature_cols)}")
        
        return df
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate required columns exist."""
        required = ['player', 'game_date', 'pts', 'trb', 'ast', 'mp', 'fga', 'opp']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _create_base_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base rolling average features."""
        grouped = df.groupby('player')
        season_grouped = df.groupby(['player', 'season'], dropna=False)

        stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fta', '3pa', 'tov', 'stl', 'blk']
        windows = [5, 10, 20]

        # Games played counter
        df['games_played'] = grouped.cumcount()
        df['season_games_played'] = season_grouped.cumcount()
        df['season_sample_reliability'] = (
            df['season_games_played'] / (df['season_games_played'] + 10.0)
        ).fillna(0.0).clip(0.0, 1.0)
        
        for stat in stats:
            if stat not in df.columns:
                continue
            
            for w in windows:
                df[f'{stat}_l{w}'] = grouped[stat].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
            df[f'{stat}_season_avg'] = season_grouped[stat].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
        
        # Std dev for main stats
        for stat in ['pts', 'trb', 'ast', 'mp']:
            if stat not in df.columns:
                continue
            df[f'{stat}_std_l10'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=2).std()
            )
        
        # Trend: L5/L20 ratio
        for stat in ['pts', 'trb', 'ast']:
            if f'{stat}_l5' in df.columns and f'{stat}_l20' in df.columns:
                df[f'{stat}_trend'] = df[f'{stat}_l5'] / df[f'{stat}_l20'].replace(0, np.nan)
        
        print(f"  Created rolling features for {len(stats)} stats")
        return df
    
    def _create_ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create EWMA-weighted features."""
        grouped = df.groupby('player')
        span = self.ewma_span
        season_reliability = df.get('season_sample_reliability', pd.Series(0.0, index=df.index)).fillna(0.0)
        
        stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fta', '3pa', 'stl', 'blk', 'tov']
        
        for stat in stats:
            if stat not in df.columns:
                continue
            df[f'{stat}_ewma{span}'] = grouped[stat].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=1, adjust=False).mean()
            )
        
        # Consistency (CV)
        for stat in ['pts', 'trb', 'ast', 'mp']:
            if f'{stat}_ewma{span}' in df.columns:
                df[f'{stat}_ewma_std'] = grouped[stat].transform(
                    lambda x: x.shift(1).ewm(span=span, min_periods=2, adjust=False).std()
                )
                df[f'{stat}_consistency'] = df[f'{stat}_ewma_std'] / df[f'{stat}_ewma{span}'].replace(0, np.nan)
                df[f'{stat}_consistency'] = df[f'{stat}_consistency'].fillna(1.0)

        # Blend short, medium, and season-long form into a single recency-aware anchor.
        for stat in stats:
            ewma_col = f'{stat}_ewma{span}'
            season_col = f'{stat}_season_avg'
            l5_col = f'{stat}_l5'
            l10_col = f'{stat}_l10'
            l20_col = f'{stat}_l20'
            if not all(col in df.columns for col in [ewma_col, l5_col, l10_col, l20_col]):
                continue

            recent_stack = (
                df[ewma_col] * 0.40 +
                df[l5_col] * 0.25 +
                df[l10_col] * 0.20 +
                df[l20_col] * 0.15
            )
            season_anchor = df[season_col] if season_col in df.columns else df[l20_col]
            season_anchor = season_anchor.fillna(df[l20_col]).fillna(recent_stack)
            season_share = 0.12 + 0.23 * season_reliability
            df[f'{stat}_recency_anchor'] = (
                recent_stack * (1.0 - season_share) +
                season_anchor * season_share
            )
            if season_col in df.columns:
                df[f'{stat}_form_vs_season'] = df[f'{stat}_recency_anchor'] - df[season_col]
        
        print(f"  Created EWMA features (span={span})")
        return df
    
    def _create_opponent_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create strict-pregame, pace-normalized opponent context features."""
        team_agg = _build_team_allowed_context_source(df)

        # Dean Oliver team possessions; the legacy version omitted ORB.
        team_agg['poss_est'] = (
            team_agg['fga_allowed'] + 0.44 * team_agg['fta_allowed']
            - team_agg['orb_allowed'] + team_agg['tov_allowed']
        ).clip(lower=1.0)

        # Inclusive postgame states are joined only to later dates below.
        team_agg = team_agg.sort_values(['season', 'opp', 'game_date'])
        team_grouped = team_agg.groupby(['season', 'opp'], group_keys=False)
        raw_stats = ['pts_allowed', 'trb_allowed', 'ast_allowed', 'fga_allowed', 'poss_est']
        for stat in raw_stats:
            team_agg[f'{stat}_post_l10'] = team_grouped[stat].transform(
                lambda x: x.rolling(10, min_periods=3).mean()
            )
        
        # Build a 30-team state panel per date before ranking. The legacy code
        # ranked repeated player rows on the active slate and reversed the rank.
        state_cols = [f'{stat}_post_l10' for stat in raw_stats]
        states = []
        for season in sorted(team_agg['season'].astype(str).unique()):
            season_mask = team_agg['season'].astype(str).eq(season)
            # Include target prediction dates as as-of lookup dates.  They do
            # not enter the rolling history; merge_asof below uses only prior
            # completed games (allow_exact_matches=False).
            requested_dates = pd.to_datetime(
                df.loc[df['season'].astype(str).eq(season), 'game_date'],
                errors='coerce',
            ).dropna().dt.normalize().unique()
            history_dates = team_agg.loc[season_mask, 'game_date'].unique()
            calendar = pd.DataFrame({
                'game_date': sorted(set(history_dates) | set(requested_dates))
            })
            for opponent in sorted(team_agg.loc[season_mask, 'opp'].dropna().unique()):
                history = team_agg.loc[
                    season_mask & team_agg['opp'].eq(opponent), ['game_date', *state_cols]
                ].sort_values('game_date')
                state = pd.merge_asof(
                    calendar, history, on='game_date', direction='backward',
                    allow_exact_matches=False,
                )
                state['season'] = season
                state['opp'] = opponent
                states.append(state)
        context = pd.concat(states, ignore_index=True).rename(columns={
            'pts_allowed_post_l10': 'opp_pts_allowed_l10',
            'trb_allowed_post_l10': 'opp_trb_allowed_l10',
            'ast_allowed_post_l10': 'opp_ast_allowed_l10',
            'fga_allowed_post_l10': 'opp_fga_allowed_l10',
            'poss_est_post_l10': 'opp_poss_est_l10',
        })
        league_pace = context.groupby('game_date')['opp_poss_est_l10'].transform('mean')
        context['opp_pace_factor'] = context['opp_poss_est_l10'] / league_pace
        for stat in ['pts', 'trb', 'ast', 'fga']:
            per100 = f'opp_{stat}_allowed_per100_l10'
            context[per100] = 100.0 * context[f'opp_{stat}_allowed_l10'] / context['opp_poss_est_l10']
            context[f'opp_{stat}_allowed_rank'] = context.groupby('game_date')[per100].rank(
                ascending=True, method='min'
            )
        context['opp_def_tier'] = pd.cut(
            context['opp_pts_allowed_rank'],
            bins=[0, 6, 12, 18, 24, 30],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)

        merge_cols = [
            'season', 'opp', 'game_date', 'opp_pts_allowed_l10', 'opp_trb_allowed_l10',
            'opp_ast_allowed_l10', 'opp_fga_allowed_l10', 'opp_pace_factor',
            'opp_pts_allowed_per100_l10', 'opp_trb_allowed_per100_l10',
            'opp_ast_allowed_per100_l10', 'opp_fga_allowed_per100_l10',
            'opp_pts_allowed_rank', 'opp_trb_allowed_rank',
            'opp_ast_allowed_rank', 'opp_fga_allowed_rank', 'opp_def_tier',
        ]
        df = df.merge(context[merge_cols], on=['season', 'opp', 'game_date'], how='left', suffixes=('', '_new'))
        for col in merge_cols[3:]:
            if f'{col}_new' in df.columns:
                df[col] = df[f'{col}_new']
                df = df.drop(columns=[f'{col}_new'])
        df['opp_pace_factor'] = df['opp_pace_factor'].fillna(1.0)
        df['opp_def_tier'] = df['opp_def_tier'].fillna(3)
        print("  Created opponent context features")
        return df
    
    def _create_situational_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create situational split features."""
        window = self.split_window
        
        # Home/Away splits
        for stat in ['pts', 'trb', 'ast']:
            # Home
            df[f'{stat}_home_temp'] = np.where(df['is_home'] == 1, df[stat], np.nan)
            df[f'{stat}_home_l{window}'] = df.groupby('player')[f'{stat}_home_temp'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).mean()
            )
            
            # Away
            df[f'{stat}_away_temp'] = np.where(df['is_home'] == 0, df[stat], np.nan)
            df[f'{stat}_away_l{window}'] = df.groupby('player')[f'{stat}_away_temp'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).mean()
            )
            
            # Differential
            df[f'{stat}_home_away_diff'] = df[f'{stat}_home_l{window}'] - df[f'{stat}_away_l{window}']
            
            # Situational average
            df[f'{stat}_situational_avg'] = np.where(
                df['is_home'] == 1,
                df[f'{stat}_home_l{window}'],
                df[f'{stat}_away_l{window}']
            )
            df[f'{stat}_situational_avg'] = df[f'{stat}_situational_avg'].fillna(df[f'{stat}_l10'])
        
        # vs Defense tier
        for stat in ['pts', 'trb', 'ast']:
            df[f'{stat}_vs_good_def_temp'] = np.where(df['opp_def_tier'] <= 2, df[stat], np.nan)
            df[f'{stat}_vs_good_def'] = df.groupby('player')[f'{stat}_vs_good_def_temp'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).mean()
            )
            
            df[f'{stat}_vs_bad_def_temp'] = np.where(df['opp_def_tier'] >= 4, df[stat], np.nan)
            df[f'{stat}_vs_bad_def'] = df.groupby('player')[f'{stat}_vs_bad_def_temp'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).mean()
            )
            
            df[f'{stat}_defense_adj'] = np.where(
                df['opp_def_tier'] <= 2,
                df[f'{stat}_vs_good_def'],
                np.where(df['opp_def_tier'] >= 4, df[f'{stat}_vs_bad_def'], df[f'{stat}_l10'])
            )
            df[f'{stat}_defense_adj'] = df[f'{stat}_defense_adj'].fillna(df[f'{stat}_l10'])
        
        # B2B splits
        if 'is_b2b_second' in df.columns:
            for stat in ['pts', 'trb', 'ast', 'mp']:
                df[f'{stat}_b2b_temp'] = np.where(df['is_b2b_second'] == 1, df[stat], np.nan)
                df[f'{stat}_on_b2b'] = df.groupby('player')[f'{stat}_b2b_temp'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=2).mean()
                )
                
                rested_mask = df['days_rest'].fillna(2) >= 2
                df[f'{stat}_rested_temp'] = np.where(rested_mask, df[stat], np.nan)
                df[f'{stat}_rested'] = df.groupby('player')[f'{stat}_rested_temp'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=2).mean()
                )
                
                df[f'{stat}_b2b_penalty'] = df[f'{stat}_on_b2b'] - df[f'{stat}_rested']
        
        # Clean temp columns
        temp_cols = [c for c in df.columns if '_temp' in c]
        df = df.drop(columns=temp_cols, errors='ignore')
        
        print(f"  Created situational splits (window={window})")
        return df
    
    def _create_role_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create role and usage trend features."""
        grouped = df.groupby('player')
        
        # Minutes trend
        df['mp_l3'] = grouped['mp'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['mp_trend_ratio'] = df['mp_l3'] / df['mp_l10'].replace(0, np.nan)
        df['mp_trend_ratio'] = df['mp_trend_ratio'].fillna(1.0)
        
        df['role_expanding'] = (df['mp_trend_ratio'] > 1.1).astype(int)
        df['role_shrinking'] = (df['mp_trend_ratio'] < 0.9).astype(int)
        
        # Usage proxy
        df['usage_proxy'] = (df['fga'] + df['fta'] * 0.44 + df['tov']) / df['mp'].replace(0, np.nan)
        df['usage_l5'] = grouped['usage_proxy'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df['usage_l10'] = grouped['usage_proxy'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        df['usage_trend'] = df['usage_l5'] / df['usage_l10'].replace(0, np.nan)
        df['usage_trend'] = df['usage_trend'].fillna(1.0)
        
        # Starter inference
        df['likely_starter'] = (df['mp_l10'] >= 28).astype(int)
        
        # FGA trend
        df['fga_l3'] = grouped['fga'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['fga_trend'] = df['fga_l3'] / df['fga_l10'].replace(0, np.nan)
        df['fga_trend'] = df['fga_trend'].fillna(1.0)
        
        print(f"  Created role/usage features")
        return df

    def _create_shot_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create shot-volume and shot-efficiency features for PTS modeling."""
        grouped = df.groupby('player')

        for pct_stat in ['fg_pct', '3p_pct', 'ft_pct']:
            if pct_stat not in df.columns:
                continue
            df[pct_stat] = pd.to_numeric(df[pct_stat], errors='coerce')
            for window, min_periods in [(5, 2), (10, 3), (20, 5)]:
                df[f'{pct_stat}_l{window}'] = grouped[pct_stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()
                )
            df[f'{pct_stat}_luck'] = df[f'{pct_stat}_l5'] - df[f'{pct_stat}_l20']

        if all(col in df.columns for col in ['pts', 'fga', 'fta']):
            df['_ts_raw'] = pd.to_numeric(df['pts'], errors='coerce') / (
                2.0 * (
                    pd.to_numeric(df['fga'], errors='coerce') +
                    0.44 * pd.to_numeric(df['fta'], errors='coerce') +
                    1e-6
                )
            ).clip(lower=0.01)
            df['ts_pct_l10'] = df.groupby('player')['_ts_raw'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            ts_pct_l5 = df.groupby('player')['_ts_raw'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).mean()
            )
            df['ts_pct_trend'] = ts_pct_l5 / df['ts_pct_l10'].replace(0, np.nan)
            df['ts_pct_trend'] = df['ts_pct_trend'].fillna(1.0)
            df = df.drop(columns=['_ts_raw'], errors='ignore')

        for window in [5, 10, 20]:
            pts_col = f'pts_l{window}'
            fga_col = f'fga_l{window}'
            fta_col = f'fta_l{window}'
            fg_pct_col = f'fg_pct_l{window}'
            if pts_col in df.columns and fga_col in df.columns:
                df[f'points_per_fga_l{window}'] = (
                    pd.to_numeric(df[pts_col], errors='coerce') /
                    pd.to_numeric(df[fga_col], errors='coerce').replace(0, np.nan)
                )
            if fga_col in df.columns and fta_col in df.columns:
                df[f'scoring_opps_l{window}'] = (
                    pd.to_numeric(df[fga_col], errors='coerce') +
                    0.44 * pd.to_numeric(df[fta_col], errors='coerce')
                )
            if fga_col in df.columns:
                fg_proxy = pd.to_numeric(
                    df.get(fg_pct_col, pd.Series(np.nan, index=df.index)),
                    errors='coerce',
                )
                df[f'shot_volume_quality_l{window}'] = (
                    pd.to_numeric(df[fga_col], errors='coerce') * fg_proxy
                )

        if {'points_per_fga_l5', 'points_per_fga_l20'}.issubset(df.columns):
            df['points_per_fga_trend'] = (
                df['points_per_fga_l5'] / df['points_per_fga_l20'].replace(0, np.nan)
            ).fillna(1.0)
        if {'scoring_opps_l5', 'scoring_opps_l20'}.issubset(df.columns):
            df['scoring_opps_trend'] = (
                df['scoring_opps_l5'] / df['scoring_opps_l20'].replace(0, np.nan)
            ).fillna(1.0)
        if {'shot_volume_quality_l5', 'shot_volume_quality_l20'}.issubset(df.columns):
            df['shot_volume_quality_trend'] = (
                df['shot_volume_quality_l5'] / df['shot_volume_quality_l20'].replace(0, np.nan)
            ).fillna(1.0)
        if {'usage_l10', 'fga_l10'}.issubset(df.columns):
            df['usage_fga_interaction_l10'] = (
                pd.to_numeric(df['usage_l10'], errors='coerce').fillna(0.0) *
                pd.to_numeric(df['fga_l10'], errors='coerce').fillna(0.0)
            )

        print("  Created shot-quality features")
        return df
    
    def _create_vegas_features(self, df: pd.DataFrame, 
                               vegas_lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create Vegas intelligence features."""
        
        # Synthetic lines from averages
        for stat in ['pts', 'trb', 'ast']:
            synthetic_source = (
                df.get(f'{stat}_season_avg',
                df.get(f'{stat}_l20',
                df.get(f'{stat}_l10')))
            )
            df[f'{stat}_synthetic_line'] = synthetic_source
            pred_col = f'{stat}_recency_anchor' if f'{stat}_recency_anchor' in df.columns else (
                f'{stat}_ewma5' if f'{stat}_ewma5' in df.columns else f'{stat}_l5'
            )
            
            if pred_col in df.columns:
                df[f'{stat}_edge_vs_avg'] = df[pred_col] - df[f'{stat}_synthetic_line']
                df[f'{stat}_vegas_slow'] = (abs(df[f'{stat}_edge_vs_avg']) > df[f'{stat}_l10'] * 0.1).astype(int)
                df[f'{stat}_pct_diff'] = df[f'{stat}_edge_vs_avg'] / df[f'{stat}_synthetic_line'].replace(0, np.nan)
                df[f'{stat}_pct_diff'] = df[f'{stat}_pct_diff'].fillna(0)
        
        # Team scoring trends
        team_key = None
        if 'TEAM_ABBREVIATION' in df.columns:
            team_key = 'TEAM_ABBREVIATION'
        elif 'team' in df.columns:
            team_key = 'team'

        if team_key is not None:
            df = df.drop(columns=['team_pts_l5', 'team_pts_l10', 'team_scoring_trend'], errors='ignore')
            team_pts = df.groupby([team_key, 'game_date'])['pts'].sum().reset_index()
            team_pts.columns = [team_key, 'game_date', 'team_total_pts']
            team_pts = team_pts.sort_values([team_key, 'game_date'])
            
            team_grouped = team_pts.groupby(team_key)
            team_pts['team_pts_l5'] = team_grouped['team_total_pts'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            team_pts['team_pts_l10'] = team_grouped['team_total_pts'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean()
            )
            team_pts['team_scoring_trend'] = team_pts['team_pts_l5'] / team_pts['team_pts_l10'].replace(0, np.nan)
            
            df = df.merge(
                team_pts[[team_key, 'game_date', 'team_pts_l5', 'team_pts_l10', 'team_scoring_trend']],
                on=[team_key, 'game_date'],
                how='left'
            )
        
        if 'team_scoring_trend' not in df.columns:
            df['team_scoring_trend'] = 1.0
        else:
            df['team_scoring_trend'] = df['team_scoring_trend'].fillna(1.0)
        df['high_scoring_game'] = (df['team_scoring_trend'] > 1.05).astype(int)
        
        # Game environment score
        pace     = df['opp_pace_factor']      if 'opp_pace_factor'      in df.columns else pd.Series(1.0,  index=df.index)
        scoring  = df['team_scoring_trend']   if 'team_scoring_trend'   in df.columns else pd.Series(1.0,  index=df.index)
        def_rank = df['opp_pts_allowed_rank'] if 'opp_pts_allowed_rank' in df.columns else pd.Series(15.0, index=df.index)

        df['game_environment_score'] = (
            pace * 0.4 +
            scoring * 0.3 +
            (30 - def_rank.fillna(15)) / 30 * 0.3
        )

        # Normalize using fixed bounds (not dataset min/max which leaks future data).
        # Components: pace~0.4, scoring~0.3, defense 0-0.3 → range roughly [0.4, 1.0]
        df['game_environment_score'] = ((df['game_environment_score'] - 0.4) / 0.6).clip(0, 1)
        
        print(f"  Created Vegas intelligence features")
        return df
    
    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite/interaction features."""
        
        # Best prediction estimate (weighted combination)
        for stat in ['pts', 'trb', 'ast']:
            components = []
            weights = []
            
            if f'{stat}_recency_anchor' in df.columns:
                components.append(df[f'{stat}_recency_anchor'])
                weights.append(0.40)
            elif f'{stat}_ewma5' in df.columns:
                components.append(df[f'{stat}_ewma5'])
                weights.append(0.35)
            
            if f'{stat}_defense_adj' in df.columns:
                components.append(df[f'{stat}_defense_adj'])
                weights.append(0.20)
            
            if f'{stat}_situational_avg' in df.columns:
                components.append(df[f'{stat}_situational_avg'])
                weights.append(0.15)

            if f'{stat}_season_avg' in df.columns:
                components.append(df[f'{stat}_season_avg'].fillna(df.get(f'{stat}_l20')))
                weights.append(0.15)

            if f'{stat}_l20' in df.columns:
                components.append(df[f'{stat}_l20'])
                weights.append(0.10)
            
            if components:
                total_weight = sum(weights)
                df[f'{stat}_best_estimate'] = sum(c * w for c, w in zip(components, weights)) / total_weight
        
        # Confidence score (inverse of consistency * games played factor)
        for stat in ['pts', 'trb', 'ast']:
            if f'{stat}_consistency' in df.columns:
                consistency_penalty = df[f'{stat}_consistency'].clip(0.2, 2.0)
                games_factor = (df['games_played'] / 20).clip(0, 1)
                sample_factor = 0.6 + 0.4 * df.get('season_sample_reliability', pd.Series(0.0, index=df.index)).fillna(0.0)
                df[f'{stat}_confidence'] = (1 / consistency_penalty) * games_factor * sample_factor
                df[f'{stat}_confidence'] = df[f'{stat}_confidence'].clip(0, 1)
        
        # Risk factor (high variance + bad matchup + B2B)
        b2b         = df['is_b2b_second']   if 'is_b2b_second'   in df.columns else pd.Series(0,   index=df.index)
        consistency = df['pts_consistency'] if 'pts_consistency' in df.columns else pd.Series(0.5, index=df.index)
        def_tier    = df['opp_def_tier']    if 'opp_def_tier'    in df.columns else pd.Series(3.0, index=df.index)

        df['risk_factor'] = (
            b2b * 0.3 +
            (consistency / 2) * 0.4 +
            (def_tier / 5) * 0.3
        )
        df['risk_factor'] = df['risk_factor'].clip(0, 1)
        
        print(f"  Created composite features")
        return df
    
    def _catalog_features(self, df: pd.DataFrame, initial_cols: int):
        """Catalog all feature columns."""
        all_cols = df.columns.tolist()
        
        # Identify feature columns (exclude identifiers and targets)
        non_features = ['player', 'game_date', 'game_id', 'matchup_id', 'opp', 'tm', 
                        'TEAM_ABBREVIATION', 'TEAM_NAME', 'PLAYER_ID', 'TEAM_ID',
                        'SEASON_YEAR', 'GAME_ID', 'matchup_raw', 'WL', 'NICKNAME',
                        'MIN_SEC', 'season', 'res']
        target_cols = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'fg', 'fga', 
                       '3p', '3pa', 'ft', 'fta', 'mp', 'OREB', 'DREB', 'orb', 'drb',
                       'PF', 'PFD', 'plus_minus', 'won', 'fg_pct', '3p_pct', 'ft_pct']
        
        self.feature_cols = [c for c in all_cols 
                            if c not in non_features 
                            and c not in target_cols]
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_cols
    
    def save_feature_list(self, path: str):
        """Save feature column list to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'feature_columns': self.feature_cols,
                'count': len(self.feature_cols)
            }, f, indent=2)
        print(f"Feature list saved to: {path}")


def main():
    """Run the integrated pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run feature pipeline')
    parser.add_argument('--input', type=str, default=r'C:\Users\jakep\Downloads\Fresh_Start_NBA\data\nba_data.csv',
                        help='Input CSV file')
    parser.add_argument('--output', type=str, default=r'C:\Users\jakep\Downloads\Fresh_Start_NBA\data\model_data_advanced.csv',
                        help='Output CSV file')
    parser.add_argument('--features-json', type=str, default=r'C:\Users\jakep\Downloads\Fresh_Start_NBA\feature_cols_advanced.json',
                        help='Feature columns JSON file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Run pipeline
    pipeline = FeaturePipeline(ewma_span=5, split_window=15, min_games=10)
    df = pipeline.fit_transform(df)
    
    # Filter for model readiness (enough games played)
    df_filtered = df[df['games_played'] >= 10].copy()
    print(f"\nFiltered to {len(df_filtered):,} rows with 10+ games played")
    
    # Save outputs
    df_filtered.to_csv(args.output, index=False)
    print(f"Saved model data to: {args.output}")
    
    pipeline.save_feature_list(args.features_json)
    
    # Print feature summary
    print(f"\n{'='*70}")
    print("FEATURE SUMMARY")
    print("="*70)
    features = pipeline.get_feature_columns()
    print(f"Total features: {len(features)}")
    
    # Group by category
    categories = {
        'Rolling': [f for f in features if '_l5' in f or '_l10' in f or '_l20' in f],
        'EWMA': [f for f in features if 'ewma' in f],
        'Situational': [f for f in features if 'home_' in f or 'away_' in f or 'vs_' in f or 'b2b' in f or 'rested' in f],
        'Opponent': [f for f in features if 'opp_' in f],
        'Usage/Role': [f for f in features if 'usage' in f or 'role_' in f or 'starter' in f or 'trend' in f],
        'Vegas/Composite': [f for f in features if 'vegas' in f or 'environment' in f or 'confidence' in f or 'risk' in f or 'best_estimate' in f]
    }
    
    for cat, cols in categories.items():
        print(f"\n{cat}: {len(cols)} features")
        for col in cols[:5]:
            print(f"  - {col}")
        if len(cols) > 5:
            print(f"  ... and {len(cols) - 5} more")


if __name__ == "__main__":
    main()
