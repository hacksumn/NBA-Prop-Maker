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
        
        stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fta', '3pa', 'tov', 'stl', 'blk']
        windows = [5, 10, 20]
        
        # Games played counter
        df['games_played'] = grouped.cumcount()
        
        for stat in stats:
            if stat not in df.columns:
                continue
            shifted = grouped[stat].shift(1)
            
            for w in windows:
                df[f'{stat}_l{w}'] = grouped[stat].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
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
        
        print(f"  Created EWMA features (span={span})")
        return df
    
    def _create_opponent_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced opponent context features."""
        
        # Team-level aggregations
        team_agg = df.groupby(['opp', 'game_date']).agg({
            'pts': 'sum',
            'trb': 'sum',
            'ast': 'sum',
            'fga': 'sum',
            'fta': 'sum',
            'tov': 'sum'
        }).reset_index()
        
        team_agg.columns = ['opp', 'game_date', 'pts_allowed', 'trb_allowed', 
                           'ast_allowed', 'fga_allowed', 'fta_allowed', 'tov_allowed']
        
        # Estimate possessions (pace proxy)
        team_agg['poss_est'] = team_agg['fga_allowed'] + 0.44 * team_agg['fta_allowed'] + team_agg['tov_allowed']
        
        # Rolling opponent stats
        team_agg = team_agg.sort_values(['opp', 'game_date'])
        team_grouped = team_agg.groupby('opp')
        
        for stat in ['pts_allowed', 'trb_allowed', 'ast_allowed', 'fga_allowed', 'poss_est']:
            team_agg[f'{stat}_l10'] = team_grouped[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
        
        # Pace factor — use fixed constant to avoid leaking future-game data
        # into the normalization denominator (NBA average ~240 team possessions/game)
        league_avg_pace = 240
        team_agg['pace_factor'] = team_agg['poss_est_l10'] / league_avg_pace
        
        # Rename for merge
        merge_cols = ['opp', 'game_date', 'pts_allowed_l10', 'trb_allowed_l10', 
                      'ast_allowed_l10', 'fga_allowed_l10', 'pace_factor']
        team_agg = team_agg[merge_cols].copy()
        team_agg.columns = ['opp', 'game_date', 'opp_pts_allowed_l10', 'opp_trb_allowed_l10',
                           'opp_ast_allowed_l10', 'opp_fga_allowed_l10', 'opp_pace_factor']
        
        # Merge to player data
        df = df.merge(team_agg, on=['opp', 'game_date'], how='left', suffixes=('', '_new'))
        
        # Handle duplicate columns
        for col in ['opp_pts_allowed_l10', 'opp_trb_allowed_l10', 'opp_ast_allowed_l10', 'opp_fga_allowed_l10']:
            if f'{col}_new' in df.columns:
                df[col] = df[col].fillna(df[f'{col}_new'])
                df = df.drop(columns=[f'{col}_new'])
        
        # Defense tier
        df['opp_pts_allowed_rank'] = df.groupby('game_date')['opp_pts_allowed_l10'].rank(ascending=False)
        df['opp_trb_allowed_rank'] = df.groupby('game_date')['opp_trb_allowed_l10'].rank(ascending=False)
        df['opp_ast_allowed_rank'] = df.groupby('game_date')['opp_ast_allowed_l10'].rank(ascending=False)
        
        df['opp_def_tier'] = pd.cut(
            df['opp_pts_allowed_rank'].fillna(15),
            bins=[0, 6, 12, 18, 24, 30],
            labels=[1, 2, 3, 4, 5]
        ).astype(float)
        
        # Fill missing
        df['opp_pace_factor'] = df['opp_pace_factor'].fillna(1.0)
        df['opp_def_tier'] = df['opp_def_tier'].fillna(3)
        
        print(f"  Created opponent context features")
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
    
    def _create_vegas_features(self, df: pd.DataFrame, 
                               vegas_lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create Vegas intelligence features."""
        
        # Synthetic lines from averages
        for stat in ['pts', 'trb', 'ast']:
            df[f'{stat}_synthetic_line'] = df[f'{stat}_l10']
            pred_col = f'{stat}_ewma5' if f'{stat}_ewma5' in df.columns else f'{stat}_l5'
            
            if pred_col in df.columns:
                df[f'{stat}_edge_vs_avg'] = df[pred_col] - df[f'{stat}_synthetic_line']
                df[f'{stat}_vegas_slow'] = (abs(df[f'{stat}_edge_vs_avg']) > df[f'{stat}_l10'] * 0.1).astype(int)
                df[f'{stat}_pct_diff'] = df[f'{stat}_edge_vs_avg'] / df[f'{stat}_synthetic_line'].replace(0, np.nan)
                df[f'{stat}_pct_diff'] = df[f'{stat}_pct_diff'].fillna(0)
        
        # Team scoring trends
        if 'TEAM_ABBREVIATION' in df.columns:
            team_pts = df.groupby(['TEAM_ABBREVIATION', 'game_date'])['pts'].sum().reset_index()
            team_pts.columns = ['TEAM_ABBREVIATION', 'game_date', 'team_total_pts']
            team_pts = team_pts.sort_values(['TEAM_ABBREVIATION', 'game_date'])
            
            team_grouped = team_pts.groupby('TEAM_ABBREVIATION')
            team_pts['team_pts_l5'] = team_grouped['team_total_pts'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            team_pts['team_pts_l10'] = team_grouped['team_total_pts'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean()
            )
            team_pts['team_scoring_trend'] = team_pts['team_pts_l5'] / team_pts['team_pts_l10'].replace(0, np.nan)
            
            df = df.merge(
                team_pts[['TEAM_ABBREVIATION', 'game_date', 'team_pts_l5', 'team_pts_l10', 'team_scoring_trend']],
                on=['TEAM_ABBREVIATION', 'game_date'],
                how='left'
            )
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
            
            if f'{stat}_ewma5' in df.columns:
                components.append(df[f'{stat}_ewma5'])
                weights.append(0.4)
            
            if f'{stat}_defense_adj' in df.columns:
                components.append(df[f'{stat}_defense_adj'])
                weights.append(0.3)
            
            if f'{stat}_situational_avg' in df.columns:
                components.append(df[f'{stat}_situational_avg'])
                weights.append(0.3)
            
            if components:
                total_weight = sum(weights)
                df[f'{stat}_best_estimate'] = sum(c * w for c, w in zip(components, weights)) / total_weight
        
        # Confidence score (inverse of consistency * games played factor)
        for stat in ['pts', 'trb', 'ast']:
            if f'{stat}_consistency' in df.columns:
                consistency_penalty = df[f'{stat}_consistency'].clip(0.2, 2.0)
                games_factor = (df['games_played'] / 20).clip(0, 1)
                df[f'{stat}_confidence'] = (1 / consistency_penalty) * games_factor
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
