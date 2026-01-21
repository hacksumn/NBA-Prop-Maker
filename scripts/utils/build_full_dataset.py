"""
Build Complete Dataset from Full Seasons
NBA Player Props System

This script processes the full season data directly, adding all necessary features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*60)
print("BUILD COMPLETE DATASET FROM FULL SEASONS")
print("="*60)

# Load full season data
print("\n[1/6] Loading full season data...")
df = pd.read_csv('data/database_full_seasons.csv', parse_dates=['game_date'])
print(f"  ✓ Loaded {len(df):,} rows")
print(f"  ✓ Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Sort by player and date
df = df.sort_values(['player', 'game_date']).reset_index(drop=True)

# Add rolling features
print("\n[2/6] Calculating rolling features...")

grouped = df.groupby('player')

# Games played counter
df['games_played'] = grouped.cumcount()

# Rolling stats
rolling_stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fg_pct', '3pa', '3p_pct', 'fta', 'ft_pct', 'tov', 'stl', 'blk']

for stat in rolling_stats:
    # Shift to prevent leakage
    df[f'{stat}_shifted'] = grouped[stat].shift(1)
    
    # Rolling averages
    df[f'{stat}_l5'] = grouped[f'{stat}_shifted'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df[f'{stat}_l10'] = grouped[f'{stat}_shifted'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df[f'{stat}_l20'] = grouped[f'{stat}_shifted'].transform(lambda x: x.rolling(20, min_periods=1).mean())

# Rolling std dev for volatility
for stat in ['pts', 'trb', 'ast', 'mp']:
    df[f'{stat}_std_l10'] = grouped[f'{stat}_shifted'].transform(lambda x: x.rolling(10, min_periods=2).std())

print(f"  ✓ Rolling features added")

# Add opponent features
print("\n[3/6] Adding opponent defense features...")

# Create team-level stats
team_stats = df.groupby(['opp', 'game_date']).agg({
    'pts': 'mean',
    'trb': 'mean',
    'ast': 'mean',
    'fga': 'mean'
}).reset_index()

team_stats = team_stats.rename(columns={
    'pts': 'pts_allowed',
    'trb': 'trb_allowed',
    'ast': 'ast_allowed',
    'fga': 'fga_allowed'
})

# Rolling team defense stats
team_stats = team_stats.sort_values(['opp', 'game_date'])
team_grouped = team_stats.groupby('opp')

for stat in ['pts_allowed', 'trb_allowed', 'ast_allowed', 'fga_allowed']:
    team_stats[f'{stat}_shifted'] = team_grouped[stat].shift(1)
    team_stats[f'{stat}_l10'] = team_grouped[f'{stat}_shifted'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Merge back to player data
df = df.merge(
    team_stats[['opp', 'game_date', 'pts_allowed_l10', 'trb_allowed_l10', 'ast_allowed_l10', 'fga_allowed_l10']],
    on=['opp', 'game_date'],
    how='left',
    suffixes=('', '_opp')
)

# Rename to match expected schema
if 'pts_allowed_l10_opp' in df.columns:
    df = df.rename(columns={
        'pts_allowed_l10_opp': 'opp_pts_allowed_l10',
        'trb_allowed_l10_opp': 'opp_trb_allowed_l10',
        'ast_allowed_l10_opp': 'opp_ast_allowed_l10',
        'fga_allowed_l10_opp': 'opp_fga_allowed_l10'
    })
else:
    df = df.rename(columns={
        'pts_allowed_l10': 'opp_pts_allowed_l10',
        'trb_allowed_l10': 'opp_trb_allowed_l10',
        'ast_allowed_l10': 'opp_ast_allowed_l10',
        'fga_allowed_l10': 'opp_fga_allowed_l10'
    })

# Add opponent ranks
df['opp_pts_allowed_rank'] = df.groupby('game_date')['opp_pts_allowed_l10'].rank(ascending=False, method='dense')
df['opp_trb_allowed_rank'] = df.groupby('game_date')['opp_trb_allowed_l10'].rank(ascending=False, method='dense')
df['opp_ast_allowed_rank'] = df.groupby('game_date')['opp_ast_allowed_l10'].rank(ascending=False, method='dense')

# Fill NaN
df['opp_pts_allowed_l10'] = df['opp_pts_allowed_l10'].fillna(df['opp_pts_allowed_l10'].mean())
df['opp_trb_allowed_l10'] = df['opp_trb_allowed_l10'].fillna(df['opp_trb_allowed_l10'].mean())
df['opp_ast_allowed_l10'] = df['opp_ast_allowed_l10'].fillna(df['opp_ast_allowed_l10'].mean())
df['opp_fga_allowed_l10'] = df['opp_fga_allowed_l10'].fillna(df['opp_fga_allowed_l10'].mean())

df['opp_pts_allowed_rank'] = df['opp_pts_allowed_rank'].fillna(15)
df['opp_trb_allowed_rank'] = df['opp_trb_allowed_rank'].fillna(15)
df['opp_ast_allowed_rank'] = df['opp_ast_allowed_rank'].fillna(15)

print(f"  ✓ Opponent features added")

# Add schedule/fatigue features
print("\n[4/6] Adding schedule/fatigue features...")

df['days_rest'] = grouped['game_date'].diff().dt.days - 1
df['days_rest'] = df['days_rest'].fillna(2).clip(lower=0, upper=7)

# Back-to-back games
df['is_b2b_second'] = (df['days_rest'] == 0).astype(int)

# High minutes in previous game
df['high_min_prev'] = (grouped['mp'].shift(1) > 35).astype(int).fillna(0)

# Games in last 7 days - simplified approach
df['games_last_7d'] = 2  # Default estimate

# Team pace (estimate from FGA)
df['opp_tm_pts_l10'] = df['opp_pts_allowed_l10']
df['opp_tm_fga_l10'] = df['opp_fga_allowed_l10']

print(f"  ✓ Schedule features added")

# Add matchup ID
print("\n[5/6] Adding matchup identifiers...")
df['matchup_id'] = df['player'] + '_vs_' + df['opp']

print(f"  ✓ Matchup IDs added")

# Save
print("\n[6/6] Saving dataset...")
output_file = 'data/player_games_schedule.csv'
df.to_csv(output_file, index=False)

print(f"  ✓ Saved to {output_file}")
print(f"  ✓ {len(df):,} rows, {len(df.columns)} columns")

# Summary
print(f"\n{'='*60}")
print("DATASET SUMMARY")
print("="*60)
print(f"Total games: {len(df):,}")
print(f"Date range: {df['game_date'].min().strftime('%Y-%m-%d')} to {df['game_date'].max().strftime('%Y-%m-%d')}")
print(f"Players: {df['player'].nunique()}")
print(f"Seasons: {df['season'].nunique() if 'season' in df.columns else 'N/A'}")
print(f"Columns: {len(df.columns)}")

# Data increase
print(f"\nData increase:")
print(f"  Before: 12,340 games")
print(f"  After: {len(df):,} games")
print(f"  Increase: {len(df)/12340:.1f}x more data")

print(f"\n{'='*60}")
print("NEXT STEP")
print("="*60)
print(f"Add matchup features:")
print(f"  python scripts/utils/add_matchup_features.py")
