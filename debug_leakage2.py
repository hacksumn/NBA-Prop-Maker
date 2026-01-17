"""
Deep dive into finding the leakage source
"""
import pandas as pd
import numpy as np
import pickle

# Load data
df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
df = df.sort_values(['player', 'game_date'])

# Load model
with open('model_pts.pkl', 'rb') as f:
    model_data = pickle.load(f)

features = model_data['features']

# Look at Trey Murphy III's recent games
player = 'Trey Murphy III'
player_df = df[df['player'] == player].tail(10)

print("="*80)
print(f"{player} - Last 10 games")
print("="*80)
print(player_df[['game_date', 'pts', 'pts_l5', 'pts_l10', 'games_played']].to_string())

# Check if there are any features that might be leaking
print("\n" + "="*80)
print("CHECKING FOR LEAKAGE IN ROLLING FEATURES")
print("="*80)

# For each game, check if pts_l10 includes the current game
for idx in range(len(player_df) - 1):
    current_row = player_df.iloc[idx]
    next_row = player_df.iloc[idx + 1]
    
    # The current row's pts should NOT be included in current row's pts_l10
    # But it SHOULD be included in the next row's pts_l10
    
    # Calculate what pts_l10 SHOULD be for next row
    # It should be the average of the last 10 games BEFORE next row
    # Which includes current row
    
    games_before_next = player_df.iloc[max(0, idx-8):idx+1]  # Last 10 games before next
    expected_pts_l10 = games_before_next['pts'].mean()
    actual_pts_l10 = next_row['pts_l10']
    
    diff = abs(expected_pts_l10 - actual_pts_l10)
    
    if diff > 0.1:
        print(f"\nGame {idx} -> {idx+1}:")
        print(f"  Current game date: {current_row['game_date'].date()}")
        print(f"  Next game date: {next_row['game_date'].date()}")
        print(f"  Expected pts_l10 for next game: {expected_pts_l10:.2f}")
        print(f"  Actual pts_l10 for next game: {actual_pts_l10:.2f}")
        print(f"  Difference: {diff:.2f}")

# Now check if the rolling features are INCLUDING the current game
print("\n" + "="*80)
print("CHECKING IF ROLLING FEATURES INCLUDE CURRENT GAME")
print("="*80)

# Get the latest game
latest = player_df.iloc[-1]
print(f"\nLatest game: {latest['game_date'].date()}")
print(f"Actual PTS: {latest['pts']}")
print(f"pts_l5: {latest['pts_l5']}")
print(f"pts_l10: {latest['pts_l10']}")

# Calculate what pts_l5 SHOULD be (average of previous 5 games, NOT including current)
prev_5_games = player_df.iloc[-6:-1]  # 5 games before the latest
expected_pts_l5 = prev_5_games['pts'].mean()
print(f"\nExpected pts_l5 (avg of 5 games BEFORE latest): {expected_pts_l5:.2f}")
print(f"Actual pts_l5: {latest['pts_l5']:.2f}")
print(f"Difference: {abs(expected_pts_l5 - latest['pts_l5']):.2f}")

# Calculate what pts_l10 SHOULD be
prev_10_games = player_df.iloc[-11:-1]  # 10 games before the latest
expected_pts_l10 = prev_10_games['pts'].mean()
print(f"\nExpected pts_l10 (avg of 10 games BEFORE latest): {expected_pts_l10:.2f}")
print(f"Actual pts_l10: {latest['pts_l10']:.2f}")
print(f"Difference: {abs(expected_pts_l10 - latest['pts_l10']):.2f}")

# Check if pts_l5 includes the current game
# If it does, it would be closer to including the current 35 points
prev_5_with_current = player_df.iloc[-5:]['pts'].mean()
print(f"\nIf pts_l5 INCLUDED current game: {prev_5_with_current:.2f}")

if abs(latest['pts_l5'] - prev_5_with_current) < 0.1:
    print("⚠️ LEAKAGE DETECTED! pts_l5 includes the current game!")
elif abs(latest['pts_l5'] - expected_pts_l5) < 0.1:
    print("✓ No leakage in pts_l5 - correctly excludes current game")
else:
    print("? Unclear - pts_l5 doesn't match either expected value")

# Check other features that might be causing the issue
print("\n" + "="*80)
print("CHECKING OTHER FEATURES")
print("="*80)

# Check if there are any features with suspicious values
suspicious_features = []
for feat in features:
    if feat in latest.index:
        val = latest[feat]
        if pd.notna(val) and abs(val) > 100:
            suspicious_features.append((feat, val))

if suspicious_features:
    print("\nFeatures with very high values:")
    for feat, val in suspicious_features[:10]:
        print(f"  {feat}: {val}")

# Check for features that might have current game info
print("\nFeatures in model:")
print(f"Total: {len(features)}")
print("\nFeature categories:")
rolling_feats = [f for f in features if '_l5' in f or '_l10' in f or '_l20' in f]
opp_feats = [f for f in features if 'opp_' in f]
schedule_feats = [f for f in features if any(x in f for x in ['days_rest', 'b2b', 'games_last', 'mp_l3', 'high_min'])]
other_feats = [f for f in features if f not in rolling_feats and f not in opp_feats and f not in schedule_feats]

print(f"  Rolling features: {len(rolling_feats)}")
print(f"  Opponent features: {len(opp_feats)}")
print(f"  Schedule features: {len(schedule_feats)}")
print(f"  Other features: {len(other_feats)}")
print(f"\nOther features: {other_feats}")
